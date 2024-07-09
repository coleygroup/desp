"""
Adapted from ASKCOSv2 template relevance trainer:
https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance/-/blob/main/templ_rel_trainer.py?ref_type=heads
"""

import argparse
import misc
import numpy as np
import os
import arg_parser
import matplotlib.pyplot as plt
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import utils
import wandb
from dataset import FingerprintDataset, init_loader
from datetime import datetime
from tqdm import tqdm
from typing import Any, Dict, Tuple


def get_optimizer_and_scheduler(args, model: nn.Module, state: Dict[str, Any]) -> Tuple:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",  # monitor top-1 val accuracy
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        cooldown=args.lr_cooldown,
        verbose=True,
    )

    if state and args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        misc.log_rank_0("Loaded pretrained optimizer and scheduler state_dicts.")

    return optimizer, scheduler


def _optimize(args, model: nn.Module, optimizer) -> float:
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    optimizer.step()
    g_norm = utils.grad_norm(model)
    model.zero_grad(set_to_none=True)

    return g_norm


class SynDistTrainer:
    """Class for Synthetic Distance Training"""

    def __init__(self, args):
        self.args = args

        self.model_name = args.model_name
        self.data_name = args.data_name
        self.log_file = args.log_file
        self.processed_data_path = args.processed_data_path
        self.model_path = args.model_path
        self.num_cores = args.num_cores

        self.model = None
        self.state = {}
        self.device = args.device

    def build_train_model(self) -> None:
        model, state = utils.get_model(self.args, self.device)
        misc.log_rank_0(model)
        misc.log_rank_0(f"Number of parameters = {utils.param_count(model)}")
        self.model = model
        self.state = state

    def train(self) -> None:
        # init optimizer and scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.args, self.model, self.state
        )

        # init datasets and loaders
        train_dataset = FingerprintDataset(
            fp_file=os.path.join(self.processed_data_path, "sd_train_fp.npz"),
            label_file=os.path.join(self.processed_data_path, "sd_train_labels.npy"),
        )
        val_dataset = FingerprintDataset(
            fp_file=os.path.join(self.processed_data_path, "sd_val_fp.npz"),
            label_file=os.path.join(self.processed_data_path, "sd_val_labels.npy"),
        )

        train_loader = init_loader(
            args=self.args,
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        val_loader = init_loader(
            args=self.args,
            dataset=val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        # final init
        min_val_loss = np.inf
        max_r_squared = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        r_2s = []
        start = time.time()
        misc.log_rank_0("Start training")
        for epoch in range(self.args.epochs):
            # training loop
            losses = []
            train_loader = tqdm(train_loader, desc="training")
            self.model.train()
            self.model.zero_grad(set_to_none=True)

            for data in train_loader:
                inputs, labels = data
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)

                logits = self.model(inputs).squeeze()
                loss = self.model.get_loss(logits=logits, target=labels)
                loss.backward()
                losses.append(loss.item())
                # accs.append(acc.item())

                _optimize(self.args, self.model, optimizer)

                train_loss = np.mean(losses)
                # train_acc = np.mean(accs)

                train_loader.set_description(f"training loss: {train_loss:.4f}, ")
                train_loader.refresh()

            misc.log_rank_0(
                f"End of epoch {epoch}, "
                f"training loss: {train_loss:.4f}, "
                # f"top-1 acc: {train_acc:.4f},"
                f"p_norm: {utils.param_norm(self.model):.4f}, "
                f"g_norm: {utils.grad_norm(self.model):.4f}, "
                f"lr: {utils.get_lr(optimizer):.6f}, "
                f"elapsed time: {time.time() - start:.0f}"
            )
            train_losses.append(train_loss)
            # train_accs.append(train_acc)

            # validation loop (end of each epoch)
            self.model.eval()
            losses = []
            val_loader = tqdm(val_loader, desc="validation")
            with torch.no_grad():
                for _, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs = inputs.to(self.device).float()
                    labels = labels.to(self.device)

                    logits = self.model(inputs).squeeze()
                    loss = self.model.get_loss(logits=logits, target=labels)
                    losses.append(loss.item())
                    # accs.append(acc.item())

                    val_loss = np.mean(losses)
                    # val_acc = np.mean(accs)

                    val_loader.set_description(f"validation loss: {val_loss:.4f}, ")
                    val_loader.refresh()
            # Save image of violin plot where x-axis is labels and y-axis is logits
            logits = []
            labels = []
            for _, data in enumerate(val_loader):
                inputs, label = data
                inputs = inputs.to(self.device).float()
                label = label.to(self.device)
                logit = self.model(inputs).squeeze()
                logits.append(logit)
                labels.append(label)
            logits = torch.cat(logits).cpu().detach().numpy()
            labels = torch.cat(labels).cpu().detach().numpy()
            # cap labels at args.max_label + 1
            if self.args.model_type == "dist":
                labels = np.minimum(labels, self.args.max_label + 1)
                max_label = self.args.max_label + 2
            else:
                max_label = np.max(labels) + 1
            nans = [float("nan"), float("nan")]
            plt.violinplot(
                [
                    logits[labels == i] if logits[labels == i].any() else nans
                    for i in range(1, max_label)
                ]
            )

            plt.xlabel("labels")
            plt.ylabel("logits")
            r_squared = np.corrcoef(labels, logits)[0, 1] ** 2
            plt.title(f"r^2: {r_squared}")
            if r_squared > max_r_squared:
                plt.savefig(os.path.join(self.model_path, "scatter.png"))
            plt.close()
            r_2s.append(r_squared)

            misc.log_rank_0(
                f"End of epoch {epoch}, "
                f"validation loss: {np.mean(val_losses)}, "
                # f"acc: {np.mean(val_accs): .4f},"
                f"current r_squared: {r_squared: .4f},"
                f"current val loss: {val_loss:.4f}, "
                f"p_norm: {utils.param_norm(self.model): .4f}, "
                f"g_norm: {utils.grad_norm(self.model): .4f}, "
                f"lr: {utils.get_lr(optimizer): .6f}, "
                f"elapsed time: {time.time() - start: .0f}\n"
            )
            val_losses.append(val_loss)
            wandb.log(
                {"val_loss": val_loss, "train_loss": train_loss, "r_squared": r_squared}
            )
            # val_accs.append(val_acc)
            self.model.train()

            # scheduler step
            scheduler.step(val_loss)
            misc.log_rank_0(
                f"Called a step of ReduceLROnPlateau,"
                f"current lr: {utils.get_lr(optimizer)}"
            )
            # Important: save only at one node for DDP or the ckpt would be corrupted!
            if dist.is_initialized() and dist.get_rank() > 0:
                continue

            # saving
            if r_squared > max_r_squared:
                misc.log_rank_0(f"Saving at the end of epoch {epoch}")
                state = {
                    "args": args,
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "r_2s": r_2s,
                    # "train_accs": train_accs,
                    "val_losses": val_losses,
                    # "val_accs": val_accs,
                    "min_val_loss": min_val_loss,
                }
                torch.save(
                    state,
                    os.path.join(self.model_path, "model_latest_sd.pt"),
                )

            # early stopping
            if (
                self.args.early_stop
                and max_r_squared - r_squared >= self.args.early_stop_min_delta
            ):
                if patience_counter >= self.args.early_stop_patience:
                    misc.log_rank_0(f"Early stopped at the end of epoch: {epoch}")
                    break
                else:
                    patience_counter += 1
                    misc.log_rank_0(
                        f"Increase in val acc < early stop min delta "
                        f"{self.args.early_stop_min_delta}\n"
                        f"patience count: {patience_counter}."
                    )
            else:
                patience_counter = 0
                min_val_loss = min(min_val_loss, val_loss)
                max_r_squared = max(max_r_squared, r_squared)

            # legacy, forcing synchronization
            if args.local_rank != -1:
                dist.barrier()


def train_main(args):
    args.device = (
        torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )

    with wandb.init(project="", config=args):
        config = wandb.config
        utils.init_dist(args)
        misc.log_args(args, message="Logging training args")

        os.makedirs(args.model_path, exist_ok=True)
        trainer = SynDistTrainer(config)
        trainer.build_train_model()
        trainer.train()


if __name__ == "__main__":
    wandb.login()

    parser = argparse.ArgumentParser("template_relevance")
    arg_parser.add_model_opts(parser)
    arg_parser.add_train_opts(parser)
    arg_parser.add_predict_opts(parser)
    args, unknown = parser.parse_known_args()

    # logger setup
    os.makedirs("./logs/train", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/train/{args.log_file}.{dt}.log"
    logger = misc.setup_logger(args.log_file)

    utils.set_seed(args.seed)

    train_main(args)
