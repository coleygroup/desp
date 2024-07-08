import torch
import numpy as np
from desp.inference.utils import smiles_to_fp
from desp.inference.models.syn_value.model import SyntheticDistance


class SynDistPredictor:
    """
    Predictor for synthetic distance between two molecules.
    """

    def __init__(self, syn_dist_model_path, device="cpu"):
        """
        Args:
            syn_dist_model_path (str): path to a trained synthetic distance model
        """

        self.device = device

        # Load the forward model
        checkpoint = torch.load(syn_dist_model_path, map_location="cpu")
        pretrain_args = checkpoint["args"]
        pretrain_args.output_dim = 1
        pretrain_args.model_type = "dist"
        self.model = SyntheticDistance(pretrain_args).to(self.device)
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Loaded synthetic distance model!")

    def predict(self, start, target):
        """
        Predict the synthetic distance from 'start' to 'target'.

        Args:
            start (str): start molecule SMILES
            target (str): target molecule SMILES

        Returns:
            float: synthetic distance
        """
        start_fp = smiles_to_fp(start, fp_size=512)
        target_fp = smiles_to_fp(target, fp_size=512)

        # Concatenate reactant and target
        reactant_target_fp = np.concatenate((start_fp, target_fp))
        reactant_target_fp = torch.from_numpy(reactant_target_fp).float().unsqueeze(0)
        with torch.no_grad():
            dist = self.model(reactant_target_fp)
        return dist.item()  # synthetic distance

    def predict_fp(self, fp):
        """
        Predict the synthetic distance from 'fp1' to 'fp2'.

        Args:
            fp1 (np.array): start molecule fp
            fp2 (np.array): target molecule fp

        Returns:
            float: synthetic distance
        """
        # Concatenate reactant and target
        reactant_target_fp = torch.from_numpy(fp).float().unsqueeze(0)
        with torch.no_grad():
            dist = self.model(reactant_target_fp)
        return dist.item()

    def predict_batch(self, starts, targets):
        """
        Predict the synthetic distance from each start to each target.

        Args:
            starts (list): list of start molecule fps
            targets (list): list of target molecule fps

        Returns:
            torch.Tensor: 2D tensor of synthetic distances
        """
        batch_size = 1024

        start_fps = torch.stack(starts)
        target_fps = torch.stack(targets)

        # Calculate total number of batches
        num_start_batches = (len(starts) + batch_size - 1) // batch_size
        num_target_batches = (len(targets) + batch_size - 1) // batch_size

        dists = torch.zeros(len(starts), len(targets))

        # Process in batches
        with torch.no_grad():
            for i in range(num_start_batches):
                start_idx_start = i * batch_size
                end_idx_start = min((i + 1) * batch_size, len(starts))
                batch_starts = start_fps[start_idx_start:end_idx_start]

                for j in range(num_target_batches):
                    start_idx_target = j * batch_size
                    end_idx_target = min((j + 1) * batch_size, len(targets))
                    batch_targets = target_fps[start_idx_target:end_idx_target]

                    start_expanded = batch_starts.unsqueeze(1).repeat(
                        1, end_idx_target - start_idx_target, 1
                    )
                    target_expanded = batch_targets.unsqueeze(0).repeat(
                        end_idx_start - start_idx_start, 1, 1
                    )

                    batch_dists = self.model(
                        torch.cat(
                            (
                                start_expanded.view(-1, start_fps.shape[1]),
                                target_expanded.view(-1, target_fps.shape[1]),
                            ),
                            dim=1,
                        ).float()
                    ).view(
                        end_idx_start - start_idx_start,
                        end_idx_target - start_idx_target,
                    )

                    dists[
                        start_idx_start:end_idx_start, start_idx_target:end_idx_target
                    ] = batch_dists

        return dists
