import torch
import torch.nn as nn

supp_act = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "swish": nn.SiLU(),
}


class Dense(nn.Module):
    """
    Dense layer with activation function.

    Args:
        in_features (int): input feature size
        out_features (int): output feature size
        hidden_act (nn.Module): activation function (e.g. nn.ReLU())
    """

    def __init__(self, in_features, out_features, hidden_act):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.hidden_act = hidden_act

    def forward(self, x):
        return self.hidden_act(self.linear(x))


class FwdTemplRel(nn.Module):
    """
    Forward template relevance model. On input it takes two concatenated
    fingerprints (reactant + target) and outputs the probability of each
    template.

    Args:
        fp_size (int): fingerprint size
        output_dim (int): output dimension
        hidden_sizes (list): list of hidden layer sizes (e.g. [1024, 1024])
        hidden_activation (nn.Module): activation function (e.g. nn.ReLU())
        dropout (float): dropout probability (e.g. 0.3)
    """

    def __init__(self, args):
        super().__init__()
        self.hidden_sizes = [int(size) for size in args.hidden_sizes.split(",")]
        self.model_type = args.model_type
        if self.model_type == "templ_rel":
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
            input_dim = args.fp_size * 2
        elif self.model_type == "bb":
            self.criterion = nn.BCEWithLogitsLoss(
                reduction="mean", pos_weight=torch.tensor([args.pos_weight])
            )
            input_dim = args.fp_size * 3
        else:
            raise ValueError(f"Unsupported model type: {args.model}")
        self.hidden_activation = supp_act[args.hidden_activation]
        self.layers = self._build_layers(
            input_dim, self.hidden_sizes, self.hidden_activation
        )
        self.output_layer = nn.Linear(self.hidden_sizes[-1], args.output_dim, bias=True)
        self.dropout = nn.Dropout(args.dropout)

    def _build_layers(self, fp_size, hidden_sizes, hidden_activation):
        layers = nn.ModuleList(
            [Dense(fp_size, hidden_sizes[0], hidden_act=hidden_activation)]
        )

        for layer_i in range(len(hidden_sizes) - 1):
            in_features = hidden_sizes[layer_i]
            out_features = hidden_sizes[layer_i + 1]
            layer = Dense(in_features, out_features, hidden_act=hidden_activation)
            layers.append(layer)

        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        return self.output_layer(x)

    def get_loss(self, logits, target, train=True):
        if self.model_type == "bb":
            target = target.squeeze().float()
            loss = self.criterion(logits, target)
            activated = torch.sigmoid(logits).round()
            accs = (
                (activated.float() == target).all(dim=1)
            ).float()  # exact match accuracy
            accs = accs.mean()
            cos = torch.nn.functional.cosine_similarity(activated, target, dim=1).mean()
            return loss, accs, cos
        else:
            loss = self.criterion(logits, target)
            # Get top1 accuracy
            pred_1 = torch.argmax(logits, dim=1)
            top1acc = (pred_1 == target).float().mean()

            if train:
                return loss, top1acc

            _, pred_5 = logits.topk(5, dim=1)
            _, pred_10 = logits.topk(10, dim=1)
            _, pred_25 = logits.topk(25, dim=1)
            # Get top5 accuracy
            correct_top5 = torch.tensor(
                [label in preds for label, preds in zip(target, pred_5)]
            )
            top5acc = correct_top5.float().mean()
            # Get top10 accuracy
            correct_top10 = torch.tensor(
                [label in preds for label, preds in zip(target, pred_10)]
            )
            top10acc = correct_top10.float().mean()
            # Get top25 accuracy
            correct_top25 = torch.tensor(
                [label in preds for label, preds in zip(target, pred_25)]
            )
            top25acc = correct_top25.float().mean()
            return loss, (top1acc, top5acc, top10acc, top25acc)
