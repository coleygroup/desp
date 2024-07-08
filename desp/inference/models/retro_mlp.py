import torch
import torch.nn as nn
from typing import Tuple


def get_activation(name: str) -> nn.Module:
    _activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "leakyrelu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }

    return _activations[name]


class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_act: nn.Module):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.hidden_act = hidden_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden_act(self.linear(x))


class TemplRel(nn.Module):
    def __init__(self, args):
        super().__init__()

        if isinstance(args.hidden_sizes, str):
            self.hidden_sizes = [int(size) for size in args.hidden_sizes.split(",")]

        self.args = args
        self.layers = self._build_layers(args)
        print(self.layers)
        self.output_layer = nn.Linear(
            self.hidden_sizes[-1], args.n_templates, bias=True
        )

        # we will do all the dropout here in TemplRel, for backward compatibility
        self.dropout = nn.Dropout(args.dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    def _build_layers(self, args) -> nn.ModuleList:
        hidden_act = get_activation(args.hidden_activation)
        # input projection layer; no skip connection here
        layers = nn.ModuleList(
            [Dense(args.fp_size, self.hidden_sizes[0], hidden_act=hidden_act)]
        )

        for layer_i in range(len(self.hidden_sizes) - 1):
            in_features = self.hidden_sizes[layer_i]
            out_features = self.hidden_sizes[layer_i + 1]

            if args.skip_connection == "none":
                layer = Dense(in_features, out_features, hidden_act=hidden_act)
            else:
                raise ValueError(f"Unsupported skip_connection: {args.skip_connection}")

            layers.append(layer)

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        logits = self.output_layer(x)  # returning *unnormalized* logits

        return logits

    def get_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss = self.criterion(input=logits, target=target)

        predictions = torch.argmax(logits, dim=1)
        accs = (predictions == target).float()
        acc = accs.mean()

        # Data points with label -1 (no/failed templates) should really be treated as
        # if they are predicted wrong. Therefore, we turned off the masking. This may
        # noticeably decrease the computed accuracy, but would reflect the reality.

        # mask = (target != -1).long()
        # accs = accs * mask
        # acc = accs.sum() / mask.sum()

        return loss, acc
