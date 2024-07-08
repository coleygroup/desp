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


class CustomLoss(nn.Module):
    def __init__(self, max_label):
        super().__init__()
        self.max_label = max_label

    def forward(self, logits, target):
        # if target <= self.max_label, return MSELoss(logits, target)
        # otherwise, return (max(0, self.max_label - logits + 1))^2
        loss = torch.mean(
            torch.where(
                target <= self.max_label,
                (logits - target) ** 2,
                torch.max(torch.zeros_like(logits), self.max_label - logits + 1) ** 2,
            )
        )
        return loss


class SyntheticDistance(nn.Module):
    """
    Synthetic distance model. On input it takes either concatenated
    fingerprints (starting (+) target) or difference fingerprint (target - starting)
    and outputs the predicted distance.

    Args:
        input_type (str): input type (e.g. "concat", "diff")
        max_label (int): maximum label value
        fp_size (int): fingerprint size
        output_dim (int): output dimension
        hidden_sizes (str): string representnig list of hidden layer sizes (e.g. '1024,1024')
        hidden_activation (str): activation function (e.g. "relu")
        dropout (float): dropout probability (e.g. 0.3)
    """

    def __init__(self, args):
        super().__init__()
        self.hidden_sizes = [int(size) for size in args.hidden_sizes.split(",")]
        if args.model_type == "retro":
            self.criterion = nn.MSELoss()
        elif args.model_type == "dist":
            self.criterion = CustomLoss(args.max_label)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        if args.input_type == "concat":
            input_dim = args.fp_size * 2
        elif args.input_type == "diff":
            input_dim = args.fp_size
        else:
            raise ValueError(f"Unsupported input type: {args.input_type}")
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

    def get_loss(self, logits, target):
        loss = self.criterion(logits, target.float())
        return loss
