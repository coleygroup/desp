import torch
from desp.inference.utils import smiles_to_fp
from desp.inference.models.syn_value.model import SyntheticDistance as RetroValue


class ValuePredictor:
    """
    Predictor for number of steps to synthesize a molecule.
    """

    def __init__(self, value_model_path, device="cpu"):
        """
        Args:
            value_model_path (str): path to a trained value model
        """

        self.device = device

        # Load the forward model
        checkpoint = torch.load(value_model_path, map_location="cpu")
        pretrain_args = checkpoint["args"]
        pretrain_args.output_dim = 1
        self.model = RetroValue(pretrain_args).to(self.device)
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Loaded retro value model!")

    def predict(self, target):
        """
        Predict the synthetic cost of 'target'.

        Args:
            target (str): target molecule SMILES

        Returns:
            float: synthetic distance
        """
        target_fp = smiles_to_fp(target, fp_size=2048).float().unsqueeze(0)
        with torch.no_grad():
            dist = self.model(target_fp)
        return dist.item()
