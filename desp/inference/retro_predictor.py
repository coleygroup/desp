import json
import torch
import torch.nn.functional as F
from rdchiral.initialization import rdchiralReactants
from desp.inference.utils import smiles_to_fp, run_retro
from desp.inference.models.retro_mlp import TemplRel


class RetroPredictor(TemplRel):
    """
    One-step retro predictor, which gives the highest scoring transformations
    for a given target.
    """

    def __init__(self, model_path, templates_path):
        """
        Args:
            model_path (str): path to a trained model
            templates_path (str): path to the list of templates
        """
        # Load the templates
        with open(templates_path, "r") as f:
            template_dict = json.load(f)
        self.templates = {}
        for k, v in template_dict.items():
            self.templates[int(k)] = v

        # Load the model
        retro_checkpoint = torch.load(model_path, map_location="cpu")
        pretrain_args = retro_checkpoint["args"]
        super().__init__(pretrain_args)
        state_dict = retro_checkpoint["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)
        self.eval()
        print("Loaded retro model!")

    def predict(self, target, top_n=50):
        """
        Args:
            target (str): target molecule SMILES
            top_n (int): number of top scoring templates to return
        Returns:
            predictions (list): list of dictionaries of the format:
                {
                    "score" (float): softmax template score,
                    "template" (str): template SMILES string,
                    "reactants" (list): list of reactant SMILES strings
                }
        """
        # Convert the target SMILES to fingerprint
        target_fp = smiles_to_fp(target).float().unsqueeze(0)
        target_rd = rdchiralReactants(target)

        # Run the model
        with torch.no_grad():
            output = self(target_fp)
        probs = F.softmax(output, dim=1)
        top_scores, top_indices = torch.topk(probs, top_n)
        top_scores = top_scores.detach().numpy()[0]
        top_indices = top_indices.detach().numpy()[0]
        predictions = []
        for i in range(top_n):
            template = self.templates[top_indices[i]]
            try:
                pred_reactants = run_retro(target_rd, template)
                for output in pred_reactants:
                    predictions.append(
                        {
                            "rxn_smiles": ".".join(output) + ">>" + target,
                            "score": top_scores[i],
                            "template": template,
                            "reactants": output,
                        }
                    )
            except Exception as e:
                print(f"Issue applying template {template}:\n {e} \n target: {target}")
                continue

        # For each unique reactants, add their scores and templates together
        prec_to_score = {}
        prec_to_template = {}
        for i in range(len(predictions)):
            prec = frozenset(predictions[i]["reactants"])
            if prec in prec_to_score:
                prec_to_score[prec] += predictions[i]["score"]
                prec_to_template[prec].append(predictions[i]["template"])
            else:
                prec_to_score[prec] = predictions[i]["score"]
                prec_to_template[prec] = [predictions[i]["template"]]

        # Renormalize scores
        total_score = sum(prec_to_score.values())
        for prec in prec_to_score:
            prec_to_score[prec] /= total_score
        final_predictions = []
        for prec in prec_to_score:
            final_predictions.append(
                {
                    "rxn_smiles": ".".join(prec) + ">>" + target,
                    "score": prec_to_score[prec],
                    "template": prec_to_template[prec],
                    "reactants": list(prec),
                }
            )

        return final_predictions
