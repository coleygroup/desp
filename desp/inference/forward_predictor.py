import json
import torch
import torch.nn.functional as F
import numpy as np
from rdchiral.initialization import rdchiralReactants
from desp.inference.building_block_predictor import BuildingBlockPredictor
from desp.inference.utils import (
    template_to_fp,
    smiles_to_fp,
    run_unimolecular_reaction,
    run_bimolecular_reaction,
    get_valid_rxns,
)
from desp.inference.models.fwd_model.fwd_model import FwdTemplRel


class ForwardPredictor:
    """
    One-step forward predictor, which gives the highest scoring transformations
    for a given reactant and target molecule.
    """

    def __init__(
        self,
        forward_model_path,
        templates_path,
        bb_model_path,
        bb_tensor_path,
        bb_mol2idx,
        device="cpu",
    ):
        """
        Args:
            path_forward_template_model (str): path to a trained forward template model
            templates_path (str): path to the list of templates
            bb_model_path (str): path to a trained building block model
            bb_tensor_path (str): path to the tensor of building blocks
            bb_mol2idx (dict): dict mapping building blocks to index
            device (str): device to run the KNN on
        """
        self.device = device

        # Load the forward templates
        with open(templates_path, "r") as f:
            template_dict = json.load(f)
        self.templates = {}
        for k, v in template_dict.items():
            self.templates[int(k)] = v

        # Load the forward model
        fwd_checkpoint = torch.load(forward_model_path, map_location="cpu")
        pretrain_args = fwd_checkpoint["args"]
        pretrain_args.model_type = "templ_rel"
        pretrain_args.output_dim = len(self.templates)
        self.forward_model = FwdTemplRel(pretrain_args)
        state_dict = fwd_checkpoint["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.forward_model.load_state_dict(state_dict)
        self.forward_model.eval()
        print("Loaded forward model!")

        # Initialize the building block predictor
        self.bb_predictor = BuildingBlockPredictor(
            bb_model_path, bb_tensor_path, bb_mol2idx, device=self.device
        )
        print("Loaded building block predictor!")

    def predict(self, reactant, target, top_n=50, top_k=10):
        """
        Predicts the most likely viable transformations for a given reactant and target molecule.

        Args:
            reactant (str): SMILES string of the reactant
            target (str): SMILES string of the target
            top_n (int): number of top templates to consider
            top_k (int): number of top building blocks to consider for each bimolecular reaction

        Returns:
            predictions (list): list of dictionaries of the form
                {
                    "product": SMILES string of the product,
                    "score": score of the reaction,
                    "template": template used,
                    "rxn_smiles": reaction SMILES,
                    "building_block": building block used (if bimolecular)
                }
        """
        reactant_fp = smiles_to_fp(reactant)
        target_fp = smiles_to_fp(target)

        reactant_rd = rdchiralReactants(reactant)

        # Concatenate reactant and target
        reactant_target_fp = torch.concat((reactant_fp, target_fp)).float().unsqueeze(0)
        with torch.no_grad():
            out = self.forward_model(reactant_target_fp)
            probs = F.softmax(out, dim=1)

        probs, indices = torch.topk(probs, top_n)
        probs = probs.detach().numpy()[0]
        indices = indices.detach().numpy()[0]

        # Get the top-k templates from indices
        templates = [self.templates[i] for i in indices]

        predictions = []
        prod_to_score = {}
        bimolecular_fps = []
        bimolecular_scores = []
        bimolecular_templates = []

        for i in range(len(templates)):
            template = templates[i]
            retro_template = template.split(">>")[1] + ">>" + template.split(">>")[0]
            current_products = {}

            reactants = template.split(">>")[0].split(".")
            if len(reactants) == 1:
                # Try running the reaction and append if successful
                output = run_unimolecular_reaction(reactant_rd, template)
                for product in output:
                    valid = get_valid_rxns(product, reactant, None, retro_template)
                    for p, r, bb in valid:
                        predictions.append(
                            {
                                "product": p,
                                "score": probs[i],
                                "template": template,
                                "rxn_smiles": r + ">>" + p,
                                "building_block": None,
                            }
                        )
                        current_products[p] = probs[i]
            elif len(reactants) >= 2:
                try:
                    template_fp = template_to_fp(template)
                except Exception as e:
                    print(f"Error {e} initializing template {template}")
                    continue
                full_fp = (
                    torch.concat((reactant_fp, target_fp, template_fp))
                    .float()
                    .unsqueeze(0)
                )
                bimolecular_fps.append(full_fp)
                bimolecular_scores.append(probs[i])
                bimolecular_templates.append(template)
            else:
                raise ValueError(
                    "Only unimolecular and bimolecular reactions supported"
                )
            for product, score in current_products.items():
                if product in prod_to_score:
                    prod_to_score[product] += score
                else:
                    prod_to_score[product] = score

        # Get top_k building blocks
        if len(bimolecular_fps) > 0:
            bimolecular_fps = torch.cat(bimolecular_fps, dim=0)
            with torch.no_grad():
                bb_smiles = self.bb_predictor.get_topk_bb(bimolecular_fps, top_k)
            # Try running the reaction for each building block and add successes
            for i in range(len(bb_smiles)):
                current_products = {}
                template = bimolecular_templates[i]
                retro_template = (
                    template.split(">>")[1] + ">>" + template.split(">>")[0]
                )
                for j in range(top_k):
                    output = run_bimolecular_reaction(
                        [reactant, bb_smiles[i][j]], template
                    )
                    if len(output) > 5:  # Skip if too many possible products
                        continue
                    for product in output:
                        valid = get_valid_rxns(
                            product,
                            reactant,
                            bb_smiles[i][j],
                            retro_template,
                        )
                        for p, r, bb in valid:
                            if bb is None:
                                predictions.append(
                                    {
                                        "product": p,
                                        "score": bimolecular_scores[i],
                                        "template": template,
                                        "rxn_smiles": r + ">>" + p,
                                        "building_block": None,
                                    }
                                )
                                current_products[p] = bimolecular_scores[i]
                            else:
                                predictions.append(
                                    {
                                        "product": p,
                                        "score": bimolecular_scores[i],
                                        "template": template,
                                        "rxn_smiles": r + "." + bb + ">>" + p,
                                        "building_block": bb,
                                    }
                                )
                                current_products[p] = bimolecular_scores[i]
                for product, score in current_products.items():
                    if product in prod_to_score:
                        prod_to_score[product] += score
                    else:
                        prod_to_score[product] = score

        # Renormalize scores
        total_score = sum(prod_to_score.values())
        for product in prod_to_score:
            prod_to_score[product] /= total_score
        # Readjust all prediction scores based on their product
        for i in range(len(predictions)):
            product = predictions[i]["product"]
            predictions[i]["score"] = prod_to_score[product]

        return predictions
