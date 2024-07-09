from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdchiral.main import rdchiralRunText
from tqdm import tqdm
from multiprocessing import Pool
import json
from utils import clear_atom_map


def test_example(train_ex):
    rxn_smiles = train_ex["rxn_smiles"].strip()
    reactants = rxn_smiles.split(">")[0].split(".")
    reactants = tuple([Chem.MolFromSmiles(clear_atom_map(r)) for r in reactants])
    products = rxn_smiles.split(">")[2]
    retro_template = train_ex["canon_reaction_smarts"]
    fwd_template = retro_template.split(">>")[1] + ">>" + retro_template.split(">>")[0]
    if len(reactants) < 3:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
            ran_products = list(rxn.RunReactants(reactants))
            ran_products = [
                clear_atom_map(Chem.MolToSmiles(p[0])) for p in ran_products
            ]
            if clear_atom_map(products) in ran_products:
                return {
                    "rxn_smiles": rxn_smiles,
                    "template": retro_template,
                    "error": "no error with rdkit",
                }
        except Exception as e:
            # print(e)
            pass
        try:
            reactants = rxn_smiles.split(">")[0].split(".")
            reactants = ".".join([clear_atom_map(r) for r in reactants])
            fwd_template = "(" + fwd_template.replace(">>", ")>>")
            ran_products = rdchiralRunText(fwd_template, reactants)
            if clear_atom_map(products) in [clear_atom_map(p) for p in ran_products]:
                return {
                    "rxn_smiles": rxn_smiles,
                    "template": retro_template,
                    "error": "no error with rdchiral",
                }
        except Exception as e:
            return {
                "rxn_smiles": rxn_smiles,
                "template": retro_template,
                "error": "could not run reaction",
            }
    else:
        return {
            "rxn_smiles": rxn_smiles,
            "template": retro_template,
            "error": "too many reactants",
        }
    return {
        "rxn_smiles": rxn_smiles,
        "template": retro_template,
        "error": "could not recover product",
    }


if __name__ == "__main__":
    with open("data/filtered_train.jsonl", "rb") as f:
        train = [json.loads(line) for line in f]

    failed = []

    with Pool(8) as p:
        failed = list(tqdm(p.imap(test_example, train), total=len(train)))

    with open("data/failed_fwd_rxns.jsonl", "w") as f:
        for failure in failed:
            f.write(json.dumps(failure) + "\n")
