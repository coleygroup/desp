from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions as Reactions
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
import torch


def smiles_to_fp(smiles, fp_size=2048):
    """
    Convert a SMILES string to a fingerprint.
    Args:
        smiles (str): SMILES string
    Returns:
        np.array: fingerprint of the SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=fp_size, useChirality=True
    )
    fp = torch.tensor(fp, dtype=torch.uint8)
    return fp


def template_to_fp(template):
    """
    Convert a template string to a fingerprint.
    Args:
        template (str): template
    Returns:
        np.array: fingerprint of the template
    """
    template_rxn = AllChem.ReactionFromSmarts(template)
    fp_type = Reactions.FingerprintType.names["AtomPairFP"]
    args = [False, 0.2, 10, 1, 2048, fp_type]
    params = Reactions.ReactionFingerprintParams(*args)
    template_fp = Reactions.CreateStructuralFingerprintForReaction(template_rxn, params)
    template_fp = torch.tensor(template_fp)
    return template_fp


def run_retro(product, template):
    """
    Run a reaction given the product and the template.
    Args:
        product (str): product
        template (str): template
    Returns:
        str: reactant SMILES string
    """
    reactants = template.split(">>")[0].split(".")
    if len(reactants) > 1:
        template = "(" + template.replace(">>", ")>>")
    template = rdchiralReaction(template)
    try:
        outputs = rdchiralRun(template, product)
    except Exception as e:
        print(f"Error {e} running retro reaction {template} on product {product}")
        return []
    result = []
    for output in outputs:
        result.append(output.split("."))
    return result


def run_unimolecular_reaction(reactant, template):
    """
    Run a reaction given the reactant and the template.
    Args:
        reactant (str): reactant
        template (str): template
    Returns:
        str: product SMILES string
    """
    template = "(" + template.replace(">>", ")>>")
    template = rdchiralReaction(template)
    outputs = rdchiralRun(template, reactant)
    result = []
    for output in outputs:
        if len(output.split(".")) == 1:  # should only be 1 product
            result.append(output)
    return result


def is_reactant_first(reactant, template):
    """
    Check if `reactant` is the first reactant in a bimolecular template.
    Args:
        reactant (Chem.Mol): reactant
        template (str): template
    Returns:
        bool: whether `reactant` is the first reactant
    """
    first_reactant = template.split(">>")[0].split(".")[0]
    pattern = Chem.MolFromSmarts(first_reactant)
    return reactant.HasSubstructMatch(pattern)


def is_reactant_second(reactant, template):
    """
    Check if `reactant` is the second reactant in bimolecular template.
    Args:
        reactant (Chem.Mol): reactant
        template (str): template
    Returns:
        bool: whether `reactant` is the second reactant
    """
    second_reactant = template.split(">>")[0].split(".")[1]
    pattern = Chem.MolFromSmarts(second_reactant)
    return reactant.HasSubstructMatch(pattern)


def flatten_output(outputs):
    """
    Postprocess the output of a reaction to remove duplicates and invalid SMILES.
    Args:
        outputs (list): list of products
    Returns:
        list: list of deduplicated valid products
    """
    products = []
    for product in outputs:
        if len(product) == 1:
            smiles = Chem.MolToSmiles(product[0])
            try:
                Chem.CanonSmiles(smiles)
                products.append(smiles)
            except Exception as e:
                print(e)
                pass
        else:
            # print("More than one product")
            pass
    return list(set(products))


def run_bimolecular_reaction(reactants, template):
    """
    Run a reaction with rdchiral given two reactants and the template.
    Args:
        reactants (list): list of reactants
        template (str): template
    Returns:
        str: product SMILES string
    """
    if len(reactants) != 2:
        raise ValueError("Bimolecular reaction requires two reactants!")
    # template = "(" + template.replace(">>", ")>>")
    reactants = rdchiralReactants(".".join(reactants))
    template = "(" + template.replace(">>", ")>>")
    try:
        template = rdchiralReaction(template)
    except Exception as e:
        print(f"Error {e} initializing template {template}")
        return []
    outputs = rdchiralRun(template, reactants)
    result = []
    for output in outputs:
        if len(output.split(".")) == 1:  # should only be 1 product
            result.append(output)
    return result


def get_valid_rxns(product, reactant, building_block, retro_template):
    product_rd = rdchiralReactants(product)
    retro_result = run_retro(product_rd, retro_template)
    # print("getting vlid rxns")
    # print(product)
    # print(retro_template)
    # print(retro_result)
    valid_reactions = []
    for result in retro_result:
        if result == [reactant, building_block] or result == [building_block, reactant]:
            valid_reactions.append((product, reactant, building_block))
        elif result == [reactant]:
            valid_reactions.append((product, reactant, None))
    return valid_reactions
