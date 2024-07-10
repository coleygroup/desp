import pandas as pd
import logging
from tqdm import tqdm
from rdkit import RDLogger
from multiprocessing import Pool
import chem_utils

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    filename="logs/USPTO_process.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)


def check_cleaned(cleaned_rxn):
    reactant_smi, prod_smi = cleaned_rxn
    if chem_utils.num_heavy_atoms(list(prod_smi)) < 5:
        logging.info(f"REACTION REMOVED: {cleaned_rxn}")
        logging.info(
            "REASON: Product has less than 5 heavy atoms\n--------------------"
        )
        return False
    if chem_utils.num_heavy_atoms(list(reactant_smi)) < 5:
        logging.info(f"REACTION REMOVED: {cleaned_rxn}")
        logging.info(
            "REASON: Reactant has less than 5 heavy atoms\n--------------------"
        )
        return False
    if not chem_utils.at_least_one_carbon(list(reactant_smi)):
        logging.info(f"REACTION REMOVED: {cleaned_rxn}")
        logging.info("REASON: Reactant has no carbon\n--------------------")
        return False
    if chem_utils.max_number_of_bonds_per_mol(list(reactant_smi)) < 2:
        logging.info(f"REACTION REMOVED: {cleaned_rxn}")
        logging.info("REASON: Reactant has less than 2 bonds\n--------------------")
        return False
    if not chem_utils.at_least_one_larger_product_different(
        reactant_smi, prod_smi, canonicalize=False, num_heavy_atom_required=2
    ):
        logging.info(f"REACTION REMOVED: {cleaned_rxn}")
        logging.info(
            "REASON: Needs to be at least one larger product that is not the same as the reactants\n--------------------"
        )
        return False
    if not chem_utils.not_deprotonation(reactant_smi, prod_smi):
        logging.info(f"REACTION REMOVED: {cleaned_rxn}")
        logging.info("REASON: Reaction is deprotonation\n--------------------")
        return False
    return True


def check_reaction(rxn):
    try:
        cleaned_rxn = chem_utils.clean_smiles_reaction(rxn)
        if check_cleaned(cleaned_rxn):
            return cleaned_rxn
        else:
            return -1
    except Exception as e:
        logging.info(f"REACTION REMOVED: {rxn}")
        logging.info(f"REASON: Exception: {e}\n--------------------")
        return -1


def process_and_check_one(rxn):
    cleaned = chem_utils.separate_and_clean_smiles(rxn)
    checks = {}
    for cleaned_rxn in cleaned:
        check = check_reaction(cleaned_rxn)
        checks[check] = cleaned_rxn
    return checks


if __name__ == "__main__":
    # Read 1976_Sep2016_USPTOgrants_smiles.rsmi
    df = pd.read_csv("data/1976_Sep2016_USPTOgrants_smiles.rsmi", sep="\t")
    # Convert ReactionSmiles column to list
    smiles = df["ReactionSmiles"].tolist()

    with Pool(20) as p:
        results = list(tqdm(p.imap(process_and_check_one, smiles), total=len(smiles)))

    print(f"Done processing {len(results)} reactions... flattening and deduplicating")
    deduplicated = {}
    for result in results:
        deduplicated.update(result)
    # Delete k, v with key = -1
    del deduplicated[-1]
    print(f"Done deduplicating! Now have {len(deduplicated)} reactions!")

    # Write deduplicated reactions to file
    with open("data/USPTO_processed_with_smiles.txt", "w") as f:
        for fs, rxn in deduplicated.items():
            f.write(f"{fs}\t{rxn}\n")
