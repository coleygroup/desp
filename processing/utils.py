import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdChemReactions as Reactions
from scipy import sparse


def clear_atom_map(smi):
    """
    Clear atom map numbers from and canonicalize a SMILES string.

    Args:
        smi (str): SMILES string to clear atom map numbers from.

    Returns:
        str: Canonicalized SMILES string with atom map numbers cleared
    """
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))


def smi_to_fp(mol_smi, radius=2, fp_size=2048, dtype="int32", as_numpy=True):
    """
    Convert a SMILES string to a Morgan fingerprint.

    Args:
        mol_smi (str): SMILES string to convert to fingerprint
        radius (int): Radius of Morgan fingerprint
        fp_size (int): Size of fingerprint
        dtype (str): Data type of fingerprint

    Returns:
        np.ndarray: Morgan fingerprint
    """
    mol = Chem.MolFromSmiles(mol_smi)
    fp_bit = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=fp_size, useChirality=True
    )
    if as_numpy:
        fp = np.empty((1, fp_size), dtype=dtype)
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
        return fp
    return fp_bit


def template_to_fp(template, fp_type="AtomPairFP", dtype="int32"):
    """
    Convert a template string to a fingerprint.
    Args:
        template (str): template
    Returns:
        sparse.csr_matrix: fingerprint of the template
    """
    template_rxn = AllChem.ReactionFromSmarts(template)
    fp_type = Reactions.FingerprintType.names[fp_type]
    args = [False, 0.2, 10, 1, 2048, fp_type]
    params = Reactions.ReactionFingerprintParams(*args)
    template_fp = Reactions.CreateStructuralFingerprintForReaction(template_rxn, params)
    template_fp = np.array(template_fp)
    return sparse.csr_matrix(template_fp, dtype=dtype)


def tanimoto(smi1, smi2):
    """
    Calculate the Tanimoto similarity between two SMILES strings.

    Args:
        smi1 (str): First SMILES string
        smi2 (str): Second SMILES string

    Returns:
        float: Tanimoto similarity between the two SMILES strings
    """
    fp1 = smi_to_fp(smi1, as_numpy=False)
    fp2 = smi_to_fp(smi2, as_numpy=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)
