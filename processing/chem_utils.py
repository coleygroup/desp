"""
A set of methods for manipulating and obtaining properties from SMILES strings/RDKit molecules.

Original authors: John Bradshaw, Jihye Roh
"""

import re
import collections
import itertools
import functools
import typing
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit.Chem import rdqueries
from rdkit.Chem import rdmolops
from rdkit.Chem.SaltRemover import SaltRemover

from functools import reduce
import numpy as np

smiles_tokenizer_pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|\%\([0-9]{3}\)|[0-9])"
# ^ added the 3 digit bond connection number
regex_smiles_tokenizer = re.compile(smiles_tokenizer_pattern)
salts = "[Cl,Br,I,Li,Na,K,Ca,Mg,O,N,Al,Fe,]"  # salts & metal ions to be removed from the product side

isCAtomsQuerier = rdqueries.AtomNumEqualsQueryAtom(6)


def num_heavy_atoms(iter_of_smiles):
    return sum(
        [Lipinski.HeavyAtomCount(Chem.MolFromSmiles(smi)) for smi in iter_of_smiles]
    )


def at_least_one_carbon(iter_of_smiles):
    return any(
        [
            len(Chem.MolFromSmiles(smi).GetAtomsMatchingQuery(isCAtomsQuerier)) >= 1
            for smi in iter_of_smiles
        ]
    )


def max_number_of_bonds_per_mol(iter_of_smiles):
    return max(
        [len(list(Chem.MolFromSmiles(smi).GetBonds())) for smi in iter_of_smiles]
    )


def num_unmapped_atoms(iter_of_smiles):
    return sum(
        [
            len(
                [
                    atom
                    for atom in mol.GetAtoms()
                    if not atom.HasProp("molAtomMapNumber")
                ]
            )
            for mol in (Chem.MolFromSmiles(smi) for smi in iter_of_smiles)
        ]
    )


def at_least_one_larger_product_different(
    reactant_frozenset, product_frozenset, canonicalize=True, num_heavy_atom_required=2
):
    """
    Check that at least one of the products is not in the reactant frozen set, *and* that this given product has a
    certain number of heavy atoms.

    nb this function is currently multiset naive
    """
    canon_op = try_to_canonicalize if canonicalize else lambda x: x
    reactants_set = {canon_op(el) for el in reactant_frozenset}
    for prod in product_frozenset:
        # ^ ignore count, i.e., second element -- see function docstring
        if prod not in reactants_set:
            mol = Chem.MolFromSmiles(prod)
            if mol:
                num_heavy_atoms = mol.GetNumHeavyAtoms()
                if num_heavy_atoms >= num_heavy_atom_required:
                    return True
    return False


def check_atom_identity(reactant_frozenset, product_frozenset):
    """
    Check that the atom with the same atom mapping number in the reactants and products is the same atom
    """
    reactant_atom_map = {}
    for reactant, _ in reactant_frozenset:
        mol = Chem.MolFromSmiles(reactant)
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum():
                reactant_atom_map[atom.GetAtomMapNum()] = atom.GetAtomicNum()

    product_atom_map = {}
    for product, _ in product_frozenset:
        mol = Chem.MolFromSmiles(product)
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum():
                product_atom_map[atom.GetAtomMapNum()] = atom.GetAtomicNum()

    for atom_map in product_atom_map:
        if reactant_atom_map[atom_map] != product_atom_map[atom_map]:
            return False

    return True


def not_deprotonation(reactant_frozenset, product_frozenset):
    """
    Checks the reaction is not a simple (de)protonation

    Do this by putting each molecule into a canonical neutralized form and then

    Note currently naive wrt multisets. Also note that have not exhaustively checked this function,
    expected to have false negatives, i.e., low recall.
    """
    neutralized_reactants = {try_neutralize_smi(smi) for smi in reactant_frozenset}
    neutralized_products = {try_neutralize_smi(smi) for smi in product_frozenset}
    if (len(neutralized_products) == 0) or (
        len(neutralized_products - neutralized_reactants) == 0
    ):
        return False
    else:
        return True


CHARGED_ATOM_PATTERN = Chem.MolFromSmarts(
    "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
)
# i.e. +1 charge, at least one hydrogen, and not linked to negative charged atom; or -1 charge and not linked to
# positive atom


def try_neutralize_smi(smi, canonical=True, log=None):
    mol = Chem.MolFromSmiles(smi)
    try:
        mol = neutralize_atoms(mol)
    except Exception as ex:
        err_str = f"Failed to neutralize {smi}"
        warnings.warn(err_str)

        # skipping for now, can check out a few of them and see
    else:
        smi = Chem.MolToSmiles(mol, canonical=canonical)
    return smi


def neutralize_atoms(mol):
    """
    from http://www.rdkit.org/docs/Cookbook.html
    note changed so that returns a RWCopy
    """
    mol = Chem.RWMol(mol)
    at_matches = mol.GetSubstructMatches(CHARGED_ATOM_PATTERN)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def remove_isotope_info_from_mol_in_place(mol):
    """
    adapted from https://www.rdkit.org/docs/Cookbook.html#isomeric-smiles-without-isotopes
    see limitations at link about needing to canonicalize _after_.
    """
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(0)
    return mol


def separate_and_clean_smiles(smiles_in, min_num_product_heavy_atoms=5):
    """
    Takes in an atom mapped reaction SMILES string, separates the reaction into single product reactions and cleans them
    Returns a list of cleaned SMILES strings
    """

    reactant, reagent, product = smiles_in.split(">")
    separated = []

    for p in product.split("."):
        try:
            cleaned = clean_smiles(
                reactant + ">" + reagent + ">" + p, min_num_product_heavy_atoms
            )
            separated.append(cleaned)
        except Exception as e:
            pass

    return separated


def clean_smiles(smiles_in, min_num_product_heavy_atoms=5):
    """ "
    Takes in an atom mapped reaction SMILES string and returns a new SMILES of the reaction
    with matched atom mapping style (i.e. only atoms in both reactants and products are mapped)

    With the following cleaning steps:
        1) Reagents moved to reactants
        2) Chemicals in both reactants and products (with same atom mapping) moved to reagents
        3) Product(s) removed if
            3-1) it does not contain any atom mapping
            3-2) the number of heavy atoms is less than min_num_product_heavy_atoms
        4) Remove atom mapping for atoms that are not in both reactants and products
        5) Reactants without atom mapping moved to reagents

    If no atom mapping exists, returns original smiles
    """

    smiles_in = smiles_in.split()[
        0
    ]  # some will have an end part, which we will ignore for now.

    if ":" not in smiles_in:  # TODO: may need to raise error
        return smiles_in

    # move reagents to reactants
    reactants, reagents, products = smiles_in.split(">")
    reactants = reactants + "." + reagents if reagents else reactants
    reactants_smi = reactants.split(".")
    products_smi = products.split(".")

    # move items in both sides to reagents
    common = set(reactants_smi) & set(products_smi)
    reactants_smi = [smi for smi in reactants_smi if smi not in common]
    products_smi = [smi for smi in products_smi if smi not in common]

    reagents = []
    for smi in common:
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        reagents.append(Chem.MolToSmiles(mol))

    reactant_mols = []

    # move unmapped reactants to reagents
    for reactant in reactants_smi:
        if ":" in reactant:
            reactant_mols.append(Chem.MolFromSmiles(reactant))
        else:
            reagents.append(reactant)

    # remove products not meeting the min_num_product_heavy_atoms requirement or not containing any atom mapping
    product_mols = [
        Chem.MolFromSmiles(product) for product in products_smi if ":" in product
    ]

    if min_num_product_heavy_atoms > 1:
        new_product_mols = [
            mol
            for mol in product_mols
            if mol.GetNumHeavyAtoms() >= min_num_product_heavy_atoms
        ]
    else:
        new_product_mols = product_mols

    # get atom mapping that are in both products and reactants
    product_mapping = [
        atom.GetAtomMapNum() for mol in new_product_mols for atom in mol.GetAtoms()
    ]
    reactant_mapping = [
        atom.GetAtomMapNum() for mol in reactant_mols for atom in mol.GetAtoms()
    ]

    matched_mapping_set = set(product_mapping) & set(reactant_mapping)

    # remove atom mapping if the atom mapping is not in the matched set
    new_products_smi = []
    for mol in new_product_mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() not in matched_mapping_set:
                atom.SetAtomMapNum(0)
        smi = Chem.MolToSmiles(mol)

        if ":" in smi:
            new_products_smi.append(smi)

    new_reactants_smi = []
    for mol in reactant_mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() not in matched_mapping_set:
                atom.SetAtomMapNum(0)

        smi = Chem.MolToSmiles(mol)

        if ":" in smi:
            new_reactants_smi.append(smi)
        else:
            reagents.append(smi)

    return (
        ".".join(sorted(new_reactants_smi))
        + ">"
        + ".".join(sorted(reagents))
        + ">"
        + ".".join(sorted(new_products_smi))
    )


def clean_smiles_reaction(smiles_in):
    """
    Creates a 2-element tuple of frozen sets of counters -- reactants and products.

    Does the following "cleaning" operations:
    - remove atom mapping
    - canonicalizes the molecules
    - puts the reagents with the reactants
    - splits the individual molecules up and puts them in counters
    """
    smiles_in = smiles_in.split()[
        0
    ]  # some will have an end part, which we will ignore for now.

    # TODO: add sanitization step here
    # smiles = clean_smiles(smiles_in)

    reactants, reagents, products = smiles_in.split(">")
    reactants = create_canon(reactants)
    products = create_canon(products)
    return frozenset(reactants), frozenset(products)


def create_canon(smiles_in):
    smiles_of_each_mol = smiles_in.split(".")
    canon_smiles = filter(len, map(try_to_canonicalize, smiles_of_each_mol))
    return canon_smiles


def canonicalize(
    smiles, remove_atm_mapping=True, remove_isotope_info=True, **otherargs
):
    mol = Chem.MolFromSmiles(smiles)
    if remove_isotope_info and mol is not None:
        mol = remove_isotope_info_from_mol_in_place(mol)
    return canonicalize_from_molecule(mol, remove_atm_mapping, **otherargs)


def try_to_canonicalize(smiles, *args, **kwargs):
    try:
        return canonicalize(smiles, *args, **kwargs)
    except Exception as ex:
        return smiles


def canonicalize_from_molecule(mol, remove_atm_mapping=True, **otherargs):
    mol_copy = Chem.RWMol(mol)
    if remove_atm_mapping:
        for atom in mol_copy.GetAtoms():
            atom.ClearProp("molAtomMapNumber")
    smiles = Chem.MolToSmiles(mol_copy, canonical=True, **otherargs)
    return smiles


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction

    from: https://github.com/pschwllr/MolecularTransformer
    """
    tokens = [token for token in regex_smiles_tokenizer.findall(smi)]
    and_back = "".join(tokens)
    if smi != and_back:
        raise RuntimeError(f"{smi} was tokenized incorrectly to {tokens}")
    return " ".join(tokens)


class InconsistentSMILES(RuntimeError):
    pass


def get_atom_map_nums(mol_str, accept_invalid=False) -> typing.Iterator[int]:
    """
    :return: iterator of the atom mapping numbers of the atoms in the reaction string
    """
    mol = Chem.MolFromSmiles(mol_str)
    if accept_invalid and mol is None:
        return []
    return (
        int(a.GetProp("molAtomMapNumber"))
        for a in mol.GetAtoms()
        if a.HasProp("molAtomMapNumber")
    )


def _not_unique_elements(itr_in) -> bool:
    lst_ = list(itr_in)
    return True if len(lst_) != len(set(lst_)) else False


def get_changed_bonds(
    rxn_smi, strict_mode=False
) -> typing.List[typing.Tuple[int, int, float]]:
    """
    Note unless `strict_mode` is `True`, this method does not check that the reagents have been seperated correctly
    (i.e., reagent atoms are not in the products) or that atom map numbers have been repeated in the reactants or
    products, which would break the rest of the implementation in a silent manner. If `strict_mode` is `True` and these
    conditions are violated then an `InconsistentSMILES` is raised.
    API should match the alternative:
    https://github.com/connorcoley/rexgen_direct/blob/master/rexgen_direct/scripts/prep_data.py
    :return: list of tuples of atom map numbers for each bond and new bond order
    """
    # 1. Split into reactants and products.
    reactants_smi, reagents_smi, products_smi = rxn_smi.split(">")

    # 2. If necessary check for repeated atom map numbers in the SMILES and products.
    if strict_mode:
        if _not_unique_elements(get_atom_map_nums(reactants_smi)):
            raise InconsistentSMILES("Repeated atom map numbers in reactants")
        if _not_unique_elements(get_atom_map_nums(products_smi)):
            raise InconsistentSMILES("Repeated atom maps numbers in products")

    # 3. Get the bonds and their types in reactants and products
    bonds_prev = {}
    bonds_new = {}
    for bond_dict, bonds in [
        (bonds_prev, Chem.MolFromSmiles(reactants_smi).GetBonds()),
        (bonds_new, Chem.MolFromSmiles(products_smi).GetBonds()),
    ]:
        for bond in bonds:
            try:
                bond_atmmap = frozenset(
                    (
                        int(bond.GetBeginAtom().GetProp("molAtomMapNumber")),
                        int(bond.GetEndAtom().GetProp("molAtomMapNumber")),
                    )
                )
            except KeyError:
                continue
            bond_dict[bond_atmmap] = float(bond.GetBondTypeAsDouble())

    # 4. Go through the bonds before and after...
    bond_changes: typing.List[typing.Tuple[int, int, float]] = []
    product_atmmap_nums = set(get_atom_map_nums(products_smi))
    if strict_mode and (
        len(set(get_atom_map_nums(reagents_smi)) & product_atmmap_nums) > 0
    ):
        raise InconsistentSMILES("Reagent atoms end up in products.")
    for bnd in {*bonds_prev, *bonds_new}:
        bnd_different_btwn_reacts_and_products = not (
            bonds_prev.get(bnd, None) == bonds_new.get(bnd, None)
        )
        bnd_missing_in_products = len(bnd & product_atmmap_nums) == 0

        # ... and if a bond has (a) changed or (b) is half missing in the products then it must have changed!
        if bnd_different_btwn_reacts_and_products and (not bnd_missing_in_products):
            bond_changes.append((*sorted(list(bnd)), bonds_new.get(bnd, 0.0)))
            # ^ note if no longer in products then new order is 0.

    return bond_changes


def split_reagents_out_from_reactants_and_products(
    rxn_smi, strict_mode=False
) -> typing.Tuple[str, str, str]:
    """
    Splits reaction into reactants, reagents, and products. Can deal with reagents in reactants part of SMILES string.
    Note that this method expects relatively well done atom mapping.
    Reagent defined as either:
    1. in the middle part of reaction SMILES string, i.e. inbetween the `>` tokens.
    2. in the reactants part of the SMILES string and all of these are true:
            a. no atoms in the product(s).
            b. not involved in the reaction center (atoms for which bonds change before and after) -- depending on the
                center identification code this will be covered by a, but is also checked to allow for cases where
                center can include information about changes in say a reactant that results in two undocumented minor
                products.
            c. reaction has been atom mapped (i.e., can we accurately check conditions a and b) -- currently judged by
                a center being able to be identified.
    3. in the reactants and products part of the SMILES string and both:
            a. not involved in reaction center
            b. unchanged before and after the reaction (comparing with canonicalized, atom-map removed strings)
    :param rxn_smi: the reaction SMILES string
    :param strict_mode: whether to run `get_changed_bonds` in strict mode when determining atom map numbers involved in
                        center.
    :return: tuple of reactants, reagents, and products
    """

    # 1. Split up reaction and get involved atom counts.
    reactant_all_str, reagents_all_str, product_all_str = rxn_smi.split(">")
    atoms_involved_in_reaction = set(
        itertools.chain(
            *[
                (int(el[0]), int(el[1]))
                for el in get_changed_bonds(rxn_smi, strict_mode)
            ]
        )
    )
    reactants_str = reactant_all_str.split(".")
    products_str = product_all_str.split(".")
    products_to_keep = collections.Counter(products_str)
    product_atom_map_nums = functools.reduce(
        lambda x, y: x | y, (set(get_atom_map_nums(prod)) for prod in products_str)
    )
    reaction_been_atom_mapped = len(atoms_involved_in_reaction) > 0

    # 2. Store map from canonical products to multiset of their SMILES in the products --> we will class
    canon_products_to_orig_prods = collections.defaultdict(collections.Counter)
    for smi in products_str:
        canon_products_to_orig_prods[canonicalize(smi)].update([smi])

    # 3. Go through the remaining reactants and check for conditions 2 or 3.
    reactants = []
    reagents = reagents_all_str.split(".")
    for candidate_reactant in reactants_str:
        atom_map_nums_in_candidate_reactant = set(get_atom_map_nums(candidate_reactant))

        # compute some flags useful for checks 2 and 3
        # 2a any atoms in products
        not_in_product = (
            len(list(product_atom_map_nums & atom_map_nums_in_candidate_reactant)) == 0
        )
        # 2b/3a any atoms in reaction center
        not_in_center = (
            len(
                list(
                    set(
                        atoms_involved_in_reaction & atom_map_nums_in_candidate_reactant
                    )
                )
            )
            == 0
        )

        # Check 2.
        if reaction_been_atom_mapped and not_in_product and not_in_center:
            reagents.append(candidate_reactant)
            continue

        # Check 3.
        canonical_reactant = canonicalize(candidate_reactant)
        reactant_possibly_unchanged_in_products = (
            canonical_reactant in canon_products_to_orig_prods
        )
        # ^ possibly as it could be different when we include atom maps -- we will check for this later.
        if not_in_center and reactant_possibly_unchanged_in_products:
            # We also need to match this reactant up with the appropriate product SMILES string and remove this from
            # the product.  To do this we shall go through the possible product SMILES strings.
            possible_prod = None
            for prod in canon_products_to_orig_prods[canonical_reactant]:
                # if the atom mapped numbers intersect then this must be the product we are after and can break!
                if (
                    len(
                        set(get_atom_map_nums(prod))
                        & set(get_atom_map_nums(candidate_reactant))
                    )
                    > 0
                ):
                    break

                # if the product in the reaction SMILES has no atom map numbers it _could_ match but check other
                # possibilities first to see if we get an atom map match.
                if len(list(get_atom_map_nums(prod))) == 0:
                    possible_prod = prod
            else:
                prod = possible_prod  # <-- if we are here then we did not get an exact atom map match

            if prod is not None:
                # ^ if it is still None then a false alarm and not the same molecule due to atom map numbers.
                # (we're going to defer to atom map numbers and assume they're right!)
                reagents.append(candidate_reactant)
                products_to_keep.subtract([prod])  # remove it from the products too

                # we also need to do some book keeping on our datastructure mapping canonical SMILES to product strings
                # to indicate that we have removed one.
                canon_products_to_orig_prods[canonical_reactant].subtract([prod])
                canon_products_to_orig_prods[canonical_reactant] += (
                    collections.Counter()
                )
                # ^ remove zero and negative values
                if len(canon_products_to_orig_prods[canonical_reactant]) == 0:
                    del canon_products_to_orig_prods[canonical_reactant]

                continue

        # if passed check 2 and 3 then it is a reactant!
        reactants.append(candidate_reactant)

    product_all_str = ".".join(products_to_keep.elements())
    return ".".join(reactants), ".".join(reagents), product_all_str
