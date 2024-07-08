"""
Adapted from ASKCOSv2 template relevance module:
https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance/-/blob/main/utils.py?ref_type=heads
"""

import datetime
import json
import logging
import math
import misc
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from model import SyntheticDistance
from rdchiral.initialization import rdchiralReaction
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy import sparse
from torch.nn.init import xavier_uniform_
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Any, Dict, List, Tuple


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_dist(args):
    if args.local_rank != -1:
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            timeout=datetime.timedelta(0, 7200),
        )
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = False

    if dist.is_initialized():
        logging.info(f"Device rank: {dist.get_rank()}")
        sys.stdout.flush()


def canonicalize_smiles(smiles: str, remove_atom_number: bool = True) -> str:
    """Adapted from Molecular Transformer"""
    smiles = "".join(smiles.split())
    cano_smiles = ""

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        if remove_atom_number:
            [a.ClearProp("molAtomMapNumber") for a in mol.GetAtoms()]

        cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        # Sometimes stereochem takes another canonicalization... (just in case)
        mol = Chem.MolFromSmiles(cano_smiles)
        if mol is not None:
            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    return cano_smiles


def canonicalize_smarts(smarts: str) -> str:
    templ = Chem.MolFromSmarts(smarts)
    if templ is None:
        logging.info(f"Could not parse {smarts}")
        return smarts

    canon_smarts = Chem.MolToSmarts(templ)
    if "[[se]]" in canon_smarts:  # strange parse error
        canon_smarts = smarts

    return canon_smarts


def mol_smi_to_count_fp_min_htoo(
    mol_smi: str, radius: int = 2, fp_size: int = 2048, dtype: str = "int32"
) -> sparse.csr_matrix:
    fp_gen = GetMorganGenerator(
        radius=radius, countSimulation=True, includeChirality=True, fpSize=fp_size
    )
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)

    return sparse.csr_matrix(count_fp, dtype=dtype)


def mol_smi_to_count_fp(
    mol_smi: str, radius: int = 2, fp_size: int = 2048, dtype: str = "int32"
) -> sparse.csr_matrix:
    mol = Chem.MolFromSmiles(mol_smi)
    fp_bit = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=fp_size, useChirality=True
    )
    fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(fp_bit, fp)

    return sparse.csr_matrix(fp, dtype=dtype)


def get_model(args, device) -> Tuple[nn.Module, Dict[str, Any]]:
    state = {}
    print(args)
    if args.load_from:
        misc.log_rank_0(f"Loading pretrained state from {args.load_from}")
        state = torch.load(args.load_from, map_location=torch.device("cpu"))
        pretrain_args = state["args"]
        misc.log_args(pretrain_args, message="Logging pretraining args")

        model = SyntheticDistance(pretrain_args)
        pretrain_state_dict = state["state_dict"]
        pretrain_state_dict = {
            k.replace("module.", ""): v for k, v in pretrain_state_dict.items()
        }
        model.load_state_dict(pretrain_state_dict)
        misc.log_rank_0("Loaded pretrained model state_dict.")

        # Overwrite for preprocessing during (likely) inference
        args.fp_size = pretrain_args.fp_size
        args.radius = pretrain_args.radius
    else:
        args.output_dim = 1
        model = SyntheticDistance(args)
        for p in model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)

    model.to(device)
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        misc.log_rank_0("DDP setup finished")

    return model, state


def save_templates_from_list(templates: List[Dict[str, Any]], template_file: str):
    with open(template_file, "w") as of:
        for template in templates:
            if "rxn" in template:
                del template["rxn"]
            of.write(f"{json.dumps(template)}\n")


def save_templates_from_dict(templates: Dict[str, Dict[str, Any]], template_file: str):
    with open(template_file, "w") as of:
        for canon_templ, metadata in templates.items():
            assert metadata["reaction_smarts"] == canon_templ
            if "rxn" in metadata:
                del metadata["rxn"]
            of.write(f"{json.dumps(metadata)}\n")


def load_templates_as_list(
    template_file: str,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    templates = []
    template_attributes = []

    with open(template_file, "r") as f:
        for line in f:
            template = json.loads(line.strip())
            del template["references"]

            templates.append(template)
            template_attributes.append(template.get("attributes", {}))
    template_attributes = pd.DataFrame(template_attributes)

    return templates, template_attributes


def load_templates_as_dict(template_file: str) -> Dict[str, Dict[str, Any]]:
    templates = {}

    with open(template_file, "r") as f:
        for line in f:
            template = json.loads(line.strip())
            templates[template["reaction_smarts"]] = template

    return templates


def get_lr(optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def param_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def param_norm(m: nn.Module) -> float:
    return math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))


def grad_norm(m: nn.Module) -> float:
    return math.sqrt(
        sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None])
    )
