import copy
import pickle
import json
import numpy as np
import os

from tqdm import tqdm
from scipy import sparse
from nodes import MolNode, RxnNode
from network import ReactionNetwork
from search import NetworkSearcher
from utils import clear_atom_map, smi_to_fp, template_to_fp


if __name__ == "__main__":
    with open("data/filtered_fwd_train.jsonl", "r") as f:
        fwd_rxns = [json.loads(line) for line in f]
    with open("data/val_rxns_with_template.jsonl", "r") as f:
        val_rxns = [json.loads(line) for line in f]
    with open("data/building_blocks.pkl", "rb") as f:
        building_blocks = pickle.load(f)
    print("Loaded fwd reaction set and building block set")

    # Populate the reaction network with loaded reactions
    train_network = ReactionNetwork()
    num_nodes, num_edges = train_network.populate_with_templates(fwd_rxns)
    print(
        "Populated training network with {} nodes and {} edges".format(
            num_nodes, num_edges
        )
    )
    val_network = copy.deepcopy(train_network)
    num_nodes, num_edges = val_network.populate_with_templates(val_rxns)
    print(
        "Populated validation network with {} nodes and {} edges".format(
            num_nodes, num_edges
        )
    )

    # Get nodes that have at least one incoming edge
    train_nodes = []
    for node in train_network.nodes:
        if (
            len(list(train_network.predecessors(node))) > 0
            and train_network.nodes[node]["label"] == "molecule"
        ):
            train_nodes.append(node)
    val_nodes = []
    for node in val_network.nodes:
        if (
            len(list(val_network.predecessors(node))) > 0
            and val_network.nodes[node]["label"] == "molecule"
        ):
            val_nodes.append(node)

    train_ex = []
    for target in tqdm(train_nodes):
        searcher = NetworkSearcher(target)
        graph, result = searcher.run_search(train_network, building_blocks)
        for node in graph.nodes:
            if isinstance(node, RxnNode):
                rxn_smi = node.smiles
                retro_template = node.template
                fwd_template = (
                    retro_template.split(">>")[1] + ">>" + retro_template.split(">>")[0]
                )
                train_ex.append(
                    {
                        "rxn_smiles": rxn_smi,
                        "template": fwd_template,
                        "target": target,
                    }
                )
    dedup_train_rxns = [dict(t) for t in {tuple(d.items()) for d in train_ex}]
    print(f"Extracted {len(dedup_train_rxns)} reactions from train network")

    val_ex = []
    for target in tqdm(val_nodes):
        searcher = NetworkSearcher(target)
        graph, result = searcher.run_search(val_network, building_blocks)
        for node in graph.nodes:
            if isinstance(node, RxnNode):
                rxn_smi = node.smiles
                retro_template = node.template
                if (
                    ">>" in retro_template
                ):  # some reactions may not have properly extracted template
                    fwd_template = (
                        retro_template.split(">>")[1]
                        + ">>"
                        + retro_template.split(">>")[0]
                    )
                    val_ex.append(
                        {
                            "rxn_smiles": rxn_smi,
                            "template": fwd_template,
                            "target": target,
                        }
                    )
    dedup_val_rxns = [dict(t) for t in {tuple(d.items()) for d in val_ex}]
    # Only keep reactions in val_rxns
    val_rxn_smiles = {}
    for rxn in val_rxns:
        rxn_smiles = rxn["rxn_smiles"]
        if rxn_smiles in val_rxn_smiles:
            val_rxn_smiles[rxn_smiles] += 1
        else:
            val_rxn_smiles[rxn_smiles] = 1
    dedup_val_rxns = [
        rxn for rxn in dedup_val_rxns if rxn["rxn_smiles"] in val_rxn_smiles
    ]

    """
    TRAINING DATA GENERATION
    """
    # Remove reactions between two non-buyables
    fp_matrix = []
    labels_temp = []
    fp_matrix_bb = []
    labels_bb = []
    fwd_templ_to_idx = {}
    rxn_count = {}
    for rxn in tqdm(dedup_train_rxns):
        rxn_smiles = rxn["rxn_smiles"]
        template = rxn["template"]
        target = rxn["target"]
        reactants = rxn_smiles.split(">")[0].split(".")
        assert len(reactants) <= 2
        if len(reactants) == 2:
            reactant1 = clear_atom_map(reactants[0])
            reactant2 = clear_atom_map(reactants[1])
            # Case 1: Both reactants are not building blocks, skip
            if reactant1 not in building_blocks and reactant2 not in building_blocks:
                continue
            # Case 2: Reactant 2 is a building block, add examples for Reactant 1 expansion
            if reactant2 in building_blocks:
                reactant_fp = sparse.csr_matrix(smi_to_fp(reactant1), dtype="int32")
                target_fp = sparse.csr_matrix(smi_to_fp(target), dtype="int32")
                template_fp = template_to_fp(template)
                concat_fp = sparse.hstack([reactant_fp, target_fp])
                bb_concat_fp = sparse.hstack([concat_fp, template_fp])
                fp_matrix.append(concat_fp)
                fp_matrix_bb.append(bb_concat_fp)
                if template not in fwd_templ_to_idx:
                    fwd_templ_to_idx[template] = len(fwd_templ_to_idx)
                labels_temp.append(fwd_templ_to_idx[template])
                labels_bb.append(
                    sparse.csr_matrix(smi_to_fp(reactant2, fp_size=256), dtype="int32")
                )
            # Case 3 (not mutually exclusive): Reactant 1 is a building block, add examples for Reactant 2 expansion
            if reactant1 in building_blocks:
                reactant_fp = sparse.csr_matrix(smi_to_fp(reactant2), dtype="int32")
                target_fp = sparse.csr_matrix(smi_to_fp(target), dtype="int32")
                template_fp = template_to_fp(template)
                concat_fp = sparse.hstack([reactant_fp, target_fp])
                bb_concat_fp = sparse.hstack([concat_fp, template_fp])
                fp_matrix.append(concat_fp)
                fp_matrix_bb.append(bb_concat_fp)
                if template not in fwd_templ_to_idx:
                    fwd_templ_to_idx[template] = len(fwd_templ_to_idx)
                labels_temp.append(fwd_templ_to_idx[template])
                labels_bb.append(
                    sparse.csr_matrix(smi_to_fp(reactant1, fp_size=256), dtype="int32")
                )
        else:  # Unimolecular, only add example for fwd template model
            reactant_fp = sparse.csr_matrix(smi_to_fp(reactants[0]), dtype="int32")
            target_fp = sparse.csr_matrix(smi_to_fp(target), dtype="int32")
            concat_fp = sparse.hstack([reactant_fp, target_fp])
            fp_matrix.append(concat_fp)
            if template not in fwd_templ_to_idx:
                fwd_templ_to_idx[template] = len(fwd_templ_to_idx)
            labels_temp.append(fwd_templ_to_idx[template])
        if rxn_smiles in rxn_count:
            rxn_count[rxn_smiles] += 1
        else:
            rxn_count[rxn_smiles] = 1
    fp_matrix = sparse.vstack(fp_matrix)
    labels_temp = np.array(labels_temp)
    fp_matrix_bb = sparse.vstack(fp_matrix_bb)
    labels_bb = sparse.vstack(labels_bb)

    print("Finished extracting fwd training set. Final stats for template relevance: ")
    print(f"\tnum rxn examples: {len(labels_temp)}")
    print(f"\tnum unique templates: {len(fwd_templ_to_idx)}")
    print(f"\tnum unique reactions: {len(rxn_count)}\n")
    print("Final stats for building block model: ")
    print(f"\tfp matrix shape: {fp_matrix_bb.shape}")
    print(f"\tbb labels shape: {labels_bb.shape}")

    print("Saving final fwd training set...")
    # Save final fwd training set
    sparse.save_npz("output/fwd_train_fp.npz", fp_matrix)
    np.save("output/fwd_train_labels.npy", labels_temp)
    # Save final BB training set
    sparse.save_npz("output/fwd_train_fp_bb.npz", fp_matrix_bb)
    sparse.save_npz("output/fwd_train_labels_bb.npz", labels_bb)
    # Save template to index mapping
    with open("output/fwd_templ_to_idx.json", "w") as f:
        json.dump(fwd_templ_to_idx, f)

    """
    VALIDATION DATA GENERATION
    """
    # Remove reactions between two non-buyables for validation
    fp_matrix = []
    labels_temp = []
    fp_matrix_bb = []
    labels_bb = []
    rxn_count = {}
    for rxn in tqdm(dedup_val_rxns):
        rxn_smiles = rxn["rxn_smiles"]
        template = rxn["template"]
        target = rxn["target"]
        reactants = rxn_smiles.split(">")[0].split(".")
        if len(reactants) > 2:
            continue
        elif len(reactants) == 2:
            reactant1 = clear_atom_map(reactants[0])
            reactant2 = clear_atom_map(reactants[1])
            # Case 1: Both reactants are not building blocks, skip
            if reactant1 not in building_blocks and reactant2 not in building_blocks:
                continue
            # Case 2: Reactant 2 is a building block, add examples for Reactant 1 expansion
            if reactant2 in building_blocks:
                reactant_fp = sparse.csr_matrix(smi_to_fp(reactant1), dtype="int32")
                target_fp = sparse.csr_matrix(smi_to_fp(target), dtype="int32")
                template_fp = template_to_fp(template)
                concat_fp = sparse.hstack([reactant_fp, target_fp])
                bb_concat_fp = sparse.hstack([concat_fp, template_fp])
                fp_matrix.append(concat_fp)
                fp_matrix_bb.append(bb_concat_fp)
                if template not in fwd_templ_to_idx:
                    labels_temp.append(-1)
                else:
                    labels_temp.append(fwd_templ_to_idx[template])
                labels_bb.append(
                    sparse.csr_matrix(smi_to_fp(reactant2, fp_size=256), dtype="int32")
                )
            # Case 3 (not mutually exclusive): Reactant 1 is a building block, add examples for Reactant 2 expansion
            if reactant1 in building_blocks:
                reactant_fp = sparse.csr_matrix(smi_to_fp(reactant2), dtype="int32")
                target_fp = sparse.csr_matrix(smi_to_fp(target), dtype="int32")
                template_fp = template_to_fp(template)
                concat_fp = sparse.hstack([reactant_fp, target_fp])
                bb_concat_fp = sparse.hstack([concat_fp, template_fp])
                fp_matrix.append(concat_fp)
                fp_matrix_bb.append(bb_concat_fp)
                if template not in fwd_templ_to_idx:
                    labels_temp.append(-1)
                else:
                    labels_temp.append(fwd_templ_to_idx[template])
                labels_bb.append(
                    sparse.csr_matrix(smi_to_fp(reactant1, fp_size=256), dtype="int32")
                )
        else:  # Unimolecular, only add example for fwd template model
            reactant_fp = sparse.csr_matrix(smi_to_fp(reactants[0]), dtype="int32")
            target_fp = sparse.csr_matrix(smi_to_fp(target), dtype="int32")
            concat_fp = sparse.hstack([reactant_fp, target_fp])
            fp_matrix.append(concat_fp)
            if template not in fwd_templ_to_idx:
                labels_temp.append(-1)
            else:
                labels_temp.append(fwd_templ_to_idx[template])
        if rxn_smiles in rxn_count:
            rxn_count[rxn_smiles] += 1
        else:
            rxn_count[rxn_smiles] = 1
    fp_matrix = sparse.vstack(fp_matrix)
    labels_temp = np.array(labels_temp)
    fp_matrix_bb = sparse.vstack(fp_matrix_bb)
    labels_bb = sparse.vstack(labels_bb)

    print(
        "Finished extracting fwd validation set. Final stats for template relevance: "
    )
    print(f"\tnum rxn examples: {len(labels_temp)}")
    print(f"\tnum unique templates: {len(fwd_templ_to_idx)}")
    print(f"\tnum unique reactions: {len(rxn_count)}\n")
    print("Final stats for building block model: ")
    print(f"\tfp matrix shape: {fp_matrix_bb.shape}")
    print(f"\tbb labels shape: {labels_bb.shape}")

    print("Saving final fwd validation set...")
    # Save final fwd training set
    sparse.save_npz("output/fwd_val_fp.npz", fp_matrix)
    np.save("output/fwd_val_labels.npy", labels_temp)
    # Save final BB training set
    sparse.save_npz("output/fwd_val_fp_bb.npz", fp_matrix_bb)
    sparse.save_npz("output/fwd_val_labels_bb.npz", labels_bb)
