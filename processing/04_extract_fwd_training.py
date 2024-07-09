import pickle
import json
import numpy as np

from tqdm import tqdm
from scipy import sparse
from nodes import MolNode, RxnNode
from network import ReactionNetwork
from search import NetworkSearcher
from utils import clear_atom_map, smi_to_fp, template_to_fp


if __name__ == "__main__":
    with open("data/filtered_fwd_train.jsonl", "r") as f:
        fwd_rxns = [json.loads(line) for line in f]
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

    # Get nodes that have at least one incoming edge
    nodes = []
    for node in train_network.nodes:
        if (
            len(list(train_network.predecessors(node))) > 0
            and train_network.nodes[node]["label"] == "molecule"
        ):
            nodes.append(node)

    reactions = []
    for target in tqdm(nodes):
        searcher = NetworkSearcher(target)
        graph, result = searcher.run_search(train_network, building_blocks)
        for node in graph.nodes:
            if isinstance(node, RxnNode):
                rxn_smi = node.smiles
                retro_template = node.template
                fwd_template = (
                    retro_template.split(">>")[1] + ">>" + retro_template.split(">>")[0]
                )
                reactions.append(
                    {
                        "rxn_smiles": rxn_smi,
                        "template": fwd_template,
                        "target": target,
                    }
                )
    dedup_rxns = [dict(t) for t in {tuple(d.items()) for d in reactions}]
    print(f"Extracted {len(dedup_rxns)} reactions from network")

    # Remove reactions between two non-buyables
    fp_matrix = []
    labels_temp = []
    fp_matrix_bb = []
    labels_bb = []
    fwd_templ_to_idx = {}
    rxn_count = {}
    for rxn in tqdm(dedup_rxns):
        rxn_smiles = rxn["rxn_smiles"]
        template = rxn["template"]
        target = rxn["target"]
        reactants = rxn_smiles.split(">")[0].split(".")
        assert len(reactants) <= 2
        if len(reactants) == 2:
            reactant1 = clear_atom_map(reactants[0])
            reactant2 = clear_atom_map(reactants[1])
            if reactant1 not in building_blocks and reactant2 not in building_blocks:
                continue
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
        else:
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
    sparse.save_npz("data/fwd_train_fp.npz", fp_matrix)
    np.save("data/fwd_train_labels.npy", labels_temp)
    # Save final BB training set
    sparse.save_npz("data/fwd_train_fp_bb.npz", fp_matrix_bb)
    sparse.save_npz("data/fwd_train_labels_bb.npz", labels_bb)
    # Save template to index mapping
    with open("data/fwd_templ_to_idx_v2.json", "w") as f:
        json.dump(fwd_templ_to_idx, f)
