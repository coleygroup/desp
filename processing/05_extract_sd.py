import copy
import numpy as np
import pickle
import json
import random
import sys
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from scipy import sparse

from network import ReactionNetwork
from search import NetworkSearcher
from nodes import MolNode
from utils import smi_to_fp, tanimoto

if __name__ == "__main__":
    with open("data/filtered_train.jsonl", "r") as f:
        fwd_rxns = [json.loads(line) for line in f]
    with open("data/val_rxns_with_template.jsonl", "r") as f:
        val_rxns = [json.loads(line) for line in f]
    with open("data/building_blocks.pkl", "rb") as f:
        building_blocks = pickle.load(f)
    print("Loaded reaction set and building block set")

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

    # Get unbuyable nodes
    unbuyable_nodes = train_network.get_unbuyable(building_blocks)
    unbuyable_val = val_network.get_unbuyable(building_blocks)
    unbuyable_val = list(set(unbuyable_val) - set(unbuyable_nodes))

    """
    TRAINING DATA GENERATION
    """

    def extract_one(target, network):
        data = {}
        searcher = NetworkSearcher(target)
        graph, result = searcher.run_search(network, building_blocks)
        if result:
            for node in graph.nodes():
                if isinstance(node, MolNode) and node.solved and node != target:
                    distance = node.total_value - node.reaction_number
                    if (node.smiles, target) in data:
                        print(f"Somehow met {(node.smiles, target)} more than once")
                        assert (
                            node.smiles in building_blocks
                            or data[(node.smiles, target.smiles)] == distance
                        )
                    if not np.isinf(distance):
                        data[(node.smiles, target)] = distance
        else:
            search2 = NetworkSearcher(target)
            for node in graph.nodes():
                if (
                    isinstance(node, MolNode)
                    and node.smiles not in building_blocks
                    and node.smiles != target
                ):
                    search2.search_graph = copy.deepcopy(searcher.search_graph)
                    # Get node with no parents in search_graph
                    for sg_node in search2.search_graph:
                        if search2.search_graph.in_degree(sg_node) == 0:
                            sg_target = sg_node
                            break
                    assert sg_target.solved is False
                    for sg_node in search2.search_graph.nodes():
                        if sg_node.smiles == node.smiles:
                            ref = sg_node
                            break
                    ref.solved = True
                    ref.reaction_number = 0
                    ref.descendent_costs = {ref.__hash__(): 0}
                    search2.run_updates(list(search2.search_graph.predecessors(ref)))
                    if sg_target.solved:
                        distance = ref.total_value - ref.reaction_number
                        data[(ref.smiles, target)] = distance
        # Now sample a random node that cannot reach the target
        found = False
        while found is False:
            rand_node_smiles = random.sample(list(network.nodes()), 1)[0]
            if ">" in rand_node_smiles:
                continue
            # Make sure no node in route has the same smiles
            matches = False
            for node in graph.nodes():
                if node.smiles == rand_node_smiles:
                    matches = True
            # Also make sure the node is not too similar to the target
            if tanimoto(rand_node_smiles, target) < 0.5 and not matches:
                found = True
        data[(rand_node_smiles, target)] = 999
        return data

    extract_one_train = partial(extract_one, network=train_network)

    with Pool(16) as p:
        data_dicts = tqdm(
            p.map(extract_one_train, unbuyable_nodes), total=len(unbuyable_nodes)
        )
    data = {}
    for data_dict in data_dicts:
        data.update(data_dict)

    print(f"Got {len(data)} data points for training")

    FP_SIZE = 512
    fps = []
    values = []
    for sm, target in data:
        sm_fp = sparse.csr_matrix(smi_to_fp(sm, fp_size=FP_SIZE), dtype="int32")
        target_fp = sparse.csr_matrix(smi_to_fp(target, fp_size=FP_SIZE), dtype="int32")
        concat_fp = sparse.hstack([sm_fp, target_fp])
        fps.append(concat_fp)
        values.append(data[(sm, target)])

    fps = sparse.vstack(fps)
    values = np.array(values)

    print("Generated training data with shapes:")
    print("\tfps:", fps.shape)
    print("\tvalues:", values.shape)

    # Save the training data
    sparse.save_npz("data/sd_train_fp.npz", fps)
    np.save("data/sd_train_labels.npy", values)
    print("Saved training data!")

    """
    VALIDATION DATA GENERATION
    """
    extract_one_val = partial(extract_one, network=val_network)

    with Pool(16) as p:
        data_dicts = tqdm(
            p.map(extract_one_val, unbuyable_val), total=len(unbuyable_val)
        )
    data = {}
    for data_dict in data_dicts:
        data.update(data_dict)

    print(f"Got {len(data)} data points for validation")

    val_fps = []
    val_values = []
    for sm, target in data:
        sm_fp = sparse.csr_matrix(smi_to_fp(sm, fp_size=FP_SIZE), dtype="int32")
        target_fp = sparse.csr_matrix(smi_to_fp(target, fp_size=FP_SIZE), dtype="int32")
        concat_fp = sparse.hstack([sm_fp, target_fp])
        val_fps.append(concat_fp)
        val_values.append(data[(sm, target)])

    fps = sparse.vstack(val_fps)
    values = np.array(val_values)

    print("Generated validation data with shapes:")
    print("\tfps:", fps.shape)
    print("\tvalues:", values.shape)

    # Save the validation data
    sparse.save_npz("data/sd_val_fp.npz", fps)
    np.save("data/sd_val_labels.npy", values)
    print("Saved validation data!")
