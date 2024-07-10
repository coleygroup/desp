import copy
import numpy as np
import pickle
import json
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from scipy import sparse

from network import ReactionNetwork
from search import NetworkSearcher
from utils import smi_to_fp

if __name__ == "__main__":
    with open("data/filtered_train.jsonl", "r") as f:
        train_rxns = [json.loads(line) for line in f]
    with open("data/val_rxns_with_template.jsonl", "r") as f:
        val_rxns = [json.loads(line) for line in f]
    with open("data/building_blocks.pkl", "rb") as f:
        building_blocks = pickle.load(f)
    print("Loaded reaction set and building block set")

    # Populate the reaction network with loaded reactions
    train_network = ReactionNetwork()
    num_nodes, num_edges = train_network.populate_with_templates(train_rxns)
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
    FP_SIZE = 2048

    def extract_one(target, network):
        searcher = NetworkSearcher(target)
        _, result = searcher.run_search(network, building_blocks)
        if result:
            fp = smi_to_fp(target, fp_size=FP_SIZE)
            value = searcher.target.reaction_number
            return fp, value
        else:
            return None, -1

    extract_one_train = partial(extract_one, network=train_network)

    with Pool(16) as p:
        data_tups = tqdm(
            p.map(extract_one_train, unbuyable_nodes), total=len(unbuyable_nodes)
        )

    fps = []
    values = []
    for fp, value in data_tups:
        if fp is not None:
            target_fp = sparse.csr_matrix(fp, dtype="int32")
            fps.append(target_fp)
            values.append(value)

    fps = sparse.vstack(fps)
    values = np.array(values)

    print("Generated training data with shapes:")
    print("\tfps:", fps.shape)
    print("\tvalues:", values.shape)

    # Save the training data
    sparse.save_npz("output/ret_train_fp.npz", fps)
    np.save("output/ret_train_labels.npy", values)
    print("Saved training data!")

    """
    VALIDATION DATA GENERATION
    """
    extract_one_val = partial(extract_one, network=val_network)

    with Pool(16) as p:
        data_tups = tqdm(
            p.map(extract_one_val, unbuyable_val), total=len(unbuyable_val)
        )

    fps = []
    values = []
    for fp, value in data_tups:
        if fp is not None:
            target_fp = sparse.csr_matrix(fp, dtype="int32")
            fps.append(target_fp)
            values.append(value)

    fps = sparse.vstack(fps)
    values = np.array(values)

    print("Generated validation data with shapes:")
    print("\tfps:", fps.shape)
    print("\tvalues:", values.shape)

    # Save the validation data
    sparse.save_npz("output/ret_val_fp.npz", fps)
    np.save("output/ret_val_labels.npy", values)
    print("Saved validation data!")
