import json
import networkx as nx
import os
import pickle
import sys
from networkx.algorithms.dag import dag_longest_path
from parseargs import parse_args
from rdkit.Chem import Descriptors, MolFromSmiles
from tqdm import tqdm

sys.path.append("../..")
from desp.search.desp_search import DespSearch
from desp.inference.retro_predictor import RetroPredictor
from desp.inference.syn_dist_predictor import SynDistPredictor
from desp.inference.retro_value import ValuePredictor
from desp.inference.forward_predictor import ForwardPredictor


def zero(smiles_1, smiles_2):
    return 0


def predict_one(target, starting):
    searcher = DespSearch(
        target,
        [starting],
        retro_predictor,
        fwd_predictor,
        building_blocks,
        strategy=args.strategy,
        heuristic_fn=value_predictor.predict,
        distance_fn=distance_fn,
        iteration_limit=500,
        top_m=25,
        top_k=2,
        max_depth_top=21,
        max_depth_bot=11,
        stop_on_first_solution=True,
        must_use_sm=True,
        retro_only=False if args.strategy in ["f2e", "f2f"] else True,
    )
    print(f"Starting search towards {target} from {starting}")
    result = searcher.run_search()
    print(f"Result for {target} from {starting}: {result}")
    return target, starting, result, searcher.search_graph


if __name__ == "__main__":
    args = parse_args()

    # Load retro predictor
    retro_predictor = RetroPredictor(
        model_path=args.retro_model, templates_path=args.retro_templates
    )

    # Load building blocks
    with open(args.bb_mol2idx, "r") as f:
        building_blocks = json.load(f)

    if args.strategy in ["f2e", "f2f"]:
        # Load fwd predictor
        fwd_predictor = ForwardPredictor(
            forward_model_path=args.fwd_model,
            templates_path=args.fwd_templates,
            bb_model_path=args.bb_model,
            bb_tensor_path=args.bb_tensor,
            bb_mol2idx=building_blocks,
            device=args.device,
        )
    else:
        fwd_predictor = None

    # Load synthetic distance and value models
    device = args.device if args.strategy == "f2f" else "cpu"
    sd_predictor = SynDistPredictor(args.sd_model, device)
    value_predictor = ValuePredictor(args.value_model)

    # Load test set
    targets = []
    with open(args.test_path, "r") as f:
        for line in f:
            target = eval(line)
            targets.append(target)

    # Use synthetic distance if DESP
    if args.strategy == "f2f":
        distance_fn = sd_predictor.predict_batch
    elif args.strategy in ["f2e", "retro_sd"]:
        distance_fn = sd_predictor.predict
    else:
        distance_fn = zero

    results = []
    graphs = []

    # Construct the directory path
    directory = os.path.join(args.test_set)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{args.strategy}")

    with open(file_path + ".txt", "a") as f:
        for target, starting in tqdm(targets):
            target, starting, result, graph = predict_one(target, starting)
            results.append((target, starting, result))
            graphs.append(graph)
            f.write(f"('{target}', '{starting}', {result})\n")
            f.flush()

    # Save graphs to pickle file
    with open(file_path + ".pkl", "wb") as f:
        pickle.dump(graphs, f)
