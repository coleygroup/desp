import graphviz
import json
import pickle
import sys
import tempfile
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.append("..")
from desp.search.data_structures.nodes import (
    TopDownRxnNode,
    TopDownMolNode,
    BottomUpRxnNode,
)
from desp.search.desp_search import DespSearch
from desp.inference.retro_predictor import RetroPredictor
from desp.inference.syn_dist_predictor import SynDistPredictor
from desp.inference.retro_value import ValuePredictor
from desp.inference.forward_predictor import ForwardPredictor

retro_model = "./data/model_retro.pt"
retro_templates = "./data/idx2template_retro.json"
bb_mol2idx = "./data/canon_building_block_mol2idx_no_isotope.json"
fwd_model = "./data/model_fwd.pt"
fwd_templates = "./data/idx2template_fwd.json"
bb_model = "./data/model_bb.pt"
bb_tensor = "./data/building_block_fps.npz"
sd_model = "./data/syn_dist.pt"
value_model = "./data/retro_value.pt"
device = 0


def zero(smiles_1, smiles_2):
    return 0


class DESP:
    def __init__(self, strategy="f2e"):
        """
        Initialize the DESP class.

        Args:
            strategy (str, optional): The strategy for the search process. Defaults to "f2e",
                                      but also supports "f2f", "retro", "retro_sd", "bfs", "random"
        """
        self.strategy = strategy

        # Load retro predictor
        self.retro_predictor = RetroPredictor(
            model_path=retro_model, templates_path=retro_templates
        )

        # Load building blocks
        with open(bb_mol2idx, "r") as f:
            self.building_blocks = json.load(f)

        # Load fwd predictor
        if self.strategy in ["f2e", "f2f"]:
            self.fwd_predictor = ForwardPredictor(
                forward_model_path=fwd_model,
                templates_path=fwd_templates,
                bb_model_path=bb_model,
                bb_tensor_path=bb_tensor,
                bb_mol2idx=self.building_blocks,
                device=device,
            )
        else:
            self.fwd_predictor = None

        # Load synthetic distance and value models
        self.device = device if self.strategy == "f2f" else "cpu"
        self.sd_predictor = SynDistPredictor(sd_model, self.device)
        self.value_predictor = ValuePredictor(value_model)

        if self.strategy == "f2f":
            self.distance_fn = self.sd_predictor.predict_batch
        elif self.strategy in ["f2e", "retro_sd"]:
            self.distance_fn = self.sd_predictor.predict
        else:
            self.distance_fn = zero

    def search(
        self,
        target,
        starting,
        iteration_limit=500,
        top_n=50,
        top_m=25,
        top_k=2,
        max_depth_top=21,
        max_depth_bot=11,
        must_use_sm=True,
    ):
        """
        Perform a search to find a synthetic route for the target molecule from the given starting materials.

        Args:
            target (str): The SMILES string of the target molecule.
            starting (list): List of SMILES strings of the starting material(s).
            iteration_limit (int, optional): The maximum number of iterations for the search. Defaults to 500.
            top_n (int, optional): The number of top nodes to keep in the top-down search. Defaults to 50.
            top_m (int, optional): The number of top nodes to keep in the bottom-up search. Defaults to 25.
            top_k (int, optional): The number of top building blocks to consider for each bimolecular reaction. Defaults to 2.
            max_depth_top (int, optional): The maximum depth for the top-down search. Defaults to 21, which corresponds
                                           to a max depth of 11 molecule nodes.
            max_depth_bot (int, optional): The maximum depth for the bottom-up search. Defaults to 11, which corresponds
                                           to a max depth of 6 molecule nodes.
            must_use_sm (bool, optional): Whether the search must use the starting materials. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - bool: True if a solution was found, False otherwise.
                - dict or None: The synthetic route as a dictionary if a solution was found, None otherwise.
        """
        searcher = DespSearch(
            target,
            starting,
            self.retro_predictor,
            self.fwd_predictor,
            self.building_blocks,
            strategy=self.strategy,
            heuristic_fn=self.value_predictor.predict,
            distance_fn=self.distance_fn,
            iteration_limit=iteration_limit,
            top_n=top_n,
            top_m=top_m,
            top_k=top_k,
            max_depth_top=max_depth_top,
            max_depth_bot=max_depth_bot,
            stop_on_first_solution=True,
            must_use_sm=must_use_sm,
            retro_only=False if self.strategy in ["f2e", "f2f"] else True,
        )
        print(f"Starting search towards {target} from {starting}")
        result = searcher.run_search()
        print(f"Result for {target} from {starting}: {result}")
        if result[0] is True:
            route = self._extract_solved_route(
                searcher.search_graph.graph, searcher.search_graph.target_node
            )
        else:
            route = None
        return result, route, searcher

    def _helper_extract(self, node, graph):
        """
        Helper function to extract the synthetic route from the search graph.

        Args:
            node (TopDownMolNode or BottomUpRxnNode): The root node to extract the subroute from.
            graph (networkx.DiGraph): The original search graph.

        Returns:
            dict: The discovered subroute.
        """
        if len(list(graph.successors(node))) == 0:  # if leaf node
            route = {
                "smiles": node.smiles,
                "type": "mol",
                "orientation": "top" if isinstance(node, TopDownMolNode) else "bottom",
                "mol_type": "building",
            }
            return route
        else:
            solved_children = []
            for child in graph.successors(node):
                if isinstance(child, TopDownRxnNode) and child.desp_solved:
                    solved_children.append(child)
            if len(solved_children) == 0:
                for child in graph.successors(node):
                    if child.solved:
                        solved_children.append(child)
            best_child = min(solved_children, key=lambda x: x.total_value)
            children_routes = []
            for reactant in graph.successors(best_child):
                route = self._helper_extract(reactant, graph)
                children_routes.append(route)
            if isinstance(node, TopDownMolNode):
                node.is_building_block = node.inherently_solved
            route = {
                "smiles": node.smiles,
                "type": "mol",
                "mol_type": "intermediate"
                if not node.is_building_block
                else "building",
                "orientation": "top" if isinstance(node, TopDownMolNode) else "bottom",
                "children": [
                    {
                        "smiles": best_child.smiles,
                        "template": best_child.template,
                        "type": "reaction",
                        "orientation": "top"
                        if isinstance(best_child, TopDownRxnNode)
                        else "bottom",
                        "children": children_routes,
                    }
                ],
            }
            return route

    def _extract_solved_route(self, graph, target):
        """
        Extract a solved synthetic route from the search graph.

        Args:
            graph (networkx.DiGraph): The search graph.
            target (TopDownMolNode): The target node.

        Returns:
            dict: The synthetic route as a dictionary.
        """
        solved_children = []
        for child in graph.successors(target):
            if (
                isinstance(child, TopDownRxnNode)
                and child.desp_solved
                or isinstance(child, BottomUpRxnNode)
            ):
                solved_children.append(child)
        best_child = min(solved_children, key=lambda x: x.total_value)
        children_routes = []
        for reactant in graph.successors(best_child):
            route = self._helper_extract(reactant, graph)
            children_routes.append(route)
        route = {
            "smiles": target.smiles,
            "total_cost": target.total_value,
            "type": "mol",
            "mol_type": "target",
            "children": [
                {
                    "smiles": best_child.smiles,
                    "template": best_child.template,
                    "type": "reaction",
                    "orientation": "top"
                    if isinstance(best_child, TopDownRxnNode)
                    else "bottom",
                    "children": children_routes,
                }
            ],
        }
        return route

    def _draw_and_connect_children(
        self, parent_node, child, img_map, dot, temp_img_dir
    ):
        """
        Helper function to draw and connect the children nodes in the synthetic route visualization.

        Args:
            parent_node (str): The SMILES string of the parent node.
            child (dict): The dictionary representing the child node.
            img_map (dict): A dictionary to store the filepaths of the temporary molecule images.
            dot (graphviz.Digraph): The GraphViz object for rendering the visualization.
            temp_img_dir (str): The path to the temporary directory for storing molecule images.
        """
        child_node = child["smiles"]
        if child["type"] == "mol":
            mol = Chem.MolFromSmiles(child_node)
            escaped = child_node.replace("/", "_")
            file_path = f"{temp_img_dir}/{escaped}.png"
            Draw.MolToFile(mol, file_path, size=(200, 200))
            img_map[child_node] = file_path
            if child["mol_type"] == "starting":
                color = "plum1"
            elif (
                child["mol_type"] == "intermediate"
                and child["orientation"] == "top"
                and child_node not in self.building_blocks
            ):
                color = "royalblue"
            elif (
                child["mol_type"] == "intermediate"
                and child_node not in self.building_blocks
            ):
                color = "skyblue3"
            elif child["mol_type"] == "building" and child["orientation"] == "top":
                color = "springgreen4"
            else:
                color = "springgreen3"
            dot.node(
                child_node,
                label="",
                image=file_path,
                shape="box",
                color=color,
                penwidth="2",
            )
        elif child["type"] == "reaction":
            if child["orientation"] == "top":
                color = "lightgoldenrod1"
            else:
                color = "lightgoldenrod3"
            child_node = "rxn" + parent_node
            dot.node(
                child_node,
                label="",
                shape="box",
                style="rounded",
                color=color,
                penwidth="2",
            )
        else:
            raise TypeError("Child type not recognized")
        dot.edge(parent_node, child_node, color="darkgrey")
        if "children" in child:
            for child in child["children"]:
                self._draw_and_connect_children(
                    child_node, child, img_map, dot, temp_img_dir
                )

    def visualize_route(self, route, path):
        """
        Visualize the synthetic route and save the image to the specified path.

        Args:
            route (dict): The dictionary representing the synthetic route.
            path (str): The filename to save the visualization image (path + ".png")
        """
        dot = graphviz.Digraph(format="png")
        root_node = route["smiles"]
        mol = Chem.MolFromSmiles(root_node)
        img_map = {}
        with tempfile.TemporaryDirectory() as temp_img_dir:
            # Escape / characters in root_node
            escaped = root_node.replace("/", "_")
            file_path = f"{temp_img_dir}/{escaped}.png"
            Draw.MolToFile(mol, file_path, size=(200, 200))
            img_map[root_node] = file_path
            dot.node(
                root_node,
                label="",
                image=file_path,
                shape="rect",
                color="lightsalmon",
                penwidth="2",
            )
            for child in route["children"]:
                self._draw_and_connect_children(
                    root_node, child, img_map, dot, temp_img_dir
                )
            dot.render(path)
        return
