import math
import numpy as np
import torch
from tqdm import tqdm
from desp.search.data_structures.nodes import (
    TopDownMolNode,
    BottomUpMolNode,
    zero,
    zero_single,
)
from desp.search.data_structures.desp_tree import DespTree


class DespSearch:
    def __init__(
        self,
        target,
        starting_materials,
        retro_model,
        fwd_model,
        building_blocks,
        strategy="f2e",
        distance_fn=zero,
        heuristic_fn=zero_single,
        top_n=50,
        top_m=25,
        top_k=2,
        iteration_limit=500,
        time_limit=None,
        max_depth_top=50,
        max_depth_bot=50,
        stop_on_first_solution=True,
        must_use_sm=True,
        retro_only=False,
        bottom_only=False,
    ):
        self.search_graph = DespTree(
            target,
            starting_materials,
            strategy,
            heuristic_fn,
            distance_fn,
            building_blocks,
            must_use_sm,
        )
        self.strategy = strategy
        self.retro_model = retro_model
        self.fwd_model = fwd_model
        self.top_n = top_n
        self.top_m = top_m
        self.top_k = top_k
        self.iteration_limit = iteration_limit
        self.time_limit = time_limit
        self.max_depth_top = max_depth_top
        self.max_depth_bot = max_depth_bot
        self.stop_on_first_solution = stop_on_first_solution
        self.must_use_sm = must_use_sm
        assert not (retro_only and bottom_only)
        self.retro_only = retro_only
        self.bottom_only = bottom_only

    def can_expand_retro(self, node):
        """
        Checks if a node can be expanded. A node can be expanded if:
            1. Node is a TopDownMolNode
            2. Node has not reached the max depth
            3. Node has not been expanded yet

        Args:
            node (TopDownMolNode): Node to check

        Returns:
            bool: True if node can be expanded, False otherwise
        """
        return (
            isinstance(node, TopDownMolNode)
            and node.depth < self.max_depth_top
            and not node.is_expanded
            and not node.inherently_solved
        )

    def can_expand_fwd(self, node):
        """ """
        return (
            isinstance(node, BottomUpMolNode)
            and node.depth < self.max_depth_bot
            and not node.is_expanded
        )

    def expand_retro(self, node, graph):
        """
        Expands a node by predicting the next set of nodes using the reaction model.
        The predictions are then added to the graph, and the graph is updated.

        Args:
            node (TopDownMolNode): Node to expand
            graph (AndOrGraph): Graph to update

        Returns:
            list: List of new nodes added to the graph
        """
        assert self.can_expand_retro(node)
        predictions = self.retro_model.predict(node.smiles, top_n=self.top_n)
        new_nodes, met = graph.expand_top(predictions, node)
        updated_bot = set()
        for new_node in new_nodes:
            if self.can_expand_retro(new_node):
                self.search_graph.open_nodes_top.add(new_node)
        self.search_graph.open_nodes_top.remove(node)
        self.search_graph.expanded_nodes_top.add(node)
        if self.strategy == "f2f":
            updated_bot = graph.f2f_update_top(new_nodes)
        _ = graph.run_updates(new_nodes + [node], met | updated_bot, "top_down")
        return new_nodes

    def expand_fwd(self, node, graph):
        """"""
        assert self.can_expand_fwd(node)
        predictions = self.fwd_model.predict(
            node.smiles, node.closest_node.smiles, top_n=self.top_m, top_k=self.top_k
        )
        new_nodes, met = graph.expand_bot(predictions, node)
        updated_top = set()
        for new_node in new_nodes:
            if self.can_expand_fwd(new_node):
                self.search_graph.open_nodes_bot.add(new_node)
        self.search_graph.open_nodes_bot.remove(node)
        self.search_graph.expanded_nodes_bot.add(node)
        if self.strategy == "f2f":
            updated_top = graph.f2f_update_bot(new_nodes)
        _ = graph.run_updates(new_nodes + [node], met | updated_top, "bottom_up")
        return new_nodes

    def run_search(self):
        """
        Runs the top-down search algorithm:
            1. Pop the frontier node with the highest priority
            2. If the node has not been expanded yet, expand it
            3. Add the new nodes to frontier set
            4. Repeat until solved the iteration limit is reached

        Returns:
            bool: True if the target is found, False otherwise
            int: Number of iterations run
        """

        num_iterations = 0
        i = 0
        pbar = tqdm(total=self.iteration_limit)
        while num_iterations < self.iteration_limit:
            torch.cuda.empty_cache()
            if (
                (self.stop_on_first_solution and self.search_graph.solved)
                or (math.isinf(self.search_graph.target_node.reaction_number))
                or (self.bottom_only and len(self.search_graph.open_nodes_bot) == 0)
            ):
                print(f"Search finished with solved = {self.search_graph.solved}")
                break
            if (
                (i + 1) % 3 != 0 or self.retro_only
            ) and not self.bottom_only:  # Top-down search
                open_nodes = []
                for node in self.search_graph.open_nodes_top:
                    assert self.can_expand_retro(node)
                    open_nodes.append(
                        (node, node.total_value + min(node.total_distance, default=0))
                    )
                if len(open_nodes) == 0:
                    break

                # Pop node
                if self.strategy == "random":
                    # Pick random node
                    best_node = open_nodes[np.random.choice(len(open_nodes))][0]
                elif self.strategy == "bfs":
                    # Find the lowest cost among parent for each open node
                    parent_costs = []
                    for node, _ in open_nodes:
                        parent_costs.append(
                            min(
                                [
                                    rxn.cost
                                    for rxn in self.search_graph.graph.predecessors(
                                        node
                                    )
                                ],
                                default=0,
                            )
                        )
                    # Pick lowest depth open node with tie broken by lowest parent cost
                    best_node = min(
                        open_nodes,
                        key=lambda x: (x[0].depth, parent_costs[open_nodes.index(x)]),
                    )[0]
                else:
                    best_node = min(open_nodes, key=lambda x: x[1])[0]
                _ = self.expand_retro(best_node, self.search_graph)
                num_iterations += 1
                pbar.update(1)
            else:  # Bottom-up search
                open_nodes = []
                for node in self.search_graph.open_nodes_bot:
                    assert self.can_expand_fwd(node)
                    open_nodes.append((node, node.total_value))
                if len(open_nodes) == 0:
                    print("bottom-up done")
                    i -= 1
                    continue

                # Pop node
                best_node = min(open_nodes, key=lambda x: x[1])[0]
                _ = self.expand_fwd(best_node, self.search_graph)
                num_iterations += 1
                pbar.update(1)

            i += 1

        # Find closest nodes if f2f
        if self.strategy == "f2f":
            open_nodes = []
            for node in self.search_graph.open_nodes_top:
                open_nodes.append((node, node.distance_number_estimate))
            best_node = min(open_nodes, key=lambda x: x[1])[0]
            print(
                f"Closest nodes from top: {best_node.smiles}>>{best_node.closest_node.smiles}"
            )

        return (
            self.search_graph.solved,
            num_iterations,
        )
