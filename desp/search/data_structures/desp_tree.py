import numpy as np
from rdkit import Chem
from collections import deque
import math
import torch
from desp.search.data_structures.nodes import (
    TopDownRxnNode,
    TopDownMolNode,
    BottomUpMolNode,
    BottomUpRxnNode,
)
from desp.search.data_structures.and_or_graph import AndOrGraph


class DespTree:
    """ """

    def __init__(
        self,
        target,
        starting_materials,
        strategy,
        heuristic_fn,
        distance_fn,
        building_blocks,
        must_use_sm,
    ):
        self.graph = AndOrGraph()
        self.target = Chem.CanonSmiles(target)
        self.starting_materials = [Chem.CanonSmiles(sm) for sm in starting_materials]
        self.strategy = strategy
        assert self.strategy in [
            "f2e",
            "f2f",
            "retro",
            "retro_sd",
            "random",
            "bfs",
            "bi-bfs",
        ]
        self.building_blocks = building_blocks
        self.distance_fn = distance_fn
        self.heuristic_fn = heuristic_fn
        self.mol_to_node = {}
        self.target_node = TopDownMolNode(
            target,
            1,
            self.building_blocks,
            self.strategy,
            starting_materials=self.starting_materials,
            heuristic_fn=self.heuristic_fn,
            distance_fn=self.distance_fn,
            root=True,
        )
        self.graph.add_node(self.target_node)
        self.mol_to_node[self.target] = self.target_node
        self.sm_nodes = []
        for sm in starting_materials:
            sm_node = BottomUpMolNode(
                sm,
                self.target_node,
                1,
                self.strategy,
                distance_fn=distance_fn,
            )
            self.graph.add_node(sm_node)
            self.sm_nodes.append(sm_node)
            self.mol_to_node[sm_node.smiles] = sm_node
        self.open_nodes_bot = {sm_node for sm_node in self.sm_nodes}
        self.open_nodes_top = {self.target_node}
        self.expanded_nodes_bot = set()
        self.expanded_nodes_top = set()
        if self.strategy == "f2f":
            self.f2f_update_bot(self.sm_nodes)
        self.must_use_sm = must_use_sm
        self.solved = False

    def expand_top(self, predictions, expanded_node):
        """
        Expand a TopDownMolNode with a list of predictions from the RetroPredictor.
        Args:
            predictions (list): list of dictionaries of the format:
                {
                    "rxn_smiles" (str): reaction SMILES string,
                    "score" (float): softmax template score,
                    "template" (str): template SMILES string,
                    "reactants" (list): list of reactant SMILES strings
                }
            expanded_node (TopDownMolNode): node to expand
        """
        assert not expanded_node.is_expanded
        expanded_node.is_expanded = True
        new_nodes = []
        met = set()
        for prediction in predictions:
            reactants = prediction["reactants"]
            rxn_smiles = prediction["rxn_smiles"]
            template = prediction["template"]
            score = prediction["score"]
            cost = -1 * np.log(score)
            # Add the reaction AND node
            rxn_node = TopDownRxnNode(
                rxn_smiles, template, cost, expanded_node.depth + 1
            )
            # Skip if any reactants would form a cycle
            skip = False
            for reactant in reactants:
                if reactant in self.mol_to_node:
                    reactant_node = self.mol_to_node[reactant]
                    if isinstance(
                        reactant_node, TopDownMolNode
                    ) and reactant in self.graph.get_ancestors(expanded_node):
                        skip = True
            if skip:
                continue
            # Now can add the reactant nodes and make connections
            self.graph.add_node(rxn_node)
            self.graph.add_edge(expanded_node, rxn_node)

            for reactant in reactants:
                if reactant in self.mol_to_node:
                    reactant_node = self.mol_to_node[reactant]
                else:
                    reactant_node = TopDownMolNode(
                        reactant,
                        expanded_node.depth + 2,
                        self.building_blocks,
                        self.strategy,
                        starting_materials=self.starting_materials,
                        heuristic_fn=self.heuristic_fn,
                        distance_fn=self.distance_fn,
                    )
                    new_nodes.append(reactant_node)
                    self.mol_to_node[reactant] = reactant_node
                    self.graph.add_node(reactant_node)
                if (
                    isinstance(reactant_node, BottomUpMolNode)
                    and not reactant_node.is_building_block
                ):
                    # print(f"Met at {reactant_node.smiles} from top")
                    met.add(reactant_node)
                new_nodes.append(rxn_node)
                self.graph.add_edge(rxn_node, reactant_node)
        return new_nodes, met

    def f2f_update_top(self, new_nodes):
        updated_bot = set()
        starts = list(self.open_nodes_bot | self.expanded_nodes_bot)
        targets = [node for node in new_nodes if isinstance(node, TopDownMolNode)]
        if len(starts) == 0 or len(targets) == 0:
            return updated_bot
        dists = self.distance_fn(
            [node.fp for node in starts], [node.fp for node in targets]
        )
        closest_indices_bot = torch.argmin(dists, axis=1)
        best_values_bot = dists[torch.arange(dists.shape[0]), closest_indices_bot]
        closest_indices_top = torch.argmin(dists, axis=0)
        best_values_top = dists[closest_indices_top, torch.arange(dists.shape[1])]
        for i, start in enumerate(starts):
            if start in self.open_nodes_bot:
                best_target = targets[closest_indices_bot[i]]
                best_value = best_values_bot[i]
                if start.reaction_number_estimate > best_value:
                    start.closest_node = best_target
                    start.reaction_number_estimate = best_value
                    updated_bot.add(start)
        for i, target in enumerate(targets):
            best_start = starts[closest_indices_top[i]]
            best_value = best_values_top[i]
            target.closest_node = best_start
            target.distance_number_estimate = (
                best_value - target.reaction_number_estimate
            )
        return updated_bot

    def expand_bot(self, predictions, expanded_node):
        """
        Expand a BottomUpMolNode with a list of predictions from the ForwardPredictor.
        Args:
            predictions (list): list of dictionaries of the format:
                {
                    "product" (str): product SMILES string,
                    "smiles" (str): reaction SMILES string,
                    "template" (str): reaction template string,
                    "score" (float): softmax template score,
                    "building_block" (str): building block SMILES string
                }
            expanded_node (BottomUpMolNode): node to expand
        """
        assert not expanded_node.is_expanded
        expanded_node.is_expanded = True
        new_nodes = []
        met = set()
        for prediction in predictions:
            product = prediction["product"]
            rxn_smiles = prediction["rxn_smiles"]
            template = prediction["template"]
            score = prediction["score"]
            cost = -1 * np.log(score)  # negative log likelihood
            building_block = prediction["building_block"]

            # Add the product OR node
            if product in self.mol_to_node:
                product_node = self.mol_to_node[product]
                if product in self.graph.get_descendants(expanded_node):
                    continue  # would form a cycle / break depth ordering
            else:
                product_node = BottomUpMolNode(
                    product,
                    self.target_node,
                    expanded_node.depth + 2,
                    self.strategy,
                    self.distance_fn,
                )
                self.graph.add_node(product_node)
                self.mol_to_node[product] = product_node
                new_nodes.append(product_node)

            if isinstance(product_node, TopDownMolNode):
                met.add(product_node)

            # Add the reaction AND node and connect the product OR node to it
            rxn_node = BottomUpRxnNode(
                rxn_smiles, template, cost, expanded_node.depth + 1
            )
            new_nodes.append(rxn_node)
            self.graph.add_node(rxn_node)
            self.graph.add_edge(rxn_node, expanded_node)
            self.graph.add_edge(product_node, rxn_node)

            # Add the building block OR node and connect the reaction AND node to it
            if building_block is not None:
                bb_node = BottomUpMolNode(
                    building_block,
                    self.target_node,
                    expanded_node.depth,
                    self.strategy,
                    is_building_block=True,
                )
                self.graph.add_node(bb_node)
                self.graph.add_edge(rxn_node, bb_node)

        return new_nodes, met

    def f2f_update_bot(self, new_nodes):
        updated_top = set()
        starts = [node for node in new_nodes if isinstance(node, BottomUpMolNode)]
        targets = list(self.open_nodes_top | self.expanded_nodes_top)
        if len(starts) == 0 or len(targets) == 0:
            return updated_top
        dists = self.distance_fn(
            [node.fp for node in starts], [node.fp for node in targets]
        )
        closest_indices_bot = torch.argmin(dists, axis=1)
        best_values_bot = dists[torch.arange(dists.shape[0]), closest_indices_bot]
        closest_indices_top = torch.argmin(dists, axis=0)
        best_values_top = dists[closest_indices_top, torch.arange(dists.shape[1])]
        for i, start in enumerate(starts):
            best_target = targets[closest_indices_bot[i]]
            best_value = best_values_bot[i]
            start.closest_node = best_target
            start.reaction_number_estimate = best_value
        for i, target in enumerate(targets):
            if target in self.open_nodes_top:
                best_start = starts[closest_indices_top[i]]
                best_value = best_values_top[i]
                if (
                    target.distance_number_estimate + target.reaction_number_estimate
                    > best_value
                ):
                    target.closest_node = best_start
                    target.distance_number_estimate = (
                        best_value - target.reaction_number_estimate
                    )
                    updated_top.add(target)
        return updated_top

    def run_updates(self, nodes, met, direction):
        """
        Updates the graph by propagating values up or down the graph.
        1. If direction is "bottom_up", then "reaction number" is propagated down the graph
           and "total value" is propagated up the graph.
        2. If direction is "top_down", then "reaction number" is propagated up the graph
           and "total value" is propagated down the graph.
        Args:
            nodes (list): list of nodes to update
            direction (str): direction to update the graph
        Returns:
            set: set of nodes that were updated
        """
        nodes_to_update = set(nodes)

        if direction == "bottom_up":
            nodes_to_update.update(
                self.downpropagate_bot(
                    sorted(nodes_to_update, key=lambda node: node.depth, reverse=True)
                )
            )
            nodes_to_update.update(
                self.uppropagate_bot(
                    sorted(nodes_to_update, key=lambda node: node.depth, reverse=False)
                )
            )
            if len(met) > 0:
                # print("Met nodes from bot to top!")
                met.update(self.uppropagate_top(met))
                self.downpropagate_top(met)
        elif direction == "top_down":
            nodes_to_update.update(
                self.uppropagate_top(
                    sorted(nodes_to_update, key=lambda node: node.depth, reverse=True)
                )
            )
            nodes_to_update.update(
                self.downpropagate_top(
                    sorted(nodes_to_update, key=lambda node: node.depth, reverse=False)
                )
            )
            if len(met) > 0:
                # print("Met nodes from top to bot!")
                met.update(self.downpropagate_bot(met))
                self.uppropagate_bot(met)
        if self.target_node.desp_solved or (
            not self.must_use_sm and self.target_node.solved
        ):
            self.solved = True
        return nodes_to_update

    def uppropagate_top(self, nodes):
        """ """
        queue = deque(nodes)
        updated = set()
        while len(queue) > 0:
            node = queue.popleft()
            children = list(self.graph.successors(node))
            if isinstance(node, TopDownRxnNode):
                new_dict = {}
                for child in children:
                    new_dict.update(child.descendent_costs)
                new_dict[node.__hash__()] = node.cost
                new_reaction_number = sum(new_dict.values())
                node.descendent_costs = new_dict
                new_distance_numbers = []
                for child in children:
                    new_distance_numbers += child.distance_numbers
                new_solved = all(
                    [
                        (
                            isinstance(child, BottomUpMolNode)
                            and not child.is_building_block
                        )
                        or child.solved
                        or child.desp_solved
                        for child in children
                    ]
                )
                met = any(
                    [
                        (
                            isinstance(child, BottomUpMolNode)
                            and not child.is_building_block
                        )
                        or child.desp_solved
                        for child in children
                    ]
                )
                new_desp_solved = met and new_solved
            elif isinstance(node, TopDownMolNode):
                if node.is_expanded:
                    if any([isinstance(child, BottomUpRxnNode) for child in children]):
                        new_reaction_number = 0
                        new_distance_numbers = [0]
                        node.descendent_costs = {node.__hash__(): 0}
                        node.lowest_child = next(
                            child
                            for child in children
                            if isinstance(child, BottomUpRxnNode)
                        )
                    elif len(children) > 0:
                        min_child = min(
                            children, key=lambda child: child.reaction_number
                        )
                        new_reaction_number = min_child.reaction_number
                        new_distance_numbers = min_child.distance_numbers
                        node.descendent_costs = min_child.descendent_costs
                        node.lowest_child = min_child
                    else:
                        new_reaction_number = math.inf
                        new_distance_numbers = [math.inf]
                        node.descendent_costs[node.__hash__()] = math.inf
                else:
                    if any([isinstance(child, BottomUpRxnNode) for child in children]):
                        new_reaction_number = 0
                        new_distance_numbers = [0]
                        node.descendent_costs = {node.__hash__(): 0}
                    else:
                        new_reaction_number = node.reaction_number_estimate
                        new_distance_numbers = [node.distance_number_estimate]

                        node.descendent_costs[node.__hash__()] = new_reaction_number
                new_solved = node.inherently_solved or any(
                    [
                        isinstance(child, BottomUpRxnNode) or child.solved
                        for child in children
                    ]
                )
                new_desp_solved = any(
                    [
                        isinstance(child, BottomUpRxnNode) or child.desp_solved
                        for child in children
                    ]
                )
                if node.inherently_solved:
                    node.is_expanded = True
            else:
                continue
            old_solved = node.solved
            old_desp_solved = node.desp_solved
            old_reaction_number = node.reaction_number
            old_distance_numbers = node.distance_numbers
            node.solved = new_solved
            node.reaction_number = new_reaction_number
            node.distance_numbers = new_distance_numbers
            node.desp_solved = new_desp_solved
            if (
                old_reaction_number != new_reaction_number
                or sorted(old_distance_numbers) != sorted(new_distance_numbers)
                or old_solved != new_solved
                or old_desp_solved != new_desp_solved
            ):
                updated.add(node)
                for parent in self.graph.predecessors(node):
                    queue.append(parent)
        return updated

    def downpropagate_top(self, nodes):
        """ """
        queue = deque(nodes)
        updated = set()
        while len(queue) > 0:
            node = queue.popleft()
            parents = list(self.graph.predecessors(node))
            if isinstance(node, TopDownRxnNode):
                new_total_value = (
                    node.reaction_number
                    - parents[0].reaction_number
                    + parents[0].total_value
                )
                min_distance_numbers = parents[0].lowest_child.distance_numbers
                parent_distance_numbers = parents[0].distance_numbers.copy()
                for num in min_distance_numbers:
                    parent_distance_numbers.remove(num)
                new_total_distance = parent_distance_numbers + node.distance_numbers
            elif isinstance(node, TopDownMolNode):
                if len(parents) == 0:  # target
                    new_total_value = node.reaction_number
                    new_total_distance = node.distance_numbers
                else:
                    parent_rns = [p.reaction_number for p in parents]
                    best_parent = parents[parent_rns.index(min(parent_rns))]
                    best_parent = parents[0]
                    new_total_value = best_parent.total_value
                    new_total_distance = best_parent.distance_numbers
            else:
                continue
            old_total_distance = node.total_distance
            old_total_value = node.total_value
            node.total_value = new_total_value
            node.total_distance = new_total_distance
            if old_total_value != new_total_value or (
                sorted(old_total_distance) != sorted(new_total_distance)
            ):
                updated.add(node)
                for child in self.graph.successors(node):
                    queue.append(child)
        return updated

    def downpropagate_bot(self, nodes):
        """ """
        queue = deque(nodes)
        updated = set()
        while len(queue) > 0:
            node = queue.popleft()
            parents = list(self.graph.predecessors(node))
            if isinstance(node, BottomUpRxnNode):
                assert len(parents) == 1
                if isinstance(parents[0], TopDownMolNode):
                    new_reaction_number = node.cost
                else:
                    new_reaction_number = node.cost + parents[0].reaction_number
                new_solved = isinstance(parents[0], TopDownMolNode) or parents[0].solved
            elif isinstance(node, BottomUpMolNode):
                if node.is_expanded:
                    if any([isinstance(parent, TopDownRxnNode) for parent in parents]):
                        new_reaction_number = 0
                    elif len(parents) > 0:
                        parent_costs = [
                            rxn.reaction_number for rxn in self.graph.predecessors(node)
                        ]
                        new_reaction_number = min(parent_costs, default=math.inf)
                    else:
                        new_reaction_number = math.inf
                else:
                    new_reaction_number = node.reaction_number_estimate
                new_solved = any(
                    [
                        isinstance(parent, TopDownRxnNode) or parent.solved
                        for parent in parents
                    ]
                )
            else:
                continue
            old_reaction_number = node.reaction_number
            old_solved = node.solved
            node.reaction_number = new_reaction_number
            node.solved = new_solved
            if old_reaction_number != new_reaction_number or old_solved != new_solved:
                updated.add(node)
                for child in self.graph.successors(node):
                    queue.append(child)
        return updated

    def uppropagate_bot(self, nodes):
        """ """
        queue = deque(nodes)
        updated = set()
        while len(queue) > 0:
            node = queue.popleft()
            children = list(self.graph.successors(node))
            if isinstance(node, BottomUpRxnNode):
                for child in children:
                    if (
                        not child.is_building_block  # only care about the non building block
                    ):
                        new_total_value = (
                            node.reaction_number
                            - child.reaction_number
                            + child.total_value
                        )
                        break
            elif isinstance(node, BottomUpMolNode):
                if len(children) == 0:  # starting material
                    new_total_value = node.reaction_number
                else:
                    new_total_value = min([child.total_value for child in children])
            else:
                continue
            old_total_value = node.total_value
            node.total_value = new_total_value
            if old_total_value != new_total_value:
                updated.add(node)
                for parent in self.graph.predecessors(node):
                    queue.append(parent)
        return updated
