import networkx as nx
import math
from collections import deque
from nodes import MolNode, RxnNode


class NetworkSearcher:
    def __init__(self, target):
        self.target = MolNode(target, 1, building_blocks=[], root=True)
        self.mol_dict = {target: self.target}
        self.search_graph = nx.DiGraph()
        self.search_graph.add_node(self.target)
        self.solved = False
        self.open_nodes = {self.target}

    def can_expand_node(self, node):
        """
        Check if a node can be expanded; i.e. if it is a MolNode and has not been expanded yet.

        Args:
            node (MolNode): Node to check

        Returns:
            bool: True if node can be expanded, False otherwise
        """
        return isinstance(node, MolNode) and not node.is_expanded

    def run_updates(self, nodes):
        """
        Run updates on a set of nodes. First, uppropagate reaction number,
        then downpropagate total value. Mark the search as solved if the target
        node is solved.

        Args:
            nodes (list): List of nodes to update

        Returns:
            set: Set of nodes that were updated
        """
        nodes_to_update = set(nodes)
        nodes_to_update.update(
            self.uppropagate(
                sorted(nodes_to_update, key=lambda x: x.depth, reverse=True)
            )
        )
        nodes_to_update.update(
            self.downpropagate(
                sorted(nodes_to_update, key=lambda x: x.depth, reverse=False)
            )
        )
        if self.target.solved:
            self.solved = True
        return nodes_to_update

    def uppropagate(self, nodes):
        """
        Uppropagate reaction number from children to parents. Use a
        descendent costs dictionary to ensure no double counting of
        nodes that are successors of the same ancestors.

        Args:
            nodes (set): set of nodes to update

        Returns:
            set: Set of nodes that were updated
        """
        queue = deque(nodes)
        updated = set()
        while len(queue) > 0:
            node = queue.popleft()
            children = list(self.search_graph.successors(node))
            if isinstance(node, RxnNode):
                new_dict = {}
                for child in children:
                    new_dict.update(child.descendent_costs)
                new_dict[node.__hash__()] = node.cost
                new_reaction_number = sum(new_dict.values())
                node.descendent_costs = new_dict
                new_solved = all([child.solved for child in children])
            elif isinstance(node, MolNode):
                if node.is_expanded:
                    if len(children) > 0:
                        # Find child with minimum reaction number
                        min_child = min(children, key=lambda x: x.reaction_number)
                        new_reaction_number = min_child.reaction_number
                        node.descendent_costs = min_child.descendent_costs
                    else:
                        new_reaction_number = math.inf
                        node.descendent_costs[node.__hash__()] = math.inf
                    new_solved = any([child.solved for child in children])
                else:
                    new_reaction_number = node.reaction_number_estimate
                    node.descendent_costs[node.__hash__()] = new_reaction_number
                    new_solved = node.inherently_solved
                    if new_solved:
                        node.is_expanded = True
            else:
                raise TypeError("Node type not recognized")
            old_solved = node.solved
            old_reaction_number = node.reaction_number
            node.solved = new_solved
            node.reaction_number = new_reaction_number
            if old_reaction_number != new_reaction_number or old_solved != new_solved:
                updated.add(node)
            for parent in self.search_graph.predecessors(node):
                queue.append(parent)
        return updated

    def downpropagate(self, nodes):
        """
        Downpropagate total value from parents to children.

        Args:
            nodes (set): set of nodes to update

        Returns:
            set: Set of nodes that were updated
        """
        queue = deque(nodes)
        updated = set()
        while len(queue) > 0:
            node = queue.popleft()
            parents = list(self.search_graph.predecessors(node))
            if isinstance(node, RxnNode):
                assert len(parents) == 1
                new_total_value = (
                    node.reaction_number
                    - parents[0].reaction_number
                    + parents[0].total_value
                )
                if math.isnan(new_total_value):
                    new_total_value = math.inf
            elif isinstance(node, MolNode):
                if len(parents) == 0:  # target
                    new_total_value = node.reaction_number
                else:
                    new_total_value = min([parent.total_value for parent in parents])
            else:
                raise TypeError("Node type not recognized")
            old_total_value = node.total_value
            node.total_value = new_total_value
            if old_total_value != new_total_value:
                updated.add(node)
            for child in self.search_graph.successors(node):
                queue.append(child)
        return updated

    def get_ancestors(self, node):
        """
        Get all ancestors of a node, including the node itself, used to
        check for loops.

        Args:
            node (MolNode): Node to get ancestors of

        Returns:
            list: List of SMILES strings of ancestors
        """
        ancestors = nx.ancestors(self.search_graph, node)
        ancestors = [ancestor.smiles for ancestor in ancestors] + [node.smiles]
        return ancestors

    def expand_node(self, node, network, building_blocks):
        """
        Expand a node by adding relevant reactions from the network to the search graph.

        Args:
            node (MolNode): Node to expand
            network (ReactionNetwork): Network to expand from
            building_blocks (list): List of building block SMILES strings

        Returns:
            list: List of new nodes added to the graph
        """
        assert self.can_expand_node(node)
        node.is_expanded = True
        new_nodes = []
        for in_edge in network.in_edges(node.smiles):
            rxn_smi = in_edge[0]
            # Check if there is a loop, if so, don't use this reaction
            discard = False
            for reactant in network.in_edges(rxn_smi):
                ancestors = self.get_ancestors(node)
                if reactant[0] in ancestors:
                    discard = True
                    break
            if discard:
                continue
            # if the rxn_smi node has a "template" attribute
            template = None
            if "template" in network.nodes[rxn_smi]:
                template = network.nodes[rxn_smi]["template"]
            rxn_node = RxnNode(rxn_smi, template, 1, node.depth + 1)
            self.search_graph.add_node(rxn_node)
            self.search_graph.add_edge(node, rxn_node)
            new_nodes.append(rxn_node)
            for reactant in network.in_edges(rxn_smi):
                if reactant[0] in self.mol_dict:
                    reactant_node = self.mol_dict[reactant[0]]
                    self.search_graph.add_edge(rxn_node, reactant_node)
                else:
                    reactant_node = MolNode(
                        reactant[0], node.depth + 2, building_blocks
                    )
                    self.search_graph.add_edge(rxn_node, reactant_node)
                    new_nodes.append(reactant_node)
                    self.mol_dict[reactant[0]] = reactant_node
        self.run_updates(new_nodes + [node])
        return new_nodes

    def run_search(self, network, building_blocks):
        """
        Run search by expanding open nodes until nodes can no longer be expanded (full enumeration)

        Args:
            network (ReactionNetwork): Network to search
            building_blocks (list): List of building block SMILES strings

        Returns:
            nx.DiGraph: enumerated search graph
            bool: True if target is solved, False otherwise
        """
        while len(self.open_nodes) > 0:
            # if math.isinf(self.target.reaction_number):
            #     return self.search_graph, False
            open_nodes = []
            for node in self.open_nodes:
                assert self.can_expand_node(node)
                open_nodes.append((node, node.total_value))
            # Pop node
            best_node = sorted(open_nodes, key=lambda x: x[1], reverse=False)[0][0]
            new_nodes = self.expand_node(best_node, network, building_blocks)
            for new_node in new_nodes:
                if self.can_expand_node(new_node):
                    self.open_nodes.add(new_node)
            self.open_nodes.remove(best_node)

        return self.search_graph, self.target.solved
