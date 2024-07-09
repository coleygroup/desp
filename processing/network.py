import networkx as nx
from tqdm import tqdm

from utils import clear_atom_map


class ReactionNetwork(nx.DiGraph):
    def __init__(self):
        super().__init__()

    def populate_network(self, rxns):
        """
        Populate the network with reactions and molecules.

        Args:
            rxns (list): List of reaction SMILES strings. Each reaction should
                         have exactly one product.

        Returns:
            int: Number of nodes in the populated graph
            int: Number of edges in the populated graph
        """
        for rxn_smi in rxns:
            try:
                reactants, _, products = rxn_smi.split(">")
                reactants = [clear_atom_map(r) for r in reactants.split(".")]
                products = [clear_atom_map(p) for p in products.split(".")]
                assert len(products) == 1

                self.add_node(rxn_smi, label="reaction")
                for reactant in reactants:
                    self.add_node(reactant, label="molecule")
                    self.add_edge(reactant, rxn_smi)
                for product in products:
                    self.add_node(product, label="molecule")
                    self.add_edge(rxn_smi, product)
            except Exception as e:
                print(f"Error while processing reaction {rxn_smi}: {e}")
        return self.number_of_nodes(), self.number_of_edges()

    def populate_with_templates(self, rxns):
        """
        Populate the network with reactions and molecules with template information.

        Args:
            rxns (list): List of reaction dictionaries with fields:
                            rxn_smiles: SMILES strings
                            template: Reaction template

        Returns:
            int: Number of nodes in the populated graph
            int: Number of edges in the populated graph
        """
        for rxn in tqdm(rxns):
            try:
                rxn_smi = rxn["rxn_smiles"]
                try:
                    template = rxn["template"]
                except KeyError:
                    template = rxn["canon_reaction_smarts"]
                reactants, _, products = rxn_smi.split(">")
                reactants = [clear_atom_map(r) for r in reactants.split(".")]
                products = [clear_atom_map(p) for p in products.split(".")]
                assert len(products) == 1

                self.add_node(rxn_smi, label="reaction", template=template)
                for reactant in reactants:
                    self.add_node(reactant, label="molecule")
                    self.add_edge(reactant, rxn_smi)
                for product in products:
                    self.add_node(product, label="molecule")
                    self.add_edge(rxn_smi, product)
            except Exception as e:
                print(f"Error while processing reaction {rxn_smi}: {e}")
        return self.number_of_nodes(), self.number_of_edges()

    def get_unbuyable(self, building_blocks):
        """
        Get list of MolNodes that are not in a list of building blocks.

        Args:
            building_blocks (list): List of building block SMILES strings.

        Returns:
            list: List of nodes that are not in the list of building blocks.
        """
        unbuyable_nodes = []
        for node in self.nodes:
            if self.nodes[node]["label"] == "molecule" and node not in building_blocks:
                unbuyable_nodes.append(node)
        return unbuyable_nodes
