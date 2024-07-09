from rdkit import Chem


class MolNode:
    """ """

    def __init__(
        self,
        smiles,
        depth,
        building_blocks,
        root=False,
    ):
        self.smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        self.is_expanded = False
        self.solved = False
        self.reaction_number_estimate = 0
        self.reaction_number = 0
        self.descendent_costs = {}
        self.total_value = 0
        self.inherently_solved = (not root) and (self.smiles in building_blocks)
        self.depth = depth

    def __hash__(self):
        return id(self)


class RxnNode:
    """ """

    def __init__(self, smiles, template, cost, depth):
        self.smiles = smiles
        self.template = template
        self.cost = cost
        self.reaction_number = 0
        self.descendent_costs = {}
        self.total_value = 0
        self.solved = False
        self.depth = depth
