from rdkit import Chem
import math
from desp.inference.utils import smiles_to_fp


def zero(smiles_1, smiles_2):
    return 0


def zero_single(smiles):
    return 0


class MolNode:
    """ """

    def __init__(self, smiles):
        self.smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        self.is_expanded = False
        self.solved = False
        self.reaction_number_estimate = 0
        self.reaction_number = 0
        self.distance_numbers = []
        self.total_distance = []
        self.descendent_costs = {}
        self.total_value = 0

    def __hash__(self):
        return id(self)


class RxnNode:
    """ """

    def __init__(self, smiles, template, cost):
        self.smiles = smiles
        self.template = template
        self.cost = cost
        self.reaction_number = 0
        self.distance_numbers = []
        self.total_distance = []
        self.descendent_costs = {}
        self.total_value = 0
        self.solved = False

    def __hash__(self):
        return id(self)


class BottomUpMolNode(MolNode):
    """ """

    def __init__(
        self, smiles, target, depth, strategy, distance_fn=zero, is_building_block=False
    ):
        super().__init__(smiles)
        self.is_building_block = is_building_block
        self.depth = depth
        self.solved = False
        self.distance_numbers = [0]
        if not is_building_block and strategy == "f2e":
            self.reaction_number_estimate = distance_fn(self.smiles, target.smiles)
        elif is_building_block:
            self.is_expanded = True
        if strategy == "f2f":
            self.fp = smiles_to_fp(self.smiles, fp_size=512).to("cuda:0")
        else:
            self.closest_node = target


class BottomUpRxnNode(RxnNode):
    """ """

    def __init__(self, smiles, template, cost, depth):
        super().__init__(smiles, template, cost)
        self.depth = depth


class TopDownMolNode(MolNode):
    """ """

    def __init__(
        self,
        smiles,
        depth,
        building_blocks,
        strategy,
        starting_materials=[],
        heuristic_fn=zero_single,
        distance_fn=zero,
        root=False,
    ):
        super().__init__(smiles)
        self.inherently_solved = not root and self.smiles in building_blocks
        self.desp_solved = False
        self.met = False

        self.depth = depth
        if self.inherently_solved:
            self.reaction_number_estimate = 0
            self.distance_number_estimate = math.inf
        else:
            self.reaction_number_estimate = heuristic_fn(self.smiles)
            if starting_materials != [] and strategy in ["f2e", "retro_sd"]:
                closest_distance = min(
                    [distance_fn(sm, self.smiles) for sm in starting_materials]
                )
                self.distance_number_estimate = (
                    closest_distance - self.reaction_number_estimate
                )
            else:
                self.distance_number_estimate = 0
        if strategy == "f2f":
            self.fp = smiles_to_fp(self.smiles, fp_size=512).to("cuda:0")


class TopDownRxnNode(RxnNode):
    """ """

    def __init__(self, smiles, template, cost, depth):
        super().__init__(smiles, template, cost)
        self.depth = depth
        self.desp_solved = False
        self.met = False
