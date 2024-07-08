import networkx as nx


class AndOrGraph(nx.DiGraph):
    """ """

    def __init__(self):
        super().__init__()

    def get_ancestors(self, node):
        ancestors = nx.ancestors(self, node)
        ancestors = [ancestor.smiles for ancestor in ancestors] + [node.smiles]
        return ancestors

    def get_descendants(self, node):
        descendants = nx.descendants(self, node)
        descendants = [descendant.smiles for descendant in descendants] + [node.smiles]
        return descendants
