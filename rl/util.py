class Node(object):
    def __init__(self, data, parent=None, meta=None):
        if meta is None:
            meta = {}

        self.data = data
        self.meta = meta
        self.parent = parent
        self.children = {}

    @property
    def nchildren(self):
        return len(self.children)

    def add_child(self, data, meta=None):
        child = Node(data, self, meta)
        self.children[data] = child
        return child


class Tree(object):
    def __init__(self):
        self.root = None

    def reroot(self, data, meta=None):
        self.root = Node(data)
        return self.root
