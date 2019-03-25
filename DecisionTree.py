import numpy as np

class _Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.features = None
        self.threshold = None
        self.label = None
        self.numdata = None
        self.gini_index = None

    def build(self, data, target):
        
