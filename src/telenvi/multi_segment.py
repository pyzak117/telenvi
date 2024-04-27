import shapely
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd

import telenvi.vector_tools as vt
import telenvi.node as node
import telenvi.segment as segment

class MultiSegment:
    """
    Represent a set of simples segments
    """
    def __init__(self, segments):
        self.segments = segments
        self.nodes = self.get_nodes()
        self.shape = shapely.MultiLineString(self.get_segments_shapes())

    def get_nodes(self, duplicatas=False):
        """
        Return an unordered list of nodes constituing the MultiSegment instance
        """
        nodes = list(np.array([(s.a, s.b) for s in self.segments]).flatten())
        if not duplicatas:
            nodes = pd.Series(nodes).drop_duplicates().tolist()
        return nodes
        
    def get_segments_shapes(self):
        return [s.shape for s in self.segments]
        
    def get_nodes_shapes(self):
        return [n.shape for n in self.nodes]
    
    def is_continuous(self):
        """
        Check if all the segments of the road are spatially connected or not
        """
        pass

    def show(self, ax=None):
        if ax is None:
            ax = plt.subplot()
        for s in self.segments:
            s.show(ax=ax, flags=False)
        return ax

    def __repr__(self):
        self.show()
        return ''
