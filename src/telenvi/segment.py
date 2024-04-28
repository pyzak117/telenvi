import shapely
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt

import telenvi.vector_tools as vt
import telenvi.node as node
import telenvi.multi_segment as multi_segment

class Segment:
    """
    Represent a line formed by only 2 nodes
    """
    def __init__(self, node_a, node_b):
        self.a = node_a
        self.b = node_b
        self.shape = shapely.LineString((node_a.shape, node_b.shape))
        self.ab = (self.a.xy, self.b.xy)
        self.xs = (self.a.x, self.b.x)
        self.ys = (self.a.y, self.b.y)
        self.length = abs(self.a.shape.distance(self.b.shape))

    def get_theta(self):
        diff = self.a.ar - self.b.ar
        theta_rad = np.arctan2(diff[1], diff[0])
        theta_deg = np.degrees(theta_rad)
        return theta_deg
    
    def extend(self, dist, way=None):
        """
        Extent the line on a given distance. 
        If way is None, extend the line on both line boundaries.
        """
        theta = self.get_theta()
        extended_a = self.a.move_along(dist, theta)
        extended_b = self.b.move_along(dist, theta)
        return extended_a + extended_b

    def show(self, ax=None, linecolor="black", linewidth=1, node_color="black", node_size=50, labels_dist_from_nodes=100):
        if ax is None:
            ax = plt.subplot()
        ax.scatter(self.xs, self.ys, color=node_color, s=node_size)
        ax.plot(self.xs, self.ys, linewidth=linewidth, color=linecolor)
        return ax

    def __add__(self, other_segment):
        return multi_segment.MultiSegment([self, other_segment])
        
    def __eq__(self, other_segment):
        a_vs_a = self.a == other_segment.a
        a_vs_b = self.a == other_segment.b
        b_vs_a = self.b == other_segment.a
        b_vs_b = self.b == other_segment.b
        return (a_vs_a and b_vs_b) or (a_vs_b and b_vs_a)
    
    def __repr__(self):
        self.show()
        return ''
