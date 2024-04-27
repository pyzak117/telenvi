import shapely
import numpy as np
import geopandas as gpd

import telenvi.vector_tools as vt
import telenvi.segment as segment

def create_node(target=None, x=None, y=None):
    target_shape = vt.getGeoThing(target, x, y)
    return Node(target_shape.x, target_shape.y)
    
class Node:
    """
    Represent a point in space
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.shape = shapely.Point((x, y))
        self.xy = (self.x, self.y)
        self.ar = np.array(self.xy)

    def buffer(self, size):
        return self.shape.buffer(size)

    def scan_around(self, population, scope, self_included=False):
        """
        Find other nodes in geodataframe population inside the radius defined by scope
        """
        research_area = self.buffer(scope)
        nodes_in_area = population[population.intersects(research_area)]
        if not self_included:
            nodes_in_area = nodes_in_area[~nodes_in_area.geom_equals(self.shape)] 
        return [Node(t.geometry.x, t.geometry.y) for t in nodes_in_area.iloc]
    
    def move_along(self, dist, theta, rad=False):
        """
        Return a new node moved from dist on the direction given by theta
        """
        if not rad:
            theta = np.deg2rad(theta)
        delta_x = dist * np.cos(theta)
        delta_y = dist * np.sin(theta)
        return self.move(delta_x, delta_y)

    def move(self, delta_x, delta_y):
        """
        Return a new node moved from delta_x and delta_y
        """
        return Node(self.x + delta_x, self.y + delta_y)

    def copy(self):
        return Node(self.x, self.y)
        
    def __add__(self, other_node):
        return vt.segment.Segment(self, other_node)

    def __eq__(self, other_node):
        return self.x == other_node.x and self.y == other_node.y
        
    def __repr__(self):
        return f"{self.xy}"