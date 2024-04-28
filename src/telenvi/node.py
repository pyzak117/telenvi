from matplotlib import pyplot as plt
import shapely
import numpy as np
import geopandas as gpd

import telenvi.vector_tools as vt
import telenvi.segment as segment
import telenvi.geo_network as geo_network

def create_node(target=None, x=None, y=None):
    if type(target) == Node:
        return target
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
        population = vt.getGeoDf(population)
        research_area = self.buffer(scope)
        nodes_in_area = population[population.intersects(research_area)]
        if not self_included:
            nodes_in_area = nodes_in_area[~nodes_in_area.geom_equals(self.shape)]
        return geo_network.GeoNetwork([create_node(t.geometry) for t in nodes_in_area.iloc])
    
    def get_distance_from_other(self, other):
        return self.shape.distance(vt.getGeoThing(other))

    def get_nearests(self, population, self_included=False):

        # Create a geodataframe from the population
        population = vt.getGeoDf(population)

        # Compute the distance between the instance and all the others points
        population['dist_from_instance'] = population.apply(lambda row:self.get_distance_from_other(row), axis=1)

        # Remove the instance's geometry
        if not self_included:
            population = population[population.dist_from_instance > 0]

        # Get only the points with the minimal distance
        min_dist = population.dist_from_instance.min()
        population = population[population.dist_from_instance == min_dist]

        # S'il n'y a aucun point - c'est que le réseau n'était constitué que du point de départ
        if len(population) == 0:
            return None

        # On fabrique un nouveau réseau avec les points voisins
        neighbors = geo_network.GeoNetwork([create_node(n) for n in population.iloc])
        return neighbors, min_dist

    def connect_with_neighbors(self, population, max_scope):
        """
        Send a list of segments with each of the neighbors in the scope
        """
        links = [self + copain for copain in self.scan_around(population, max_scope).nodes]
        
        return links

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
        
    def show(self, ax=None, node_size=50, color='red'):
        if ax is None:
            ax = plt.subplot()
        ax.scatter(self.x, self.y, s=node_size, color=color)
        return ax

    def __add__(self, other_node):
        return segment.Segment(self, other_node)

    def __eq__(self, other_node):
        return self.x == other_node.x and self.y == other_node.y
        
    def __repr__(self):
        return f"{self.xy}"