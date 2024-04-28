from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import shapely
import numpy as np
import geopandas as gpd

import telenvi.vector_tools as vt
import telenvi.node as node
import telenvi.segment as segment
import telenvi.multi_segment as multi_segment

class GeoNetwork:
    """
    Represent a set of Nodes
    """

    def __init__(self, nodes):
        if type(nodes) == node.Node:
            nodes = [nodes]
        self.nodes = [node.create_node(n) for n in nodes]

    def connect_nodes(self, scope=5, duplicatas=False):
        """
        Create segments between network's nodes
        2 points are connected only if their distance is <= scope
        """

        links = []
        for node in tqdm(self.nodes):

            # Cherche le copain le plus proche du point de départ
            copains, dist = node.get_nearests(self)

            # S'il y a un ou plusieurs copain dans une distance inférieure à la scope:
            if dist <= scope:

                # Pour chaque copain
                for copain in copains.nodes:

                    # Connecte le point de départ au copain
                    target_link = node + copain

                    # Si et seulement si le lien n'a pas été créé auparavant par le copain
                    if target_link not in links: 
                        links.append(target_link)

        # Construit un ensemble de points
        return multi_segment.MultiSegment(links)

    def show(self, ax=None, color='red', node_size=50):
        if ax is None:
            ax = plt.subplot()
        [n.show(ax=ax, color=color, markersize=node_size) for n in self.nodes]

    def __add__(self, other):

        # On ajoute un node au réseau de l'instance
        if type(other) == node.Node:
            new_population = self.nodes + [other]

        # On combine les deux réseaux
        elif type(other) == GeoNetwork:
            new_population = self.nodes + other.nodes

        return GeoNetwork(new_population)

    def __repr__(self):
        self.show()
        return ''