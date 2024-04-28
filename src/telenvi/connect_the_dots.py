import time
import telenvi.vector_tools as vt
from shapely import LineString
import geopandas as gpd
import numpy as np
import random
from matplotlib import pyplot as plt
import sys

def connect_the_dots(population, dist_orth=6, dist_diag=8):

    # Ouverture des données d'entrée
    population = vt.getGeoSerie(population)
    initial_num_points = len(population)

    # Définition des conteneurs
    reseau = []
    pts_isoles = []
    serie = []
    anim_processing(len(population), initial_num_points)

    # Définition des compteurs
    k = True
    i = 0

    # Définition d'un point d'origine
    pt_courant = population.iloc[0]

    # Processing
    while k:

        i += 1

        # Identification des voisins orthogonaux et diagonaux
        vs_orth = vt.getNeighbors(pt_courant, population, dist_orth)
        vs_diag = vt.getNeighbors(pt_courant, population, dist_diag)

        # Si le point n'a aucun voisin
        if len(vs_orth) == 0 and len(vs_diag) == 0:

            # On l'ajoute à la liste des pts isolés qui nous serviront peut être plus tard
            pts_isoles.append(pt_courant)

            # On le dégage de la population pour ne pas retomber dessus
            population = population[~population.geometry.geom_equals(pt_courant)]

            # On vérifie qu'il y ai encore qqe chose dans la population
            if len(list(population.geometry.values)) > 0:
                pt_courant = random.choice(list(population.geometry.values))

            # Sinon : on stoppe la boucle
            else:
                k = False

            # Et on passe à l'itération suivante
            continue
        
        # Si on est tjs dans la boucle : on supprime les voisins déjà dans la série
        vs_orth_not_in_serie = [v for v in vs_orth.geometry.iloc if v not in serie]
        vs_diag_not_in_serie = [v for v in vs_diag.geometry.iloc if v not in serie]
        
        # Si on a qqe chose à moins de 5m
        if len(vs_orth_not_in_serie) > 0:

            # On ajout le pt courant à la série
            serie.append(pt_courant)

            # On met à jour le point_courant
            pt_courant = vs_orth_not_in_serie[0]

            # On passe à l'itération suivante
            continue

        # Si on est tjs dans la boucle et qu'on a qqe ch entre 5m et 8m
        if len(vs_diag_not_in_serie) > 0:
            serie.append(pt_courant)
            pt_courant = vs_diag_not_in_serie[0]
            continue

        # Si on est tjs dans la boucle : c'est que le point a des voisins
        # Mais qu'ils ont déjà tous été ajoutés à la série
        # C'est donc la fin de la série
        
        # On créée une ligne à partir des points ordonnés
        route = LineString(serie)

        # On l'ajoute au réseau
        reseau.append(route)

        # On dégage les points qui ont été ajoutés à la route
        population = population[~population.intersects(route)]
        anim_processing(len(population), initial_num_points)

        # Si on est tjs dans la boucle c'est qu'il y a encore du monde dans population
        # Donc on choisit un nouveau point courant

        # On remet la série à 0
        serie = []

        # On vérifie qu'il y ai encore qqe chose dans la population
        if len(list(population.geometry.values)) > 0:
            pt_courant = random.choice(list(population.geometry.values))

        # Sinon : on stoppe la boucle
        else:
            k = False

    return reseau, pts_isoles

def anim_processing(progress, n_init, n=50):
    current_ratio = n-int((progress / n_init) * n)
    for x in range(n):
        if x <= current_ratio:
            sys.stdout.write('■')
        else:
            sys.stdout.write('□')
        sys.stdout.flush()
        time.sleep(0.01)  # Controls the speed of the "typing"
    zeros_to_add = 4 - len(str(progress))
    zeros = ''
    for z in range(zeros_to_add):
        zeros += '0'
    sys.stdout.write( f"| {zeros}{progress} remaining points")
    sys.stdout.write('\n')
