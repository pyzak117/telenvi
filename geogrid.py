# Third-Party librairies
import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, ogr, osr
from matplotlib import pyplot as plt
import shapely.geometry

class GeoGrid:

    def __init__(
        self,
        xMin = None,
        yMax = None,
        xMax = None,
        yMin = None,
        resX = None,
        resY = None,
        rows = None,
        cols = None,
        crs = 4326
    ):

        if not None in [xMin, yMax, xMax, yMin, resX, resY, rows, cols]:
            self.CASE = "0 - Intégralité des données matricielles et d'ancrage spatiale connues"
            self.xMin = xMin
            self.yMax = yMax
            self.xMax = xMax
            self.yMin = yMin
            self.resX = resX
            self.resY = resY
            self.rows = rows
            self.cols = cols

            # Calcul des dimensions métriques
            self.xLength = self.xMax - self.xMin
            self.yLength = self.yMax - self.yMin

        elif not None in [xMin, yMax, xMax, yMin] and list(set([resX, resY, rows, cols])) == [None]:
            self.CASE = "1 - Ancrage spatial entièrement connu et données matricielles entièrement inconnues. \nAffectation par défaut : 1 ligne, 1 colonne\nA déduire: résolution spatiale X et Y"

            # Known infos affectation
            self.xMin = xMin
            self.yMax = yMax
            self.xMax = xMax
            self.yMin = yMin

            # Calcul des dimensions métriques
            self.xLength = self.xMax - self.xMin
            self.yLength = self.yMax - self.yMin

            # Default values affectation to unknown variables
            self.rows = self.cols = 1

            # Compute others unknown variables
            self.resX = self.xLength / self.cols
            self.resY = -self.yLength / self.rows

        elif not None in [xMin, yMax, resX, resY, rows, cols] and list(set([xMax, yMin])) == [None]:
            self.CASE = "2.0 - Angle gauche haut (xMin, yMax) connu, données matricielles connues. \nA déduire : angle droit bas (xMax, yMin)"

            # Known infos affectation
            self.xMin = xMin
            self.yMax = yMax
            self.resX = resX
            self.resY = resY
            self.rows = rows
            self.cols = cols

            # Calcul des dimensions métriques
            self.xLength = abs(self.cols * self.resX)
            self.yLength = abs(self.rows * self.resY)

            # Compute others unknown variables
            self.xMax = self.xMin + self.xLength
            self.yMin = self.yMax - self.yLength

        elif not None in [xMax, yMin, resX, resY, rows, cols] and list(set([xMin, yMax])) == [None]:
            self.CASE = "2.1 - Angle droit bas (xMax, yMin) connu, données matricielles connues. \nA déduire : angle gauche haut (xMin, yMax)"

            # Known infos affectation
            self.xMax = xMax
            self.yMin = yMin
            self.resX = resX
            self.resY = resY
            self.rows = rows
            self.cols = cols

            # Calcul des dimensions métriques
            self.xLength = abs(self.cols * self.resX)
            self.yLength = abs(self.rows * self.resY)

            # Compute others unknown variables
            self.xMin = self.xMax - self.xLength
            self.yMax = self.yMin + self.yLength

        elif not None in [xMin, yMin, resX, resY, rows, cols] and list(set([xMax, yMax])) == [None]:
            self.CASE = "2.2 - Angle gauche bas (xMin, yMin) connu, données matricielles connues. \nA déduire : angle droite haut (xMax, yMax)"
        
            # Known infos affectation
            self.xMin = xMin
            self.yMin = yMin
            self.resX = resX
            self.resY = resY
            self.rows = rows
            self.cols = cols

            # Calcul des dimensions métriques
            self.xLength = abs(self.cols * self.resX)
            self.yLength = abs(self.rows * self.resY)

            # Compute others unknown variables
            self.xMax = self.xMin + self.xLength
            self.yMax = self.yMin + self.yLength

        elif not None in [xMax, yMax, resX, resY, rows, cols] and list(set([xMin, yMin])) == [None]:
            self.CASE = "2.3 - Angle droit haut (xMax, yMax) connu, données matricielles connues. \nA déduire : angle gauche bas (xMin, yMin)"

            # Known infos affectation
            self.xMax = xMax
            self.yMax = yMax
            self.resX = resX
            self.resY = resY
            self.rows = rows
            self.cols = cols

            # Calcul des dimensions métriques
            self.xLength = abs(self.cols * self.resX)
            self.yLength = abs(self.rows * self.resY)

            # Compute others unknown variables
            self.xMin = self.xMax - self.xLength
            self.yMin = self.yMax - self.yLength

        elif not None in [resX, resY, rows, cols] and list(set([xMin, yMax, xMax, yMin])) == [None]:
            self.CASE = "3 - Informations matricielles connues, ancrage spatiale inconnu. \nAncrage par défaut de l'angle gauche haut sur les coordonnées spatiales (0,0)."
        
            # Known infos affectation
            self.resX = resX
            self.resY = resY
            self.rows = rows
            self.cols = cols            

            # Compute grid spatial dimensions
            self.xLength = abs(self.cols * self.resX)
            self.yLength = abs(self.rows * self.resY)

            # Default values affectation to unknown variables
            self.xMin = 0
            self.yMax = 0

            # Compute opposite angle
            self.xMax = self.xMin + self.xLength
            self.yMin = self.yMax - self.yLength

        elif not None in [xMin, yMax, xMax, yMin, rows, cols] and list(set([resX, resY])) == [None]:
            self.CASE = "4 - Ancrage spatial connu. Nombre lignes / colonnes connus. \nA déduire : résolutions spatiales X et Y"

            # Known infos affectation
            self.xMin = xMin
            self.yMax = yMax
            self.xMax = xMax
            self.yMin = yMin
            self.rows = rows
            self.cols = cols

            # Calcul des dimensions métriques
            self.xLength = self.xMax - self.xMin
            self.yLength = self.yMax - self.yMin

            # Compute resolution
            self.resX = self.xLength / self.cols
            self.resY = -self.yLength / self.rows

        elif not None in [xMin, yMax, xMax, yMin, resX, resY] and list(set([rows, cols])) == [None]:
            self.CASE = "5 - Ancrage spatial connu. Résolutions spatiales X et Y connues. \nA déduire : Nombre lignes / colonnes"

            # Known infos affectation
            self.xMin = xMin
            self.yMax = yMax
            self.xMax = xMax
            self.yMin = yMin
            self.resX = resX
            self.resY = resY

            # Calcul des dimensions métriques
            self.xLength = self.xMax - self.xMin
            self.yLength = self.yMax - self.yMin

            # Compute resolution
            self.cols = int(self.xLength / self.resX)
            self.rows = int(abs(self.yLength / self.resY))

        elif list(set([xMin, yMax, xMax, yMin, resX, resY, rows, cols])) == [None]:
            self.CASE = "6 - Aucune information connue. \nAffectation de toutes les valeurs par défaut. \nUne ligne, une colonne, ancré haut-gauche sur (0,0) de résolutions (1,1)"

            # Default values affectation
            self.xMin = 0
            self.yMax = 0
            self.xMax = 1
            self.yMin = -1
            self.resX = 1
            self.resY = -1
            self.rows = 1
            self.cols = 1
            self.xLength = self.xMax - self.xMin
            self.yLength = self.yMax - self.yMin

        else:
            raise ValueError("La combinaison d'information en entrée est insuffisante pour générer une géo-grille")

        # Format crs
        if type(crs) == int: 
            self.crs = "epsg:" + str(crs)

        # Compute shapely geometry extent
        self.shplyExtent = shapely.geometry.Polygon(
            [(self.xMin, self.yMax),(self.xMin, self.yMin),(self.xMax, self.yMin),(self.xMax, self.yMax),(self.xMin, self.yMax)]
        )

    def __repr__(self):
        pres = "xMin : {}\nyMax : {}\nxMax : {}\nyMin : {}\nrows : {}\ncols : {}\nresX : {}\nresY : {}".format(
            self.xMin,
            self.yMax,
            self.xMax,
            self.yMin,
            self.rows,
            self.cols,
            self.resX,
            self.resY
        )
        return pres

    def write_grid(self, outpath = None):
        cells = []

        # For each cell
        for row in range(self.rows):
            for col in range(self.cols):

                # left upper corner
                cell_xMin = self.xMin + (col * self.resX)
                cell_yMax = self.yMax + (row * self.resY)
                
                # right bottom corner
                cell_xMax = self.xMin + ((col+1) * self.resX)
                cell_yMin = self.yMax + ((row+1) * self.resY)

                # shapely geometry formation
                shplyGeom = shapely.geometry.Polygon([(cell_xMin, cell_yMax), (cell_xMin, cell_yMin), (cell_xMax, cell_yMin), (cell_xMax, cell_yMax), (cell_xMin, cell_yMax)])

                # Add a GeoSerie line to the cells list with the cell geometry
                cells.append(gpd.GeoSeries({"geometry":shplyGeom}))
        
        # Transform in geodataframe
        grid = gpd.GeoDataFrame(cells).set_crs(self.crs)

        # Write into a shapefile it's asked by user
        if outpath != None:
            grid.to_file(outpath)

        return grid

    def write_extent(self, outpath = None):

        # Create a single GeoSerie line with the extent geometry
        line_data = gpd.GeoSeries({"geometry":self.shplyExtent})

        # Transform it in a GeoDataFrame and set his System Reference System
        extent = gpd.GeoDataFrame([line_data])
        extent.set_crs(self.crs, inplace=True)

        # Write into a shapefile it's asked by user
        if outpath != None:
            extent.to_file(outpath)

        return extent

    def intersect(self, neighboor):
        intersection = self.shplyExtent.intersection(neighboor.shplyExtent)
        xMin, yMin, xMax, yMax = intersection.bounds
        resX = neighboor.resX
        resY = neighboor.resY
        return GeoGrid(xMin, yMax, xMax, yMin, resX, resY)