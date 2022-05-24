#%% Standard librairies
import os
import re

# Third-Party librairies
import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, ogr, osr
from matplotlib import pyplot as plt
import shapely

class GeoGrid:
    
    """
    Represent a grid of squares cells georeferenced in a 2D space.

    attributes
    ----------

        name | type  | short description
        ----------------------------------------------------------------
        crs  |int/str| epsg or a string representing the spatial coordinates system of the grid
        ----------------------------------------------------------------
        xMin | float | edges of the grid in the spatial coordinates system 
        yMax | float | "
        xMax | float | "
        yMin | float | "
        -----------------------------------------------------------------
        xLen | float | dimensions of the grid, in the spatial coordinates system unity (meters, degrees)
        yLen | float | "
        -----------------------------------------------------------------
        xRes | float | dimensions of a cell, in the spatial coordinates system unity (meters, degrees)
        yRes | float | "
        -----------------------------------------------------------------
        nRows| int   | dimensions of the grid, in number of cells
        nCols| int   | "

    methods
    -------

        getShapelyExtent()
                Return an osgeo.OGR or shapely.geometry.Polygon which represent 
                the extent of the grid, in his spatial coordinates system

        writeShapefileExtent(outPath)
                Print the extent in a shapefile

        getAllCells()
                Return a geogridFrame where each line represent a cell
                with their coordinates and their attributes

        writeAllCellsInShapefile(outPath)
                Print the grid with all their cells 
                in a shapefile - CAUTION : according, to the grid 
                size and resolutions, this may be heavy /!\\

        intersect (slave : GeoGrid)
                Return a new GeoGrid with the resolution of the reference GeoGrid, 
                only on the area where reference and slave object are intersecting    
    """

    def __init__(self, xMin, yMax, xRes, yRes, nCols, nRows, crs=None):

        # Inputs checking
        if type(nRows) != int or type(nCols) != int:
            raise ValueError("nRows or nCols not integer")

        # Get known attributes
        if type(crs) == int:
            crs = "epsg:{}".format(crs)
        self.crs = crs
        self.xMin = xMin
        self.yMax = yMax
        self.xRes = xRes
        self.yRes = yRes
        self.nCols = nCols
        self.nRows = nRows
        self.crs = crs

        # Compute unknown attributes
        self.xLen = abs(self.nCols * self.xRes)
        self.yLen = abs(self.nRows * self.yRes)
        self.xMax = self.xMin + self.xLen
        self.yMin = self.yMax - self.yLen

    def __repr__(self):
        print(self.__dict__)
        return ""
    
    def getOGRextent(self):
        """
        return a ogr.Geometry
        representing the spatial extent of the grid
        """
        # Create a ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(self.geogrid.xMin, self.geogrid.yMax)
        ring.AddPoint(self.geogrid.xMin, self.geogrid.yMin)
        ring.AddPoint(self.geogrid.xMax, self.geogrid.yMin)
        ring.AddPoint(self.geogrid.xMax, self.geogrid.yMax)
        ring.AddPoint(self.geogrid.xMin, self.geogrid.yMax)

        # Assign this ring to a polygon
        polygon_env = ogr.Geometry(ogr.wkbPolygon)
        polygon_env.AddGeometry(ring)

        return polygon_env

    def getShapelyExtent(self):
        """
        return a shapely.Geometry.Polygon
        representing the spatial extent of the grid
        """

        x = shapely.geometry.Polygon(
            [(self.xMin, self.yMax),
             (self.xMin, self.yMin),
             (self.xMax, self.yMin),
             (self.xMax, self.yMax),
             (self.xMin, self.yMax)])
        return x

    def writeShapefileExtent(self, outPath):
        """
        :descr:
        -------
            write a shapefile containing one polygon 
            representing the spatial extent of the grid

        :params:
        --------
            outPath : str - the path of the shapefile to create. 
                            Must finished by ".shp"
        """
        # Create a single GeoSerie line with the extent geometry
        line_data = gpd.GeoSeries({"geometry":self.getShapelyExtent()})

        # Transform it in a geogridFrame and set his System Reference System
        x = gpd.geogridFrame([line_data])
        x.set_crs(self.crs, inplace=True)

        # Write into a shapefile
        x.to_file(outPath)

    def getAllCells(self):
        """
        :descr:
        -------
            Compute the geogrpahic coordinates of all the cells of a GeoGrid

        :returns:
        ---------
            grid : geogridFrame - Each line represent a cell with 
                                  his coordinates and his attributes
            self.grid : save  the grid computed in an attribute of the GeoGrid.
        """

        # Empty list
        cells = []

        # For each cell
        for row in range(self.nRows):
            for col in range(self.nCols):

                # 1st corner of the cell
                cell_xMin = self.xMin + (col * self.xRes)
                cell_yMax = self.yMax + (row * self.yRes)
                
                # 2nd corner of the cell
                cell_xMax = self.xMin + ((col+1) * self.xRes)
                cell_yMin = self.yMax + ((row+1) * self.yRes)

                # shapely geometry formation
                shplyGeom = shapely.geometry.Polygon(
                    [(cell_xMin, cell_yMax),
                     (cell_xMin, cell_yMin),
                     (cell_xMax, cell_yMin),
                     (cell_xMax, cell_yMax),
                     (cell_xMin, cell_yMax)])

                # Add a GeoSerie line contain the cell geometry to the cells list
                cells.append(gpd.GeoSeries({"geometry":shplyGeom}))
        
                # [Script the getting of attributes here]
                # ...

        self.grid = gpd.geogridFrame(cells).set_crs(self.crs)
        return self.grid

    def writeAllCellsInShapefile(self, outPath):
        """
        :descr:
        -------
            write a shapefile containing a polygon for each cell of the grid

        :params:
        --------
            outPath : str - the path of the shapefile to create. 
                            Must finished by ".shp"
        """

        try :
            self.grid.to_file(outPath)
        except AttributeError:
            self.grid = self.getAllCells()
            self.grid.to_file(outPath)

    def intersect(self, master):
        """
        :descr:
        -------
            return a new GeoGrid. 
            The slave (GeoGrid from which the method is called)
            is aligned on the master (the GeoGrid give as parameter).
            The extent of the resulting GeoGrid is corresponding to 
            the common area of the master and slave. 
            If empty, None is returned.

        :params:
        --------
            slave : GeoGrid - an other GeoGrid
        
        :returns:
        ---------
            self :  GeoGrid - The slave
            child : GeoGrid - The master 
        """
        # Get reference grid x & y resolutions
        xRes = master.xRes
        yRes = master.yRes
        crs = master.crs

        # Intersect master and master
        x = self.getShapelyExtent().intersection(master.getShapelyExtent())
        xMin, yMin, xMax, yMax = x.bounds
        xLen = xMax - xMin
        yLen = yMax - yMin

        # Compute number of rows and cols
        nRows = int(abs(xLen / xRes))
        nCols = int(abs(yLen / yRes))

        # Compute grid dimensions
        return GeoGrid(crs, xMin, yMax, xRes, yRes, nCols, nRows)

class GeoIm:

    """
    Describe a georeferenced image. 
    A pixel represent a square part of the space.
    Each pixel must represent a space of the same size.

    attributes
    ----------

        name   | type       | short description
        ----------------------------------------------------------------
        pxData | np.ndarray | each pixel have one or many values. 
                              Here, they are stored in an array or a matrix. 
                              This array have 2 dimensions (x,y) if the image 
                              represented is monospectral (only one channel).
                              or 3 dimensions (x, y, channel) if the image is multispectral 
                              (just as the color images, with a Red, a Green 
                              and a Blue channel)
        ----------------------------------------------------------------
        geogrid | GeoGrid  | describe the spatial position of the image,
                             in a Spatial Coordinates System given

    methods
    -------
        [...]

    """

    def __init__(self, pxData, geogrid, crs):
        self.pxData = pxData
        self.geogrid = geogrid
        self.crs = crs

    def __add__(self, neighboor):
        res = self.pxData + neighboor.pxData
        x = GeoIm(res, self.geogrid, self.crs)
        return x
    
    def __sub__(self, neighboor):
        res = self.pxData - neighboor.pxData
        x = GeoIm(res, self.geogrid, self.crs)
        return x

    def __mul__(self, neighboor):
        res = self.pxData * neighboor.pxData
        x = GeoIm(res, self.geogrid, self.crs)
        return x

    def __truediv__(self, neighboor):
        res = self.pxData / neighboor.pxData
        x = GeoIm(res, self.geogrid, self.crs)
        return x

    def __repr__(self):
        self.quickVisual()
        return ""
    
    
    def getCoordsExtent(self):
        """
        Voué à disparaître -- En attente d'homogénéisation avec les scripts utilisant cette fonctionnalité

        :return:
        --------
            bounding-box coordinates of the GeoIm's spatial extent
        """
        return (self.geogrid.xMin, self.geogrid.yMax, self.geogrid.xMax, self.geogrid.Min)

    def getGeomExtent(self, mode = "ogr"):
        """
        the geometry of the image spatial extent
        """
        if mode == "ogr":
            return self.geogrid.getOGRextent()

        elif mode == "shply":
            return self.geogird.getShapelyExtent()
    
    def exportAsRasterFile(
        self,
        outP,
        format = gdalconst.GDT_Float32,
        driverName = "GTiff"):

        """
        Export a GeoIm object into a raster file georeferenced.

        :param:
        -------
            outP (str) : the path where you want to save the raster
            format (gdalconst) : the image format
            driverName (str) : the extension of the raster file
        """

        driver = gdal.GetDriverByName(driverName)

        # Check if the array is 2D or 3D
        dim = len(self.pxData.shape)

        if dim == 2:
            nb_bands = 1
            rows, cols = self.pxData.shape

        elif dim == 3:
            nb_bands, rows, cols = self.pxData.shape

        else:
            raise ValueError("Array must be in 2 or 3 dimensions")

        # gdal.Dataset creation
        outDs = driver.Create(outP, cols, rows, nb_bands, format)
        outDs.SetGeoTransform((self.geogrid.xMin, self.geogrid.xRes, 0.0, self.geogrid.yMax, self.geogrid.yRes))
        outDs.SetProjection(self.crs)

        # Export each band of the 3D array
        if dim == 3:
            for band in range(1, nb_bands+1):
                outDs.GetRasterBand(band).WriteArray(self.pxData[band-1])

        # Export the unique band
        else:
            outDs.GetRasterBand(1).WriteArray(self.pxData)

        outDs.FlushCache()
        print("\n" + os.path.basename(outP) + " OK")
        return None
    
    def quickVisual(self, band = 0, colors = "viridis"):
        """
        Show the data as an image

        :param:
        -------
            band : int - if the array of pixels values represent a 
                         multispectral image, with 3 dimensions, 
                         you can choose the band than you want to show here.

            colors : str - a string describing the color-range to use 
                           to show the image
        """

        if len(self.pxlV.shape) == 2:
            plt.imshow(self.pxlV, cmap = colors)

        elif len(self.pxlV.shape) == 3:
            plt.imshow(self.pxlV[band], cmap = colors)

        plt.show()
        plt.close()
        return None
