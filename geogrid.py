#%% Third-Party librairies
import geopandas as gpd
from osgeo import ogr
import shapely
import math

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
                Return a GeoDataFrame where each line represent a cell
                with their coordinates and their attributes

        writeAllCellsInShapefile(outPath)
                Print the grid with all their cells 
                in a shapefile - CAUTION : according, to the grid 
                size and resolutions, this may be heavy /!\\

        intersect (student : GeoGrid)
                Return a new GeoGrid with the resolution of the reference GeoGrid, 
                only on the area where reference and student object are intersecting    
    """

    def __init__(self, xMin, yMax, xRes, yRes, nCols, nRows, crs=None):

        # Inputs checking
        if type(nRows) != int or type(nCols) != int:
            raise ValueError("nRows or nCols not integer")

        # Get known attributes
        self.xMin = xMin
        self.yMax = yMax
        self.xRes = xRes
        self.yRes = yRes
        self.nCols = nCols
        self.nRows = nRows

        if type(crs) == int:
            crs = "epsg:{}".format(crs)
        self.crs = crs

        # Compute unknown attributes
        self.xLen = abs(self.nCols * self.xRes)
        self.yLen = abs(self.nRows * self.yRes)
        self.xMax = self.xMin + self.xLen
        self.yMin = self.yMax - self.yLen

    def __repr__(self):
        attributes = self.__dict__.copy()
        attributes.pop("crs")
        print(attributes)
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

        # Transform it in a GeoDataFrame and set his System Reference System
        x = gpd.GeoDataFrame([line_data])
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
            grid : GeoDataFrame - Each line represent a cell with 
                                  his coordinates and his attributes
            self.grid : save  the grid computed in an attribute of the GeoGrid.
        """

        # Empty list
        cells = []

        # For each cell
        for row in range(0, self.nRows):
            for col in range(0, self.nCols):

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

        self.grid = gpd.GeoDataFrame(cells).set_crs(self.crs)
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

    def intersect(self, teacher):
        """
        :descr:
        -------
            return a new GeoGrid. 
            The student (GeoGrid from which the method is called)
            is aligned on the teacher (the GeoGrid given as argument).
            The extent of the resulting GeoGrid is corresponding to 
            the common area of the teacher and student.
            The cell-size of the GeoGrid returned is those of the teacher.
            If empty, None is returned.

        :params:
        --------
            teacher : GeoGrid - The teacher
            self : GeoGrid - The student
        
        :returns:
        ---------
            child : GeoGrid - res of teacher and extent of the intersection between teacher and student
        """

        # Get reference grid x & y resolutions
        xRes = teacher.xRes
        yRes = teacher.yRes
        crs = teacher.crs

        # Intersect student and teacher
        x = self.getShapelyExtent().intersection(teacher.getShapelyExtent())
        xMin, yMin, xMax, yMax = x.bounds
        xLen = xMax - xMin
        yLen = yMax - yMin

        # Compute number of rows and cols
        nCols = math.ceil(abs(xLen / xRes))
        nRows = math.ceil(abs(yLen / yRes))

        # Compute grid dimensions
        return GeoGrid(xMin, yMax, xRes, yRes, nCols, nRows, crs)

def createGeoGridFromArray(array, xMin, yMax, xRes, yRes, crs):
    
    # Find nRows and nCols from array
    dims = len(array.shape)
    if dims == 2:
        nRows, nCols = array.shape
    elif dims == 3:
        _, nRows, nCols = array.shape
    
    # Create Geogrid
    return GeoGrid(xMin, yMax, xRes, yRes, nCols, nRows, crs=crs)

