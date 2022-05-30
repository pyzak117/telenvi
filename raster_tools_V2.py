#%% Standard librairies
import os
import re
from turtle import colormode

# Third-Party librairies
import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, ogr, osr
from matplotlib import pyplot as plt
import shapely
from PIL import Image, ImageEnhance
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
        nCols = int(abs(xLen / xRes))
        nRows = int(abs(yLen / yRes))

        # Compute grid dimensions
        return GeoGrid(xMin, yMax, xRes, yRes, nCols, nRows, crs)

def grid_vs_array(geogrid, array):
    """
    Check the compatibility of a geogrid with an array
    """

    # Get the dimensions of the array array
    dims = len(array.shape)

    if dims == 2:
        data_nRows, data_nCols = array.shape

    elif dims == 3:
        data_nRows = array.shape[1]
        data_nCols = array.shape[2]

    else:
        raise ValueError("A array array must have 2 or 3 dimensions")

    # Compare geogrid attributes and array attributes
    cols_compatibility = data_nCols == geogrid.nCols
    rows_compatibility = data_nRows == geogrid.nRows

    # return a boolean
    return cols_compatibility == rows_compatibility == True

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
                              or 3 dimensions (x, y, channel) if the image is 
                              multispectral (just as the color images, with a
                              Red, a Green and a Blue channel)
        ----------------------------------------------------------------
        geogrid | GeoGrid  | describe the spatial position of the image,
                             in a Spatial Coordinates System given

    methods
    -------
        [...]

    """

    def geogrid_fit_on_pxData(self):
        return grid_vs_array(self.geogrid, self.pxData)

    def __init__(self, pxData, geogrid):        

        # Attribute initialisation
        self.pxData = pxData
        self.geogrid = geogrid

        # Check the compatibility of geogrid and the data array
        if not self.geogrid_fit_on_pxData():
            raise AttributeError("The array size and the geogrid size are not compatible")

    def __add__(self, neighboor):
        output = self.pxData + neighboor.pxData
        x = GeoIm(output, self.geogrid, self.crs)
        return x
    
    def __sub__(self, neighboor):
        output = self.pxData - neighboor.pxData
        x = GeoIm(output, self.geogrid, self.crs)
        return x

    def __mul__(self, neighboor):
        output = self.pxData * neighboor.pxData
        x = GeoIm(output, self.geogrid, self.crs)
        return x

    def __truediv__(self, neighboor):
        output = self.pxData / neighboor.pxData
        x = GeoIm(output, self.geogrid, self.crs)
        return x

    def __repr__(self):
        self.quickVisual()
        return ""
    
    def changeGeoGrid(self, new_grid, inplace=False):
        """
        When the geogrid of a GeoIm is modified, his data have to be aligned to the new_grid grid.
        The new_grid must be a part of the old. 
        """
        
        # Rename current instance attributes
        old_grid = self.geogrid
        old_data = self.pxData
    
        # Check the CRS compatibility between the 2 grids
        if old_grid.crs != new_grid.crs :
            raise AttributeError("The new_grid and old_grid geogrid have not the same CRS")

        # Intersect the grids
        # inter_grid = old_grid.intersect(teacher = new_grid)

        # Get bounds of the intersection between the new and the old geogrid
        firstRow= int((new_grid.yMax - old_grid.yMax) / old_grid.yRes)
        firstCol= int((new_grid.xMin - old_grid.xMin) / old_grid.xRes)
        lastRow = int((new_grid.yMin - old_grid.yMax) / old_grid.yRes)
        lastCol = int((new_grid.xMax - old_grid.xMin) / old_grid.xRes)

        # Find the resizes factor
        # resizeY_factor = new_grid.resY / old_grid.resY
        # resizeX_factor = new_grid.resX / old_grid.resX

        # Compute new number of Rows and Cols
        # new_nRows = math.ceil(inter_grid.nRows * resizeY_factor)
        # new_nCols = math.ceil(inter_grid.nCols * resizeX_factor)

        # Create PIL.Image instance from the data
        im_data = Image.fromarray(old_data)

        # Crop the data and change his resolution with pillow
        im_data_resize = im_data.resize(
            size = (new_grid.nCols, new_grid.nRows),
            resample = Image.Resampling.BICUBIC,
            box = (firstCol, firstRow, lastCol, lastRow)
        )

        # Get back to array
        new_pxData = np.asarray(im_data_resize)

        # Verify the compatibility between the new grid and the new pxData
        if not self.geogrid_fit_on_pxData():
            raise AttributeError("The array size and the geogrid size are not compatible")

        if inplace:
            # Change instance attributes
            self.pxData = new_pxData
            self.geogrid = new_grid
    
        else:
            return GeoIm(new_pxData, new_grid)

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
            return self.geogrid.getShapelyExtent()
    
    def exportAsRasterFile(
        self,
        outP,
        f_format = gdalconst.GDT_Float32,
        driverName = "GTiff"):

        """
        Export a GeoIm object into a raster file georeferenced.

        :params:
        --------
            outP (str) : the path where you want to save the raster
            f_format (gdalconst) : the image file format
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
        outDs = driver.Create(outP, cols, rows, nb_bands, f_format)
        outDs.SetGeoTransform((self.geogrid.xMin, self.geogrid.xRes, 0.0, self.geogrid.yMax, 0.0, self.geogrid.yRes))
        outDs.SetProjection(self.geogrid.crs)

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

        :params:
        -------
            band : int - if the array of pixels values represent a 
                         multispectral image, with 3 dimensions, 
                         you can choose the band than you want to show here.

            colors : str - a string describing the color-range to use 
                           to show the image
        """

        if len(self.pxData.shape) == 2:
            plt.imshow(self.pxData, cmap = colors)

        elif len(self.pxData.shape) == 3:
            plt.imshow(self.pxData[band], cmap = colors)

        plt.show()
        plt.close()
        return None

    def rgbVisual(self, colorMode=[0,1,2], resize_factor=1, brightness=1, show=False, path=None):
        """
        show the image in colors
        """

        if len(self.pxData.shape) != 3:
            raise AttributeError("You need a GeoIm in 3 dimensions to display a GeoIm in RGB")

        if self.pxData.shape[0] < 3:
            raise AttributeError("The GeoIm have only {} channel and we need 3 channels to display it in RGB")

        # Convert pxData into RGB array

        # Unpack the RGB components is separates arrays
        r = self.pxData[colorMode[0]]
        g = self.pxData[colorMode[1]]
        b = self.pxData[colorMode[2]]

        # data normalization between [0-1]
        r_norm = (r - r[r!=0].min()) / (r.max() - r[r!=0].min()) * 255
        g_norm = (g - g[g!=0].min()) / (g.max() - g[g!=0].min()) * 255
        b_norm = (b - b[b!=0].min()) / (b.max() - b[b!=0].min()) * 255

        # RGB conversion
        # --------------

        # Create a target array
        rgb_ar = np.zeros((self.geogrid.nRows, self.geogrid.nCols, 3))

        # For each cell of the "board"
        for row in range(self.geogrid.nRows):
            for col in range(self.geogrid.nCols):

                # We get the separate RGB values in each band
                r = r_norm[row][col]
                g = g_norm[row][col]
                b = b_norm[row][col]

                # We get them together in little array
                rgb_pixel = np.array([r,g,b])

                # And we store this little array on the board position
                rgb_ar[row][col] = rgb_pixel

        rgb = Image.fromarray(np.uint8(rgb_ar))

        # Adjust size
        rgb = rgb.resize((self.geogrid.nCols * resize_factor, self.geogrid.nRows * resize_factor))

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(rgb)
        rgb = enhancer.enhance(brightness)

        # Display
        if show:
            rgb.show()

        # Save
        if path != None:
            rgb.save(path)

        # Return PIL.Image instance
        return rgb

def openGeoRaster(
    target_path,
    indexToLoad = None,
    roi = None,
    ft = 0,
    crs = None,
    res = None,
    algo = "near",
    numFormat = np.float32,
    ):

    """
    Make a GeoIm object from a georeferenced raster file.

    :params:
    -------
        target_path (str) : the path to the raster you want to load
        indexToLoad (int or list) : if the file is a stack, give the band or the bands you want to load here
        roi if (list) : a list of 2 tuples [(x,y),(x,y)] representing the top-left corner and bottom-right corner of a region of interest
            if (str) : a path to a shapefile containing polygone(s) or a path to an other raster. The GeoIm will be clip onto the extent of this raster.
        ft (int) : if roi is a path to a shapefile, ft give the index in the attribute table of the feature you want to use as ROI
        crs (int) : EPSG of a desired Coordinates Reference System (for WGS84, it's 4326 for example)
        res (int or float) : if you want to resample the image, you give the new resolution here. The unit of the value must be in the unit of the target crs.
        algo (str) : the resample algorithm you want to use. Resample is computed with gdal.Warp(), so see the gdal api documentation to see the others available methods.
        format (np.dtype) : the numeric format of the array values
    
    :return:
    --------
        a GeoIm object
    """

    # -------------------
    # # Inputs checking #
    # -------------------
    
    # check target validity
    if not os.path.exists(target_path):
        raise ValueError("error 1 : invalid target path")
    
    # Check format validity
    # ...

    # Check the bands extraction mode
    BANDSMODE = 0
    if indexToLoad != None:

        if type(indexToLoad) == int:
            BANDSMODE = 1

        elif type(indexToLoad) == list:
            for element in indexToLoad:
                if type(element) != int:
                    raise ValueError("error 2 : target_array index must be integer or list of integers")
            BANDSMODE = 2
        
        else:
            raise ValueError("error 2 : target_array index must be integer or list of integers")

    # Check CROP mode
    CROP = False
    if roi != None:
        if type(roi) == list:

            if len(roi) != 2:
                raise ValueError("error 3 : Coordinates of roi must be a list of 2 tuples")
            
            if len(roi[0]) != 2 or len(roi[1]) != 2:
                raise ValueError("error 4 : roi tuples must have 2 numbers")

            # Unpack crop coordinates
            xMin, yMax = roi[0]
            xMax, yMin = roi[1]

            # Check coordinates logic validity
            if xMin >= xMax or yMin >= yMax :
                raise ValueError("error 5 : Coordinates are invalid")

            # Crop mode activate
            CROP = True

        elif type(roi) == str:
            if roi[-4:].lower() == ".shp":

                # check ft input
                if ft == None:
                    raise ValueError("error 6 : ft parameter is empty")

                # shapefile loading
                layer = gpd.read_file(roi)

                # Feature geometry extraction
                xMin, yMin, xMax, yMax = layer["geometry"][ft].bounds
                
            else :
                try:
                    # get spatial extent of the raster
                    ds = gdal.Open(roi)
                    xMin, xPixSize, _, yMax, _, yPixSize = ds.GetGeoTransform()
                    xMax = xMin + (xPixSize * ds.RasterXSize)
                    yMin = yMax + (yPixSize * ds.RasterYSize)

                except AttributeError:
                    print("error 6.2 : invalid raster to clip on")

            # Crop mode activate
            CROP = True
    
    # Check reprojection mode
    REPROJ = False
    if crs != None:
        if type(crs) != int:
            raise ValueError("error 7 : The destination crs must be a number")
        REPROJ = True
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(crs)
        crs = srs.ExportToWkt()

    # Check resample mode
    RESAMPLE = False
    if res != None:
        if type(res) != float and type(res) != int:
            raise ValueError("error 8 : The resolution must be a number")
        RESAMPLE = True

    # ----------------
    # # Loading data #
    # ----------------

    inDs = gdal.Open(target_path)

    if REPROJ: inDs = gdal.Warp("", inDs, format = "VRT", dstSRS = crs)

    if RESAMPLE: inDs = gdal.Warp("", inDs, format = "VRT", xRes = res, yRes = res, resampleAlg = algo)

    # Get geographic data from the dataset
    geoTransform = inDs.GetGeoTransform() # Describe geographic area of the full image
    projection = inDs.GetProjection() # The big string which describe the Coordinates Reference System

    # Unpack geoTransform of the full image
    orX = geoTransform[0]
    orY = geoTransform[3]
    xRes = geoTransform[1]
    yRes = geoTransform[5]

    if CROP:

        # Transform geographic coordinates of the region of interest into matrix coordinates
        row1 = int((yMax-orY)/yRes)
        col1 = int((xMin-orX)/xRes)
        row2 = int((yMin-orY)/yRes)
        col2 = int((xMax-orX)/xRes)

        # Update the origine's coordinates
        orX = orX + (col1 * xRes)
        orY = orY + (row1 * yRes)

    else:
        row1 = 0
        col1 = 0
        row2 = inDs.RasterYSize - 1 # avec un démarrage à 0 pour le référentiel "matriciel"
        col2 = inDs.RasterXSize - 1 # avec un démarrage à 0 pour le référentiel "matriciel"

    # get array(s) from the dataset
    if BANDSMODE == 0:
        pxData = inDs.ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(numFormat)

    elif BANDSMODE == 1:
        pxData = inDs.GetRasterBand(indexToLoad).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(numFormat)

    elif BANDSMODE == 2:
        band1 = inDs.GetRasterBand(indexToLoad[0]).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(numFormat)
        pxData = np.array([band1])
        for index in indexToLoad[1:]:
            band = inDs.GetRasterBand(index).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(numFormat)
            pxData = np.append(pxData, [band], axis=0)

    # Construction de la GeoGrid de positionnement spatial de l'image. 
    # On ajoute +1 à col2 et row2 pour repasser en référentiel "quantitatif pur" 
    geogrid = GeoGrid(orX, orY, xRes, yRes, col2-col1+1, row2-row1+1, projection)

    print(os.path.basename(target_path + " loaded"))
    return GeoIm(pxData, geogrid)
