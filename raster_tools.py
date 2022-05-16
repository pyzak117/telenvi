#%% Standard librairies
import os
import re

# Third-Party librairies
import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, ogr, osr
from matplotlib import pyplot as plt
import shapely.geometry

class GeoGrid:
    """
    Describe a georeferenced data grid.

    attributes
    ----------
        matrixian data:
            rows
            cols

        geographic data:
            crs
            xLeft
            xRight
            yTop
            yBottom
            cellLengthX
            cellLengthY

    methods
    -------
        to_ogr(self)
        to_shapely(self)

    """

    def __init__(self, xLeft, yTop, cellLengthX, cellLengthY, rows, cols, crs = None):

        # Get known geographics data
        self.crs = crs
        self.xLeft = xLeft
        self.yTop = yTop
        self.cellLengthX = cellLengthX
        self.cellLengthY = cellLengthY

        # Get known matrixian data
        self.rows = rows
        self.cols = cols

        # Compute xRight & yBottom
        self.xRight = xLeft + cols * cellLengthX
        self.yBottom = yTop + rows * cellLengthY

        # Compute shapely geometry extent
        self.shplyGeom = shapely.geometry.Polygon(
            [(self.xLeft, self.yTop),(self.xLeft, self.yBottom),(self.xRight, self.yBottom),(self.xRight, self.yTop),(self.xLeft, self.yTop)]
        )

    def compute_ogr_extent(self):
        """
        :return:
        --------
            a ogr.Geometry object which represent the spatial extent of the GeoGrid
        """

        # Create a ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(self.xLeft, self.yTop)
        ring.AddPoint(self.xLeft, self.yBottom)
        ring.AddPoint(self.xRight,self.yBottom)
        ring.AddPoint(self.xRight,self.yTop)
        ring.AddPoint(self.xLeft, self.yTop)

        # Assign this ring to a polygon
        polygon_env = ogr.Geometry(ogr.wkbPolygon)
        polygon_env.AddGeometry(ring)

        return polygon_env

    def extent_to_shapefile(self, out_path):
        """
        :return:
        --------
            None. Create a shapefile containing the extent to the given path.
        """
        line_data = gpd.GeoSeries({"geometry":self.shplyGeom})
        gdf_extent = gpd.GeoDataFrame([line_data])
        gdf_extent.set_crs(self.crs, inplace=True)
        gdf_extent.to_file(out_path)
        return None

    def compute_gdf_grid(self, out_path=None):
        """
        :return:
        --------
            a GeoDataFrame where each line represent a square grid cell
        """

        cells = []
        for row in range(self.rows):
            # print("row {}/{}".format(row, self.rows-1))
            for col in range(self.cols):
                cell_xLeft = self.xLeft + self.cellLengthX * col
                cell_xRight = self.xLeft + self.cellLengthX * (col + 1)

                cell_yTop = self.yTop + self.cellLengthY * row
                cell_yBottom = self.yTop + self.cellLengthY * (row + 1)

                cell_data = {}
                cell_geom = [
                    (cell_xLeft, cell_yTop),
                    (cell_xLeft, cell_yBottom),
                    (cell_xRight, cell_yBottom),
                    (cell_xRight, cell_yTop),
                    (cell_xLeft, cell_yTop)]

                cell_data["geometry"] = shapely.geometry.Polygon(cell_geom)
                cells.append(gpd.GeoSeries(cell_data))
        grid = gpd.GeoDataFrame(cells)
        grid.set_crs(self.crs, inplace=True)

        if out_path != None:
            grid.to_file(out_path)
        return grid

    def common_extent_with(self, neighboor):
        return self.shplyGeom.intersection(neighboor.shplyGeom)

def print_shapely_geom(shplyGeom, out_path, crs):
    """
    :return:
    --------
        None. Create a shapefile containing the extent of the shapely object to the given path.
    """
    line_data = gpd.GeoSeries({"geometry":shplyGeom})
    gdf_extent = gpd.GeoDataFrame([line_data])
    gdf_extent.set_crs(crs, inplace=True)
    gdf_extent.to_file(out_path)
    return None

class GeoIm:
    """
    Describe a georeferenced raster.
    
    attributes
    ----------
        pxData (np.ndarray) : an array representing the pixels values
        geoGrid (tuple) : a GeoGrid object representing the pixels grid
    """

    def __init__(self, pxData, geoGrid):
        self.pxData = pxData
        self.geoGrid = geoGrid

    def __add__(self, neighboor):
        res = self.pxData + neighboor.pxData
        x = GeoIm(res, self.geoGrid)
        return x
    
    def __sub__(self, neighboor):
        res = self.pxData - neighboor.pxData
        x = GeoIm(res, self.geoGrid)
        return x

    def __mul__(self, neighboor):
        res = self.pxData * neighboor.pxData
        x = GeoIm(res, self.geoGrid)
        return x

    def __truediv__(self, neighboor):
        res = self.pxData / neighboor.pxData
        x = GeoIm(res, self.geoGrid)
        return x

    def __repr__(self):
        self.quickVisual()
        return ""

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
        outDs.SetGeoTransform((self.geoGrid.xLeft, self.geoGrid.cellLengthX, 0.0, self.geoGrid.yTop, 0.0, self.geoGrid.cellLengthY))
        outDs.SetProjection(self.geoGrid.crs)

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
        Show the pixels values of a GeoIm object.

        :param:
        -------
            band (int) : if the array of pixels values represent a multispectral image, with 3 dimensions, you can choose the band than you want to show here.
            colors (str) : a string describing the color-range you want to use to show the image
        """

        if len(self.pxData.shape) == 2:
            plt.imshow(self.pxData, cmap = colors)

        elif len(self.pxData.shape) == 3:
            plt.imshow(self.pxData[band], cmap = colors)

        plt.show()
        plt.close()
        return None

def openGeoRaster(
    targetP,
    indexToLoad = None,
    roi = None,
    ft = 0,
    crs = None,
    res = None,
    algo = "near",
    numFormat = np.float32,
    geoGridMode = False
    ):

    """
    Make a GeoIm object from a georeferenced raster file.

    :param:
    -------
        targetP (str) : the path to the raster you want to load
        indexToLoad (int or list) : if the file is a stack, give the band or the bands you want to load here
        roi if (list) : a list of 2 tuples [(x,y),(x,y)] representing the top-left corner and bottom-right corner of a region of interest
            if (str) : a path to a shapefile containing polygone(s) or a path to an other raster. The GeoIm will be clip onto the extent of this raster.
        ft (int) : if roi is a path to a shapefile, ft give the index in the attribute table of the feature you want to use as ROI
        crs (int) : EPSG of a desired Coordinates Reference System (for WGS84, it's 4326 for example)
        res (int or float) : if you want to resample the image, you give the new resolution here. The unit of the value must be in the unit of the target crs.
        algo (str) : the resample algorithm you want to use. Resample is computed with gdal.Warp(), so see the gdal api documentation to see the others available methods.
        format (np.dtype) : the numeric format of the array values
        geoGridMode (bool) : if it's True, the function will just return the geoGrid associated to the target raster
    
    :return:
    --------
        a GeoIm object
    """

    # -------------------
    # # Inputs checking #
    # -------------------
    
    # check target validity
    if not os.path.exists(targetP):
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
            XMIN, YMAX = roi[0]
            XMAX, YMIN = roi[1]

            # Check coordinates logic validity
            if XMIN >= XMAX or YMIN >= YMAX :
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
                geom = layer["geometry"][ft].bounds
                XMIN, YMAX = geom[0], geom[3]
                XMAX, YMIN = geom[2], geom[1]

            else:
                try:
                    # get the common area between data_image and crop_image
                    ds = gdal.Open(roi)
                    XMIN, xPixSize, _, YMAX, _, yPixSize = ds.GetGeoTransform()
                    XMAX = XMIN + (xPixSize * (ds.RasterXSize))
                    YMIN = YMAX + (yPixSize * (ds.RasterYSize))
                
                except AttributeError:
                    print("error 6.2 : invalid raster region of interest")

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
    
    # Check geoGridMode
    if type(geoGridMode) != bool:
        print("geoGridMode must be a boolean")
    
    # ----------------
    # # Loading data #
    # ----------------

    inDs = gdal.Open(targetP)

    if REPROJ: inDs = gdal.Warp("", inDs, format = "VRT", dstSRS = crs)

    if RESAMPLE: inDs = gdal.Warp("", inDs, format = "VRT", xRes = res, yRes = res, resampleAlg = algo)

    # Get geographic data from the dataset
    geoTransform = inDs.GetGeoTransform() # Describe geographic area of the full image
    projection = inDs.GetProjection() # The big string which describe the Coordinates Reference System

    # Unpack geoTransform of the full image
    orX = geoTransform[0]
    orY = geoTransform[3]
    widthPix = geoTransform[1]
    heightPix = geoTransform[5]

    if CROP:
        # Transform geographic coordinates of the region of interest into matrix coordinates
        row1 = int((YMAX-orY)/heightPix)
        col1 = int((XMIN-orX)/widthPix)
        row2 = int((YMIN-orY)/heightPix)
        col2 = int((XMAX-orX)/widthPix)

        # Update the origine's coordinates
        orX = orX + (col1 * widthPix)
        orY = orY + (row1 * heightPix)

    else:
        row1 = 0
        col1 = 0
        row2 = inDs.RasterYSize - 1
        col2 = inDs.RasterXSize - 1

    geoGrid = GeoGrid(orX, orY, widthPix, heightPix, inDs.RasterYSize, inDs.RasterXSize, projection)

    if geoGridMode:
        print(os.path.basename(targetP + " geogrid loaded"))
        return geoGrid

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

    print(os.path.basename(targetP + " loaded"))

    return GeoIm(pxData, geoGrid)

def openManyGeoRasters(
    folder,
    pattern,
    endKeyPos = -4,
    indexToLoad = None,
    roi = None,
    ft = 0,
    crs = None,
    res = None,
    algo = "near",
    format = np.float32):

    """
    Make a dictionnary containing GeoIm objects for all the raster files in the folder, with a part of their name matching with the pattern. You can build the keys of the dictionnary by setting the pattern and the endKeyPos.

    :param:
    -------
        folder (str) : path to a directory containing many rasters
        pattern (str) : a regular expression pattern use to detect rasters files you want to load
        endKeyPos (int) : the position in the raster file name where you want to stop the key

    :return:
    --------
        x (dictionnary) : a dictionnary with GeoIm objects inside

    :example:
    ---------
        image_sentinel_20190118
            |__ T32PMV_20190118T095331_B01.jp2
            |__ T32PMV_20190118T095331_B01.jp2.xml
            |__ T32PMV_20190118T095331_B02.jp2
            |__ T32PMV_20190118T095331_B02.jp2.xml
            |__ T32PMV_20190118T095331_B03.jp2
            |__ T32PMV_20190118T095331_B04.jp2

        params :
        folder = image_sentinel_20190118
        pattern = "B[0-9]+.jp2$" >>> It means "a 'B' followed by a serie of numbers ('[0-9]+'), then by a point, then by 'jp2', then it's the end of the filename '$'. use this pattern allow the programm to avoid the metadata files, which ended with .xml
        endKeyPos = -4

        return:
            {"B02":GeoIm object with the data of T32PMV_20190118T095331_B02.jp2,
             "B03":GeoIm object with the data of T32PMV_20190118T095331_B03.jp2,
             "B04":GeoIm object with the data of T32PMV_20190118T095331_B04.jp2}
    """

    if pattern == None:
        raise ValueError("error 2 : undefined pattern")

    if type(pattern) != str:
        raise ValueError("error 3 : pattern must be str")

    if type(endKeyPos) != int:
        raise ValueError("error 5 : end key position must be integer")
    
    # Compile pattern with regular expression
    rpattern = re.compile(pattern.upper())

    x = {}
    for fileName in sorted(os.listdir(folder)):

        try : # Get pattern start position in fileName
            startKeyPos = re.search(rpattern, fileName.upper()).span()[0]

        except AttributeError: # Pattern not find in fileName
            continue
        
        fileBandName = os.path.join(folder, fileName)
        
        # Get the key corresponding to the pattern in the fileName
        bandId = fileName[startKeyPos:endKeyPos]

        # Extract and pack all the data in a lovely dictionnary with bandId as key
        x[bandId] = openGeoRaster(
            fileBandName,
            indexToLoad,
            roi,
            ft,
            crs,
            res,
            algo,
            format)
        
        print(bandId + " loaded")

    return x

def stackGeoIm(list_GeoIm):
    """
    Make a stack from many GeoIm objects

    :params:
    --------
        list_GeoIm (list) : a list containing 2 or more GeoIm objects. They must have the same geoGrid and the same CRS.
    
    :returns:
    ---------
        a new GeoIm object with pxData in 3 dimensions. The number of the bands is the same than the order of the GeoIm in list_GeoIm.
    """
    
    # Inputs Checking
    try :
        std_geoGrid = list_GeoIm[0].geoGrid
        std_crs = list_GeoIm[0].crs

        for elt in list_GeoIm:
            if std_geoGrid != elt.geoGrid or std_crs != elt.crs:
                raise ValueError("All the GeoIm objects must have the same resolution, the same origin x and y and the same CRS")
    
    except AttributeError:
        print("error 13: stackGeoIm take a list of GeoIm objects")

    # Process
    B1 = list_GeoIm[0].pxData
    stack = np.array([B1])
    for elt in list_GeoIm[1:]:
        stack = np.append(stack, [elt.pxData], axis=0)

    # Return    
    return GeoIm(stack, std_geoGrid, std_crs)
# %%
