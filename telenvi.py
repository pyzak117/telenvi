# Standard librairies
import os
import re

# Third-Party librairies
import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, ogr
from matplotlib import pyplot as plt

class GeoIm:

    """
    Describe a georeferenced raster.
    
    attributes
    ----------
        pxlV (np.ndarray) : an array representing the pixels values
        geoData (tuple) : a tuple like
            (
                pixel_width, 
                pixel_height,
                originX,
                originY
            )
        crs : a big string representing the coordinates reference system
    """

    def __init__(self, pxlV, geoData, crs):
        self.pxlV = pxlV
        self.geoData = geoData
        self.crs = crs

    def __add__(self, neighboor):
        res = self.pxlV + neighboor.pxlV
        x = GeoIm(res, self.geoData, self.crs)
        return x
    
    def __sub__(self, neighboor):
        res = self.pxlV - neighboor.pxlV
        x = GeoIm(res, self.geoData, self.crs)
        return x

    def __mul__(self, neighboor):
        res = self.pxlV * neighboor.pxlV
        x = GeoIm(res, self.geoData, self.crs)
        return x

    def __truediv__(self, neighboor):
        res = self.pxlV / neighboor.pxlV
        x = GeoIm(res, self.geoData, self.crs)
        return x

    def __repr__(self):
        self.quickVisual()
        return ""

    def getCoordsExtent(self):
        """
        :return:
        --------
            bounding-box coordinates of the GeoIm's spatial extent
        """

        dim = len(self.pxlV.shape)

        # Compute extent coordinates
        pixW, pixH, xLeft, yTop = self.geoData
        
        if dim == 2:
            rows, cols = self.pxlV.shape

        elif dim == 3:
            _, rows, cols = self.pxlV.shape
        xRight = xLeft+cols*pixW
        yBottom = yTop+rows*pixH

        return (xLeft, yTop, xRight, yBottom)

    def getGeomExtent(self):
        """
        :return:
        --------
            a ogr.Geometry object which represent the spatial extent of the GeoIm
        """

        # Get bounding box coordinates
        xLeft, yTop, xRight, yBottom = self.getCoordsExtent()

        # Create a ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xLeft, yTop)
        ring.AddPoint(xLeft, yBottom)
        ring.AddPoint(xRight, yBottom)
        ring.AddPoint(xRight, yTop)
        ring.AddPoint(xLeft, yTop)

        # Assign this ring to a polygon
        polygon_env = ogr.Geometry(ogr.wkbPolygon)
        polygon_env.AddGeometry(ring)

        return polygon_env

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
        dim = len(self.pxlV.shape)

        if dim == 2:
            nb_bands = 1
            rows, cols = self.pxlV.shape

        elif dim == 3:
            nb_bands, rows, cols = self.pxlV.shape

        else:
            raise ValueError("Array must be in 2 or 3 dimensions")

        # gdal.Dataset creation
        outDs = driver.Create(outP, cols, rows, nb_bands, format)
        outDs.SetGeoTransform((self.geoData[2], self.geoData[0], 0.0, self.geoData[3], 0.0, self.geoData[1]))
        outDs.SetProjection(self.crs)

        # Export each band of the 3D array
        if dim == 3:
            for band in range(1, nb_bands+1):
                outDs.GetRasterBand(band).WriteArray(self.pxlV[band-1])

        # Export the unique band
        else:
            outDs.GetRasterBand(1).WriteArray(self.pxlV)

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

        if len(self.pxlV.shape) == 2:
            plt.imshow(self.pxlV, cmap = colors)

        elif len(self.pxlV.shape) == 3:
            plt.imshow(self.pxlV[band], cmap = colors)

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
    ):

    """
    Make a GeoIm object from a georeferenced raster file.

    :param:
    -------
        targetP (str) : the path to the raster you want to load
        indexToLoad (int or list) : if the file is a stack, give the band or the bands you want to load here
        roi (str or list) : a list of 2 tuples [(x,y),(x,y)] representing the top-left corner and bottom-right corner of a region of interest or a path to a shapefile containing squared polygone(s)
        ft (int) : if roi is a path to a shapefile, ft give the index in the attribute table of the feature you want to use as ROI
        crs (str) : if you want to reproject the image, this parameter is a string describing the coordinates reference system of the image
        res (int or float) : if you want to resample the image, you give the new resolution here. The unit of the value must be in the unit of the target crs.
        algo (str) : the resample algorithm you want to use
        format (np.dtype) : the numeric format of the pixels values
    
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
            if ft == None:
                raise ValueError("error 6 : ft empty")

            # Shapefile loading
            layer = gpd.read_file(roi)

            # Square check
            # ...

            # Feature geometry extraction
            geom = layer["geometry"][ft].bounds
            XMIN, YMAX = geom[0], geom[3]
            XMAX, YMIN = geom[2], geom[1]
            
            # Crop mode activate
            CROP = True
    
    # Check reprojection mode
    REPROJ = False
    if crs != None:
        if type(crs) != str:
            raise ValueError("error 7 : The destination crs must be a str")
        REPROJ = True

    # Check resample mode
    RESAMPLE = False
    if res != None:
        if type(res) != float and type(res) != int:
            raise ValueError("error 8 : The resolution must be a number")
        RESAMPLE = True
    
    # ----------------
    # # Loading data #
    # ----------------

    inDs = gdal.Open(targetP)

    if REPROJ: inDs = gdal.Warp("", inDs, format = "VRT", dstSRS = crs)

    if RESAMPLE: inDs = gdal.Warp("", inDs, format = "VRT", xRes = res, yRes = res, resampleAlg = algo)

    # Get geographic data from the dataset
    geoTransform = inDs.GetGeoTransform() # Decrsibe geographic area of the full image
    projection = inDs.GetProjection() # The big Str which decrsibe the Coordinates Reference System

    # Unpack geoTransform of the full image
    orX = geoTransform[0]
    orY = geoTransform[3]
    widthPix = geoTransform[1]
    heightPix = geoTransform[5]

    if CROP:
        # Transform geographic coordinates of the zone of interest into pixels coordinates
        row1 = int((YMAX-orY)/heightPix)
        col1 = int((XMIN-orX)/widthPix)
        row2 = int((YMIN-orY)/heightPix)
        col2 = int((XMAX-orX)/widthPix)

        # Update the geotransform
        orX = orX + (col1 * widthPix)
        orY = orY + (row1 * heightPix)

    else:
        row1 = 0
        col1 = 0
        row2 = inDs.RasterYSize - 1
        col2 = inDs.RasterXSize - 1

    # get array(s) from the dataset
    if BANDSMODE == 0:
        pxlV = inDs.ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(numFormat)

    elif BANDSMODE == 1:
        pxlV = inDs.GetRasterBand(indexToLoad).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(numFormat)

    elif BANDSMODE == 2:
        band1 = inDs.GetRasterBand(indexToLoad[0]).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(numFormat)
        pxlV = np.array([band1])
        for index in indexToLoad[1:]:
            band = inDs.GetRasterBand(index).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(numFormat)
            pxlV = np.append(pxlV, [band], axis=0)

    print(os.path.basename(targetP + " loaded"))

    return GeoIm(pxlV, (widthPix, heightPix, orX, orY), projection)

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
        list_GeoIm (list) : a list containing 2 or more GeoIm objects. They must have the same geoData and the same CRS.
    
    :returns:
    ---------
        a new GeoIm object with pxlV in 3 dimensions. The number of the bands is the same than the order of the GeoIm in list_GeoIm.
    """
    
    # Inputs Checking
    try :
        std_geoData = list_GeoIm[0].geoData
        std_crs = list_GeoIm[0].crs

        for elt in list_GeoIm:
            if std_geoData != elt.geoData or std_crs != elt.crs:
                raise ValueError("All the GeoIm objects must have the same resolution, the same origin x and y and the same CRS")
    
    except AttributeError:
        print("error 13: stackGeoIm take a list of GeoIm objects")

    # Process
    B1 = list_GeoIm[0].pxlV
    stack = np.array([B1])
    for elt in list_GeoIm[1:]:
        stack = np.append(stack, [elt.pxlV], axis=0)

    # Return    
    return GeoIm(stack, std_geoData, std_crs)
