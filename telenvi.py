#%%
"""
telenvi module
---------------
Version = 2.0
Fev. 2022
"""

VERSION = 2.0

print("\n---------\nTELENVI MODULE " + str(VERSION) + "\n---------\n")

# Standard librairies
import os
import re

# Third-Party librairies
import numpy as np
from osgeo import gdal
from osgeo import gdalconst
import geopandas as gpd
from matplotlib import pyplot as plt

class GeoIm:

    def __init__(self, array, pxlW, pxlH, orX, orY, crs):
        self.pxlV = array
        self.pxlW = pxlW
        self.pxlH = pxlH
        self.orX = orX
        self.orY = orY
        self.crs = crs

    def exportAsRaster(
        self,
        outP,
        format = gdalconst.GDT_Float32,
        driverName = "GTiff"):

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
        outDs.SetGeoTransform((self.orX, self.pxlW, 0.0, self.orY, 0.0, self.pxlH))
        outDs.SetProjection(self.crs)

        # Export each band of the 3D array
        if dim == 3:
            for band in range(1, nb_bands+1):
                print(self.pxlV[band-1])
                outDs.GetRasterBand(band).WriteArray(self.pxlV[band-1])
                outDs.GetRasterBand(band).SetNoDataValue(10000)

        # Export the unique band
        else:
            outDs.GetRasterBand(1).WriteArray(self.pxlV)

        outDs.FlushCache()
        print("\n" + os.path.basename(outP) + " OK")
        return None

    def quickVisual(self, bande = 0, colors = "viridis"):

        if len(self.pxlV.shape) == 2:
            plt.imshow(self.pxlV, cmap = colors)

        elif len(self.pxlV.shape) == 3:
            plt.imshow(self.pxlV[bande], cmap = colors)

        plt.show()
        plt.close()
        return ""

    def __add__(self, neighboor):
        res = self.pxlV + neighboor.pxlV
        x = GeoIm(res, self.pxlW, self.pxlH, self.orX, self.orY, self.crs)
        return x
    
    def __sub__(self, neighboor):
        res = self.pxlV - neighboor.pxlV
        x = GeoIm(res, self.pxlW, self.pxlH, self.orX, self.orY, self.crs)
        return x

    def __mul__(self, neighboor):
        res = self.pxlV * neighboor.pxlV
        x = GeoIm(res, self.pxlW, self.pxlH, self.orX, self.orY, self.crs)
        return x

    def __truediv__(self, neighboor):
        res = self.pxlV / neighboor.pxlV
        x = GeoIm(res, self.pxlW, self.pxlH, self.orX, self.orY, self.crs)
        return x

    def __repr__(self):
        self.quickVisual()
        return ""

def openGeoRaster(
    targetP,
    format = np.float32,
    indexToLoad = None,
    roi = None,
    ft = 0,
    crs = None,
    res = None,
    algo = "near"):

    """
    -------------------
    # Inputs checking #
    -------------------
    """
    
    # check neighboor validity
    if not os.path.exists(targetP):
        raise ValueError("error 1 : invalid neighboor path")
    
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
                    raise ValueError("error 2 : neighboor_array index must be integer or list of integers")
            BANDSMODE = 2
        
        else:
            raise ValueError("error 2 : neighboor_array index must be integer or list of integers")

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
    
    """
    ----------------
    # Loading data #
    ----------------
    """

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
        pxlV = inDs.ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(format)

    elif BANDSMODE == 1:
        pxlV = inDs.GetRasterBand(indexToLoad).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(np.float32)

    elif BANDSMODE == 2:
        band1 = inDs.GetRasterBand(indexToLoad[0]).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(np.float32)
        pxlV = np.array([band1])
        for index in indexToLoad[1:]:
            band = inDs.GetRasterBand(index).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(np.float32)
            pxlV = np.append(pxlV, [band], axis=0)

    return GeoIm(pxlV, widthPix, heightPix, orX, orY, projection)

def openManyGeoRasters(
    folder,
    pattern,
    endKeyPos = -4,
    format = np.float32,
    indexToLoad = None,
    roi = None,
    ft = 0,
    crs = None,
    res = None,
    algo = "near"):

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
        print(bandId)

        # Extract and pack all the data in a lovely dictionnary with bandId as key
        x[bandId] = openGeoRaster(
            fileBandName,
            format,
            indexToLoad,
            roi,
            ft,
            crs,
            res,
            algo)

    return x

def stackGeoIm(list_GeoIm):
    
    # Inputs Checking
    try :
        standard_res = list_GeoIm[0].pxlW
        standard_crs = list_GeoIm[0].crs
        standard_orX = list_GeoIm[0].orX
        standard_orY = list_GeoIm[0].orY

        for elt in list_GeoIm:
            if standard_res != elt.pxlW or standard_crs != elt.crs or standard_orX != elt.orX or standard_orY != elt.orY:
                raise ValueError("Al lthe GeoIm objects must have the same resolution, the same CRS and the same origin")
    
    except AttributeError:
        print("error 13: stackGeoIm take a list of GeoIm objects")

    # Process
    B1 = list_GeoIm[0].pxlV
    stack = np.array([B1])
    for elt in list_GeoIm[1:]:
        stack = np.append(stack, [elt.pxlV], axis=0)

    # Return    
    return GeoIm(stack, standard_res, standard_res, standard_orX, standard_orY, standard_crs)
