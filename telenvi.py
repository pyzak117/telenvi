#%%
# -*- coding: utf-8 -*-

"""
telenvi module
---------------
Version = 1.2
Jan. 2022
"""

# Standard librairies
import os
import re
import sys

# Third-Party librairies
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import *
import geopandas as gpd

def load_bands(fPath, 
               reflectance = False,
               pattern = None,
               endKeyPos = 1000,
               cropZone = None,
               featurePos = None,
               verbose = False):

    # 1. Inputs checking
    ####################

    if pattern == None:
        raise ValueError("error 2 : undefined pattern")

    if type(pattern) != str:
        raise ValueError("error 3 : pattern must be str")
    
    if type(endKeyPos) != int:
        raise ValueError("error 5 : end key position must be integer")

    CROP = False
    if cropZone != None:
        if type(cropZone) == list:

            if len(cropZone) != 2:
                raise ValueError("error 6 : Coordinates of cropZone must be a list of 2 tuples")
            
            if len(cropZone[0]) != 2 or len(cropZone[1]) != 2:
                raise ValueError("error 7 : cropZone tuples must have 2 numbers")

            # Unpack crop coordinates
            xMin, yMax = cropZone[0]
            xMax, yMin = cropZone[1]

            # Check coordinates logic validity
            if xMin >= xMax or yMin >= yMax :
                raise ValueError("error 8 : Coordinates are invalid")

            # Crop mode activate
            CROP = True

        elif type(cropZone) == str:
            if featurePos == None:
                raise ValueError("featurePos must be filled")

            # Shapefile loading
            layer = gpd.read_file(cropZone)

            # Feature geometry extraction
            geom = layer["geometry"][featurePos].bounds
            xMin, yMax = geom[0], geom[3]
            xMax, yMin = geom[2], geom[1]
            
            # Crop mode activate
            CROP = True

    # Compile pattern with regular expression
    rpattern = re.compile(pattern)

    # 2. Loading data
    ################

    bands = {}
    geoData = False

    ("Bands research and loading")
    for fileName in os.listdir(fPath):
        if verbose:
            print(fileName)
        try :
            # get pattern start position in filename
            startKeyPos = re.search(rpattern, fileName.upper()).span()[0]
        
        except AttributeError:
            if verbose:
                print("motif non trouve")
            continue # pattern not in filename : switch to next file

        fileBandName = os.path.join(fPath, fileName)
        ds = gdal.Open(fileBandName)
        bandId = fileName[startKeyPos:endKeyPos] # "imageNameB12.tif"[pos:-4] = "B12"

        if not geoData:

            # Get geographic data from the dataset
            geoTransform = ds.GetGeoTransform()
            projection = ds.GetProjection()

            if CROP:
                # Unpack geotransform
                orX = geoTransform[0] # or > origine
                orY = geoTransform[3]
                widthPix = geoTransform[1]
                heightPix = geoTransform[5]

                # Zone framing
                row1=int((yMax-orY)/heightPix)
                col1=int((xMin-orX)/widthPix)
                row2=int((yMin-orY)/heightPix)
                col2=int((xMax-orX)/widthPix)
                band = ds.ReadAsArray(col1,row1,col2-col1+1,row2-row1+1).astype(np.float32)

            else:
                band = ds.ReadAsArray().astype(np.float32)
        if CROP:
            crop_orX = xMin
            crop_orY = yMax
            nGeoTransform = (crop_orX, widthPix, 0.0, crop_orY, 0.0, heightPix)

        # stock array in dic bands with bandId as key
        if reflectance:
            bands[bandId] = band/10000

        else:
            bands[bandId] = band
        
        if verbose:
            print("{} loaded".format(bandId))

    if CROP:
        return [bands, nGeoTransform, projection]

    else:
        return [bands, geoTransform, projection]

def writeGeotiff(band, rows, cols, geoTransform, projection,outPath,driverName = "GTiff"):
    
    driver = gdal.GetDriverByName(driverName)
    outData = driver.Create(outPath, cols, rows, 1, gdal.GDT_Float32)
    outData.SetGeoTransform(geoTransform)
    outData.SetProjection(projection)
    outData.GetRasterBand(1).WriteArray(band)
    outData.GetRasterBand(1).SetNoDataValue(10000)
    outData.FlushCache()
    return None
