#%%
# -*- coding: utf-8 -*-

"""
telenvi module
---------------
Version = 1.5
Fev. 2022
"""

VERSION = 1.5

print("\n---------\nTELENVI MODULE v" + str(VERSION) + "\n---------\n")

# Standard librairies
import os
import re

# Third-Party librairies
import numpy as np
from osgeo import gdal
from osgeo import gdalconst
import geopandas as gpd

def getGeoRasterData(target, 
               pattern = None,
               endKeyPos = -4,
               cropZone = None,
               indexToLoad = None,
               featurePos = None,
               new_proj = None,
               new_res = None,
               rs_method = "near"
               ):

    """
    :descr:
        Transform a geospatialised raster file in numpy.array object
        You can specify an area with a shapefile or a list of coordinates to load the raster only on those area.
        You can specify instructions to change the pixel size or the Coordinates System Reference of the target image.
    
    :param:
        # Mandatory
        target (str) = the file to open, or the directory containing one or many files to open

        # Optionnal
        pattern (str) = a regular expression pattern corresponding to all the files you want to load contained in the target. If a string match, it will be the key to find the file in the dictionnary returned.
        endKeyPos (int) = help you to build keys easy to understand. By default -4, just to exclude the extension of the file.

        indexToLoad (int or list) = indexes of the bands you want load - Default : all the bands are load
        
        cropZone (list) or (str) = a list of coordinates to crop the file or a path to a shapefile which contain a square to crop. If it's a shapefile, it must be a square. The coordinates or the shapefile must be in the same SCR than the new SCR - Default : All the image is load
        featurePos (int) = the position of the polygone in the attribute table of shapefile.

        new_proj (str) = String describing the target Coordinates System Reference
        
        new_res (float) = The new pixel size. If SCR have to be modified to, resolution must be in target SCR unit (degrees or metres)
        rs_method (str) = The resample algorithm. By default, near for nearest neighboor. Must be choose among {near, bilinear, cubic, cubicspline, lanczos, average, mode, max,  min,  med, q1, q3, sum}
        
        verbose (int) = talkative-level of the function: if it's 0, only primary information will be print during the bands loading. If it's 2, many information.

    :returns:
        bands (dictionnary) = contain all the file corresponding to the pattern, organised with the str combinaison which match the pattern.
        geoTransform (list) = contain the top left corner and bottom right corner coordinates, the pixel width and the pixel height of the first file loaded
        projection (proj) = SCR of the file
        
    """

    # 1. Inputs checking
    ####################

    # check target validity
    if not os.path.exists(target):
        raise ValueError("error 1 : unvalid target")

    # check target reference (file or directory)
    if os.path.isdir(target):
        if pattern == None:
            raise ValueError("error 2 : undefined pattern")

        if type(pattern) != str:
            raise ValueError("error 3 : pattern must be str")

        if type(endKeyPos) != int:
            raise ValueError("error 5 : end key position must be integer")
       
        # Compile pattern with regular expression
        rpattern = re.compile(pattern.upper())
    
        # DIR mode activation
        MODE = "DIR"

    if os.path.isfile(target):
        MODE = "FILE"

    # Check the bands extraction mode
    BANDSMODE = 0
    if indexToLoad != None:

        if type(indexToLoad) == int:
            BANDSMODE = 1

        elif type(indexToLoad) == list:
            for element in indexToLoad:
                if type(element) != int:
                    raise ValueError("error 12 : target_array index must be integer or list of integers")
            BANDSMODE = 2
        
        else:
            raise ValueError("error 12 : target_array index must be integer or list of integers")

    # Check CROP mode
    CROP = False
    if cropZone != None:
        if type(cropZone) == list:

            if len(cropZone) != 2:
                raise ValueError("error 6 : Coordinates of cropZone must be a list of 2 tuples")
            
            if len(cropZone[0]) != 2 or len(cropZone[1]) != 2:
                raise ValueError("error 7 : cropZone tuples must have 2 numbers")

            # Unpack crop coordinates
            XMIN, YMAX = cropZone[0]
            XMAX, YMIN = cropZone[1]

            # Check coordinates logic validity
            if XMIN >= XMAX or YMIN >= YMAX :
                raise ValueError("error 8 : Coordinates are invalid")

            # Crop mode activate
            CROP = True

        elif type(cropZone) == str:
            if featurePos == None:
                raise ValueError("error 9 : featurePos empty")

            # Shapefile loading
            layer = gpd.read_file(cropZone)

            # Feature geometry extraction
            geom = layer["geometry"][featurePos].bounds
            XMIN, YMAX = geom[0], geom[3]
            XMAX, YMIN = geom[2], geom[1]
            
            # Crop mode activate
            CROP = True
    
    # Check reprojection mode
    REPROJ = False
    if new_proj != None:
        if type(new_proj) != str:
            raise ValueError("error 10 : The destination SCR must be a str")
        REPROJ = True

    # Check resample mode
    RESAMPLE = False
    if new_res != None:
        if type(new_res) != float and type(new_res) != int:
            raise ValueError("error 11 : The resolution must be a number")
        RESAMPLE = True
    
    # 2. Loading data
    ################

    def extractDataFromGdalDs(fileName):
        inDs = gdal.Open(fileName)

        if REPROJ:
            inDs = gdal.Warp("", inDs, format = "VRT", dstSRS = new_proj)

        if RESAMPLE:
            inDs = gdal.Warp("", inDs, format = "VRT", xRes = new_res, yRes = new_res, resampleAlg=rs_method)

        # Get geographic data from the dataset
        geoTransform = inDs.GetGeoTransform() # Describe geographic area of the full image
        projection = inDs.GetProjection() # The big Str which describe the Coordinates Reference System

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

            # Get the coordinates of entire pixels around the area limits
            crop_orX = orX + (col1 * widthPix)
            crop_orY = orY + (row1 * heightPix)

            # Update the geotransform
            geoTransform = (crop_orX, widthPix, 0.0, crop_orY, 0.0, heightPix)

        else:
            row1 = 0
            col1 = 0
            row2 = inDs.RasterYSize - 1
            col2 = inDs.RasterXSize - 1

        # get array(s) from the dataset
        if BANDSMODE == 0:
            stack = inDs.ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(np.float32)
            return stack, geoTransform, projection

        elif BANDSMODE == 1:
            target_array = inDs.GetRasterBand(indexToLoad).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(np.float32)
            return target_array, geoTransform, projection

        elif BANDSMODE == 2:
            # Extract the first band
            first_array = inDs.GetRasterBand(indexToLoad[0]).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(np.float32)

            # Create a 3D array and store the first band in it
            target_array = np.array([first_array])

            # Add all the others bands in the stack
            for index in indexToLoad[1:]:
                new_array = inDs.GetRasterBand(index).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(np.float32)
                target_array = np.append(target_array, [new_array], axis=0)

        return target_array, geoTransform, projection

    if MODE == "DIR":
        data = {}

        for fileName in os.listdir(target):

            try : # Get pattern start position in fileName
                startKeyPos = re.search(rpattern, fileName.upper()).span()[0]

            except AttributeError: # Pattern not find in fileName
                continue
            
            fileBandName = os.path.join(target, fileName)
            
            # Get the key corresponding to the pattern in the fileName
            bandId = fileName[startKeyPos:endKeyPos]

            # Extract and pack all the data in a lovely dictionnary with bandId as key
            data[bandId] = extractDataFromGdalDs(fileBandName)

            # Informations to the user
            if CROP:
                print(fileName + " crop and loaded with key " + bandId)
            else:
                print(fileName + " loaded with key " + bandId)

    elif MODE == "FILE":
        data = extractDataFromGdalDs(target)

        # Informations to the user
        if CROP:
            print(os.path.basename(target) + " loaded on cropZone")
        else:
            print(os.path.basename(target) + " loaded")

    return data

def createGeoRasterFile(array,
                  geoTransform,
                  projection,
                  outPath,
                  format = gdalconst.GDT_Float32,
                  driverName = "GTiff"):
    
    """
    :descr:
        Print a numpy.array into geospatialised raster

    :param:
        # Mandatory
        array (numpy.array) = the array you want to print. It must be have 2 (one image) or 3 dimensions (a stack). If it's a stack, all the bands must have the same shape.
        geoTransform (tuple) = a tuple containing all the geographic settings you want to give to printed raster
        projection (str) = a str describing the Coordinates Reference System of the printed raster
        outPath (str) = the location where write the raster

        # Optionnal
        format (gdal.gdalconst) = numeric format of the raster pixels values - default value = "GDT_Float32"
        driverName (str) = rasterfile format - default value = "GTiff"

    :returns:        
        None
    """

    # Load the file-format driver asked
    driver = gdal.GetDriverByName(driverName)

    # Get dimensions of the array (stack or just a band)
    ar_dimensions = len(array.shape)

    # size of each dimension settings
    if ar_dimensions == 2:
        nb_bands = 1
        rows, cols = array.shape
    
    elif ar_dimensions == 3:
        nb_bands, rows, cols = array.shape
    
    else:
        raise ValueError("Array to write must be in 2 or 3 dimensions")

    # gdal.Dataset creation
    outDs = driver.Create(outPath, cols, rows, nb_bands, format)

    # Geodata settings to the new gdal.Dataset
    outDs.SetGeoTransform(geoTransform)
    outDs.SetProjection(projection)

    if ar_dimensions == 3:
        # Store array(s) in band(s)
        for target_array in range(1, nb_bands):
            outDs.GetRasterBand(target_array).WriteArray(array[target_array-1])
            outDs.GetRasterBand(target_array).SetNoDataValue(10000)

    else:
        outDs.GetRasterBand(1).WriteArray(array)

    # Hide nodata values
    outDs.FlushCache()

    # Informations to the user
    print("\n" + os.path.basename(outPath) + " OK")

    return None

def quickVisual(band, colors = "viridis"):
    # Third-libraries import
    from matplotlib import pyplot as plt

    # Configuration de l'affichage
    plt.imshow(band, cmap = colors)
    plt.show()
    plt.close()
