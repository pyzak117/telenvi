# -*- coding: utf-8 -*-

"""
telenvi package
---------------
Version = 1.0
Nov. 2021
"""

# Standard librairies
import os
import re
import sys

# Third-Party librairies
import numpy as np
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *

def load_bands(fPath, 
               product = None, 
               reflectance = False,
               pattern = None,
               endKeyPos = 1000,
               cropZone = None):

    # Known-products infos
    product_data = {
        "S2A":{
            "pattern":"B[0-9]+_10M.JP2$",
            "pos":-8
            },

        "LS8":{
            "pattern":"B[0-9]+.TIF$",
            "pos":-4
            }
    }

    # 1. Inputs checking
    ####################

    # Classic mode
    if product != None:
        if product in product_data.keys():
            pattern = product_data[product]["pattern"]
            endKeyPos = product_data[product]["pos"]
        else:
            raise ValueError("error 1 : Unknown product {}".format(product))
    
    # Custom mode
    else:
        if pattern == None:
            raise ValueError("error 2 : undefined pattern")

        if type(pattern) != str:
            raise ValueError("error 3 : pattern must be str")
        
        if type(endKeyPos) != int:
            raise ValueError("error 5 : end key position must be integer")

    CROP = False
    if cropZone != None:
        if type(cropZone) != list:
            raise ValueError("error 6 : Coordinates of cropZone must be a list of 2 tuples")
            
        elif len(cropZone) != 2:
            raise ValueError("error 6.1 : Coordinates of cropZone must be a list of 2 tuples")
        
        else:
            if len(cropZone[0]) != 2 or len(cropZone[1]) != 2:
                raise ValueError("error 7 : cropZone tuples must have 2 integers")

        # Unpack crop coordinates
        xMin, yMax = cropZone[0]
        xMax, yMin = cropZone[1]

        # Check coordinates logic validity
        if xMin >= xMax or yMin >= yMax :
            raise ValueError("error 8 : Coordinates are invalid")

        # Crop mode activated
        CROP = True

    # Compile pattern with regular expression
    rpattern = re.compile(pattern)

    # 2. Loading data
    ################

    bands = {}
    geoData = False

    print("Bands research and loading")
    for fileName in os.listdir(fPath):
        try :
            # get pattern start position in filename
            startKeyPos = re.search(rpattern, fileName.upper()).span()[0]
        
        except AttributeError:
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
        print("{} loaded".format(bandId))

    if CROP:
        return [bands, nGeoTransform, projection]

    else:
        return [bands, geoTransform, projection]

def writeGeotiff(band, rows, cols, geoTransform, projection,outPath,driverName = "GTiff"):
    
    driver = gdal.GetDriverByName(driverName)
    outData = driver.Create(outPath, cols, rows, 1, gdal.GDT_Int32)
    outData.SetGeoTransform(geoTransform)
    outData.SetProjection(projection)   
    outData.GetRasterBand(1).WriteArray(band)
    outData.GetRasterBand(1).SetNoDataValue(10000)
    outData.FlushCache()
    return None
  
def NormaRaster(PathRepPrin, ProjVoulu, Res, FormatFile, algo,OptFormat=False):

    try:

        o = [os.path.join(PathRepPrin,e) for e in os.listdir(PathRepPrin) if e[-4:] == '.tif']
        optionPRJ = gdal.WarpOptions(dstSRS=ProjVoulu)
        for e in o:
            ds = gdal.Open(e)
            ds_reproj = gdal.Warp(e[:-4]+"_Reproj.tif", ds, options=optionPRJ)


        p = [os.path.join(PathRepPrin, e) for e in os.listdir(PathRepPrin) if e[-len('_Reproj.tif'):] == '_Reproj.tif']
        print(p[0][:-4])
        optionRes = gdal.WarpOptions(xRes=Res, yRes=Res, resampleAlg=algo)
        for e in p:
            ds = gdal.Open(e)
            ds_resample = gdal.Warp(e[:-4] + "_Resample.tif", ds, options=optionRes)

        if OptFormat == True:
            path = [os.path.join(PathRepPrin, i) for i in os.listdir(PathRepPrin) if i[-len('_Resample.tif'):] == '_Resample.tif']
            for e in path:
               Image.open(e).save('{}.{}'.format(e[:-4], str(FormatFile.lower())), format=FormatFile)

        else:
            pass

    except ValueError:
        print("CHEH")
        pass
    except NotADirectoryError:
        print("Le chemin d'accès vers l'image n'est pas conforme, \n par exemple sous Linux : '/home/user/Bureau/répertoire_avec_série_d'image'")
        pass
    except KeyError:
        print('Erreur dans les paramétres de la fonction')
        
        
def VtoR (PathShp, Output, Newkey, TypeImg = 'GTiff', CRS = 2154, Resolution = 0.5):
    """
    :param PathShp: Chemin d'accès absolue du fichier vecteur à convertir en raster (type = str)
    :param Output: Chemin d'accès d'enregistrement du fichier raster (type = str)
    :param Newkey: Nom du fichier qui doit créer (type = str)
    :param TypeImg: Format d'image en sorti, sert à choisir le driver pour construire le raster (type = str)
    :param CRS: Le système de projection voulu (type = int)
    :param Resolution: La résolution spatiale attendue du raster en sorti (type = float) 
    :return: 
    """
    try:
        # Chargement de l'objet vecteur
        Input_Shp = ogr.Open(PathShp)
        InputShp = Input_Shp.GetLayer()
        # Chargement de la resolution voulu, par défaut 50cm
        pixel_size = Resolution
        # Obtenir l'emprise spatiale de la couche vecteur
        x_min, x_max, y_min, y_max = InputShp.GetExtent()
        # Calcule de la dimension du raster
        x_res = int((x_max - x_min) / pixel_size)
        y_res = int((y_max - y_min) / pixel_size)
        # Chargement du driver pour créer le fichier raster
        driver = gdal.GetDriverByName(TypeImg)
        # Création du Raster, et paramétrage des attributs du raster
        new_raster = driver.Create(Output+Newkey, x_res, y_res, 1, gdal.GDT_Byte)
        new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))
        band = new_raster.GetRasterBand(1)
        # Gestion des Nodatas 
        no_data_value = -9999
        band.SetNoDataValue(no_data_value)
        band.FlushCache()
        # On 'remplit' le raster vide créer précédémment avec les données shp, qui cooresponde à une emprise spatiale
        gdal.RasterizeLayer(new_raster, [1], InputShp, burn_values=[255])
        # Gestion du système de projection du raster
        new_rasterSRS = osr.SpatialReference()
        new_rasterSRS.ImportFromEPSG(CRS)
        new_raster.SetProjection(new_rasterSRS.ExportToWkt())
        
    except ValueError:
        print('CHEH')
        sys.exit(1)
