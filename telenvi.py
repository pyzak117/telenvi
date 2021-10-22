# -*- coding: utf-8 -*-

"""
telenvi package
---------------
Version = 0.0.1
Oct. 2021
"""

# Standard librairies
import os
import re

# Third-Party librairies
import numpy as np
import gdal
from matplotlib import pyplot as plt

class Imsat:

    """
    description
    -----------
    Represent a satellite image
    Each band is automatically convert in numpy.ndarray

    attributes
    ----------
        blue(ndarray) : blue band
        red(ndarray) : red band
        ...

    mandatory parameters
    --------------------
        imgPath(str): bands folder location

    optionnals parameters
    ---------------------
        cropZone(list) : a list of tuples which represent top-left and bottom-right points of a region of interest
        product(str) : specify the image platform (his sensor source)
                       default value : "LS8"
                       enables values : {"LS8", "S2A"}
    """

    def __init__(self, imgPath, cropZone = None, product = "LS8"):

        CROP = False

        if cropZone != None:
            if len(cropZone) != 2:
                raise ValueError("unable to find cropZone - invalid coordinates")

            # Unpack crop coordinates
            xMin, yMax = cropZone[0]
            xMax, yMin = cropZone[1]

            # Check coordinates logic validity
            if xMin >= xMax or yMin >= yMax :
                raise ValueError("Coordonnées de la zone d'intérêt invalides")

            # crop mode activated
            CROP = True

        print("Bands research and loading")
        bands = {}
        for file in os.listdir(imgPath):
            
            if file.upper().endswith(".TIF"):

                # Building pattern with regular expression
                motif = re.compile("B[0-9]+.TIF$") # [0-9]+ = 1 or many numbers
                                                   # $ = string end

                try :
                    # pattern position in filename
                    pos = re.search(motif, file.upper()).span()[0]
                
                except AttributeError:
                    continue # pattern not in filename : switch to next file

                bandId = file[pos:-4] # "imageNameB12.tif"[pos:-4] = "B12"
                fileBandName = os.path.join(imgPath, file)
                ds = gdal.Open(fileBandName)

                if CROP:
                    # Get geographic data from the dataset
                    geoTransform = ds.GetGeoTransform()

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

                    # ndarray conversion
                    band = ds.ReadAsArray(col1,row1,col2-col1+1,row2-row1+1).astype(np.float32)

                else:
                    band = ds.ReadAsArray().astype(np.float32)

                # stock array in dic bands with bandId as key
                bands[bandId] = band

                print("{} loaded".format(bandId))

        if product == "LS8":
            self.blue = bands["B2"]
            self.green = bands["B3"]
            self.red = bands["B4"]
            self.nir = bands["B5"]
            self.swir1 = bands["B6"]
            self.swir2 = bands["B7"]
            self.panchro = bands["B8"]
            self.thermal1 = bands["B10"]

def computeNdvi(imsat):
    return ((imsat.nir - imsat.red) / (imsat.nir + imsat.red))

def show(band):
    plt.imshow(band)
    plt.show()
