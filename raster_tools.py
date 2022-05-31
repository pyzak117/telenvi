#%% Standard librairies
import os

# Third-Party librairies
import numpy as np
import geopandas as gpd
from osgeo import gdal

# other pieces of telenvi module
from telenvi.geogrid import GeoGrid
from telenvi.geoim import GeoIm

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