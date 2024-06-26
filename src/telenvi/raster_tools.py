module_description = """
--- telenvi.raster_tools ---
Functions to process georeferenced rasters through the
osgeo.gdal package with a more friendly and intuitive 
syntax.
"""

# telenvi modules
from telenvi.associations import npdtype_gdalconst, extensions_drivers
import telenvi.geoim as geoim
import telenvi.vector_tools as vt 

# Standard libraries
import numbers
import os
import json
import pathlib
import warnings
from pathlib import Path

# CLI libraries
from tqdm import tqdm

# Data libraries
import numpy as np
import pandas as pd

# Geo libraries
import shapely
import rasterio
from rasterio.features import shapes
from shapely.errors import ShapelyDeprecationWarning

# import richdem as rd
import geopandas as gpd
from osgeo import gdal, gdalconst, osr, ogr

"""
# --------------
# GETTERS - Functions to extract metadata and geodata from raster files 
# --------------
"""

def getDs(target, mode=''):

    if mode == 'rasterio':
        return rasterio.open(target)

    if type(target) == str:
        if not os.path.exists(target):
            raise ValueError("the target path doesn't exist")
        ds = gdal.Open(target)
    
    elif type(target) == gdal.Dataset:
        ds = target
        if ds == None:
            raise ValueError("the target file is not a valid geo raster")

    elif type(target) == pathlib.PosixPath:
        ds = gdal.Open(str(target))

    elif target == None:
        return None

    else: # It can be a geoim
        ds = target.ds

    return ds

def getOrigin(target):
    """
    target : osgeo.gdal.Dataset or str
        if str, convert into osgeo.gdal.Dataset with gdal.Open()

    send a tuple (xMin,)
    """
    target = getDs(target)
    return (target.GetGeoTransform()[0], target.GetGeoTransform()[3])

def getPixelSize(target):
    """
    target : osgeo.gdal.Dataset or str
        if str, convert into osgeo.gdal.Dataset with gdal.Open()

    send a tuple (pixelSizeX, pixelSizeY)
    """

    target = getDs(target)
    return (target.GetGeoTransform()[1], target.GetGeoTransform()[5])

def getGeoBounds(target):
    """
    target : osgeo.gdal.Dataset or str
        if str, convert into osgeo.gdal.Dataset with gdal.Open()

    send a tuple (xMin, yMin, xMax, yMax)
    """

    target = getDs(target)
    nRows, nCols=target.RasterYSize, target.RasterXSize
    xMin, yMax=getOrigin(target)
    xRes, yRes=getPixelSize(target)
    xMax=xMin + xRes * nCols
    yMin=yMax + yRes * nRows
    return xMin, yMin, xMax, yMax

def getShape(target):
    """
    target : a osgeo.gdal.Dataset
             or numpy.ndarray representing a raster multi or mono spectral

    send a tuple (numBands, numRows, numCols)
    """
    if type(target) in [np.ndarray]:
        try :
            nBands, nRows, nCols = target.shape
        except ValueError:
            nBands, nRows, nCols = (1, target.shape[0], target.shape[1])

        return nBands, nRows, nCols
    
    else:
        target = getDs(target)
        nBands=target.RasterCount
        nRows=target.RasterYSize
        nCols=target.RasterXSize
        return nBands, nRows, nCols

def drawGeomExtent(target, geomType="ogr"):
    """
    compute a geometric square from georeferenced object

    - PARAMETERS -        
    target : str - a path to a raster file 
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset
             tuple - a spatial coordinates tuple (xMin, yMin, xMax, yMax
    
    geomType : str - describe the returned geometry type
               default : 'ogr'
               alternative : 'shly'

    - RETURNS -    
    geom : osgeo.ogr.geometry
           shapely.geometry.polygon
    """
    
    if type(target) != tuple:
        target = getDs(target)
        xMin, yMin, xMax, yMax = getGeoBounds(target)

    elif type(target) == tuple:
        xMin, yMin, xMax, yMax = target

    # Compute OGR geometry
    if geomType.lower() == "ogr":

        # Create a ring
        ring=ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xMin, yMax)
        ring.AddPoint(xMin, yMin)
        ring.AddPoint(xMax, yMin)
        ring.AddPoint(xMax, yMax)
        ring.AddPoint(xMin, yMax)

        # Assign this ring to a polygon
        polygon_env=ogr.Geometry(ogr.wkbPolygon)
        polygon_env.AddGeometry(ring)

    # Compute Shapely geometry
    elif geomType.lower() == "shly":
        polygon_env=shapely.geometry.Polygon(
            [(xMin, yMax),
             (xMin, yMin),
             (xMax, yMin),
             (xMax, yMax),
             (xMin, yMax)])

    else :
        print("geomType unknown")
        return None

    return polygon_env

def getJsonProj(target):
    target = getDs(target)
    sp_ref=osr.SpatialReference(wkt=target.GetProjection())
    return json.loads(sp_ref.ExportToPROJJSON())

def getWktFromEpsg(wktProj):
    srs=osr.SpatialReference()
    srs.ImportFromEPSG(wktProj)
    wktcrs=srs.ExportToWkt()
    return wktcrs

def getCrsEpsg(target):
    """
    for the moment, target must be a path and only a path
    send the target spatial coordinates reference system epsg id
    """
    target = getDs(target, mode='rasterio')
    return target.crs.to_epsg()

def getCrsWkt(target):
    """
    for the moment, target must be a path and only a path
    send the target spatial coordinates reference string
    """
    target = getDs(target, mode='rasterio')
    return target.crs.to_wkt()

def checkRasterValidity(target):
    """
    check the validity of a file regarding to a normal georeferenced file characteristics
    """
    pass

def spaceCoord_to_arrayCoord(point, ds):
    """
    send matrixian coordinates of a point

    - PARAMETERS - 
    point     : (x, y)
    ds        : str - a path to a raster file 
                osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset

    - RETURNS -
    a tuple (row, col)
    """
    ds = getDs(ds)

    # Unpack the point
    if type(point) in (tuple, list):
        pX, pY = point
    elif type(point) == shapely.geometry.point.Point:
        pX, pY = point.x, point.y

    # Get image origin point
    imOrX, imOrY = getOrigin(ds)

    # Get image resolution
    xRes, yRes = getPixelSize(ds)

    # Find the xRow
    row = abs(int((pY - imOrY)) / xRes)
    col = abs(int((pX - imOrX)) / yRes)

    return (int(row), int(col))
    
def spaceBox_to_arrayBox(geoBounds, ds, array = None):
    """
    send matrixian coordinates of a portion of the space

    - PARAMETERS - 
    geoBounds : tuple - (xMin, yMin, xMax, yMax)
    ds        : str - a path to a raster file 
                osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset
    array     : np.ndarray (facultative) - the input array

    - RETURNS -
    a tuple (row1, col1, row2, col2)

    """

    ds = getDs(ds)
    if type(array) not in [np.ndarray]:
        array = ds.ReadAsArray()

    # Get initial image resolution
    xRes, yRes=getPixelSize(ds)

    # Get initial image extent
    A=drawGeomExtent(ds, "shly")
    AxMin=A.bounds[0]
    AyMax=A.bounds[3]

    # Get argument extent
    B=drawGeomExtent(geoBounds, "shly")

    # Get intersection extent
    C=A.intersection(B)
    CxMin, CyMin, CxMax, CyMax=C.bounds

    # Transform geographic to matrixian coordinates
    # distance between the top edge of A and the top edge of C=CyMax - AyMax
    # to find the row of the input image integrating the C line
    # number_of_pixels_between_A_top_edge_and_C_top_edge=dist_between_A_top_edge_and_C_top_edge / yRes

    row1=int((CyMax - AyMax) / yRes)
    col1=int((CxMin - AxMin) / xRes)
    row2=int((CyMin - AyMax) / yRes)
    col2=int((CxMax - AxMin) / xRes)

    return row1, col1, row2, col2

def getDriverNameFromPath(path):
    if path == "":
        return "MEM"
    try :
        return extensions_drivers[path.lower().split(".")[-1]]
    except KeyError:
        print(f"no driver found for extension {os.path.basename(path).split('.')[-1]}. Default driver assignation : GeoTiff")
        return "GTiff"

def getCentroids(target):
    """
    get pixels centroids coordinates

    - PARAMETERS -        
    target : str - a path to a raster file 
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset
             tuple - a spatial coordinates tuple (xMin, yMin, xMax, yMax
    
    - RETURNS -    
    centroids : np.ndarray - pixels centroid coordinates
    """
    target = getDs(target)

    # get metadata
    orX, orY = getOrigin(target)
    pSizeX, pSizeY = getPixelSize(target)
    _, nRows, nCols = getShape(target)

    centroids = []
    for row in range(nRows):
        for col in range(nCols):
            x = orX + (pSizeX * col) + (0.5 * pSizeX)
            y = orY + (pSizeY * row) + (0.5 * pSizeY)
            centroids.append((x,y))
    return centroids

"""
# --------------
# SETTERS - Functions to change metadata and geodata from raster files
# --------------
"""

def cropFromRaster(target, model, resMethod = "near", outpath = "", verbose = False):

    """
    crop a target raster according to the model raster extent

    - PARAMETERS - 
    target : str - a path to a raster file
            osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset

    model : str - a path to a raster file
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset

    - RETURNS -
        an osgeo.gdal.Dataset
    """

    target = getDs(target)
    model  = getDs(model)

    # Extract geometries
    target_extent=drawGeomExtent(target, geomType="shly")
    model_extent=drawGeomExtent(model, geomType="shly")

    # Intersect themselves
    inter_extent=target_extent.intersection(model_extent)

    # Get data on the intersection area
    return cropFromVector(target, inter_extent, resMethod = resMethod, outpath = outpath, verbose=verbose)

def cropFromVector(target, vector, layername='', resMethod = "near", outpath="", featureNum=0, featureCondition={}, verbose=False):
    """
    cut the image according to a vector geometry

    - PARAMETERS - 
    target : str - a path to a raster file 
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset
 
    vector : str - a path to a shapefile
             shapely.Geometry.Polygon - a geometry

            describe the spatial extent on which the image 
            will be cropped. 

    polygon (facultative) : int
            if the vector argument is a path to a shapefile,
            this argument specify the id of the polygon inside
            this shapefile to use

    - RETURNS -
        an osgeo.gdal.Dataset
    """

    target = getDs(target)

    # If vector argument is a path to a shapefile,
    # here we extract only one polygon of this shapefile
    if type(vector) == str:

        # Open the file
        if vector.endswith('.shp'):
            layer = gpd.read_file(vector)
        elif vector.endswith('.gpkg'):
            layer = gpd.read_file(vector, layer=layername)

        # Extract the feature
        if featureCondition != {}:
            column_name = list(featureCondition.keys())[0]
            wanted_value = featureCondition[column_name]
            vector = layer[layer[column_name] == wanted_value].iloc[featureNum]

        else:
            vector=layer.iloc[featureNum]

        # Extract the geometry
        vector = vector.geometry

    # Extract Coordinates extent
    if type(vector) in (shapely.geometry.polygon.Polygon, shapely.geometry.multipolygon.MultiPolygon):    
        xMin, yMin, xMax, yMax=vector.bounds

    elif type(vector) in (tuple, list):
        xMin, yMin, xMax, yMax = vector

    # get driver name
    driverName = getDriverNameFromPath(outpath)

    # inform user
    if verbose: print(f"crop\n---\nxMin : {xMin}\nyMin : {yMin}\nxMax : {xMax}\nyMax : {yMax}\n---\n")

    # crop the dataset
    new_ds=gdal.Warp(
        destNameOrDestDS=outpath,
        srcDSOrSrcDSTab=target,
        format=driverName,
        outputBounds=(xMin, yMin, xMax, yMax),
        resampleAlg=resMethod
    )

    # update target resolution (sometimes, warp change it a little bit)
    if getPixelSize(new_ds) != getPixelSize(target):
        pxSize, pySize = getPixelSize(target)
        orX, orY = getOrigin(new_ds)
        new_ds.SetGeoTransform((orX, pxSize, 0.0, orY, 0.0, pySize))

    return new_ds

def resize(target, outpath ="", xRes = None, yRes = None, model = None, method = "near"):
    """
    change the pixel spatial size of a raster, sometimes
    (wrongly) called "spatial resolution"

    - PARAMETERS - 
    target : str - a path to a raster file 
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset

    xRes : float - the X pixel size
    yRes : float - the Y pixel size
           default : xRes

    model : str - a path to a raster file 
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset

    method : str - the resampling algorithm
             default : "near"
             alternatives :
                 "bilinear"   
                 "cubic"      
                 "cubicspline"
                 "lanczos"    
                 "average"    
                 "rms"        
                 "max"        
                 "min"        
                 "med"        
                 "q1"         
                 "q3"         
                 "sum"

    - RETURNS -
        an osgeo.gdal.Dataset
    """

    target = getDs(target)

    if xRes != None and yRes == None:
        yRes = xRes

    if model != None:
        xRes, yRes = getPixelSize(model)

    driverName = getDriverNameFromPath(outpath)

    ds_resized=gdal.Warp(
        destNameOrDestDS=outpath,
        srcDSOrSrcDSTab=target,
        format=driverName,
        xRes=xRes,
        yRes=yRes,
        resampleAlg=method)
    
    return ds_resized

def reproj(target, crs = None, outpath = ""):
    """
    change the spatial projection of a raster

    - PARAMETERS - 

    target : str - a path to a raster file 
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset

    crs : int - epsg of the target CRS
          str - WKT string representing the target CRS

    - RETURNS -
        an osgeo.gdal.Dataset
    
    """
    target = getDs(target)

    if type(crs) == int:
        crs = getWktFromEpsg(crs)

    driverName = getDriverNameFromPath(outpath)

    return gdal.Warp("", target, format=driverName, dstSRS=crs)

def pickBands(target, bands, outpath=""):
    """
    extract one or many bands from a multispectral rasterfile

    - PARAMETERS - 
    target : str - a path to a raster file 
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset

    bands : int  - an index of a band to extract
            list - a list of bands indexes

    - RETURNS -
        an osgeo.gdal.Dataset
    """

    target = getDs(target)

    if type(bands) == int:
        bands=[bands]

    driverName = getDriverNameFromPath(outpath)

    new_ds = gdal.Translate(
        destName="",
        srcDS=target,
        format=driverName,
        bandList=bands,
        noData=0
    )

    return new_ds

def shiftOrigin(target, offsetX, offsetY, outpath=""):
    """
    shift the origin of a raster file

    - PARAMETERS - 
    target : str - a path to a raster file 
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset

    offsetX : float - 
        The distance to shift the image origin point
        (in general, north-west corner) along the X axe.

        This distance must be exprimate in the target CRS unit (meters or degrees) 

    offsetY : float - same as offsetX but for the Y axe

    - RETURNS -
        an osgeo.gdal.Dataset
    """

    target = getDs(target)

    # Get metadata
    xRes, yRes=getPixelSize(target)
    orX, orY=getOrigin(target)

    # shift origin
    target.SetGeoTransform((orX + offsetX, xRes, 0.0, orY + offsetY, 0.0, yRes))

    if outpath != "":
        write(target, outpath)

    return target

def stack(targets, outpath=""):
    """
    stack raster files togethers

    - PARAMETERS - 
    targets : list of str - paths to raster files
              list of osgeo.gdal.Dataset - raster files represented by gdal.Dataset

     - RETURNS -
        an osgeo.gdal.Dataset
    """
    targets = [getDs(t) for t in targets]
    new_ds = gdal.BuildVRT(outpath, targets, separate = True)
    return new_ds

def merge(targets, outpath=""):
    """
    merge raster files side by side

    - PARAMETERS - 
    targets : list of str - paths to raster files
              list of osgeo.gdal.Dataset - raster files represented by gdal.Dataset

     - RETURNS -
        an osgeo.gdal.Dataset
    """

    # Merge datasets
    new_ds = gdal.Warp(
        destNameOrDestDS='',
        srcDSOrSrcDSTab=[getDs(t) for t in targets],
        format='MEM')

    # Save the merged dataset in a raster file
    if outpath != "":
        write(new_ds, outpath)

    return new_ds

def assign_crs(target, epsg, outpath=""):
    """
    just assign a new crs to the target. 
    WARNING : It's NOT a reprojection. It's just a assignment.

    - PARAMETERS - 
    target : str - a path to a raster file 
             osgeo.gdal.Dataset - a raster file represented by a gdal.Dataset

     - RETURNS -
        an osgeo.gdal.Dataset
    """

    # Extract data from the target
    t = geoim.Geoim(getDs(target))
    orX, orY = t.getOrigin()
    psx, psy = t.getPixelSize()
    
    # Build a new dataset
    new_ds = create(
        t.array,
        outpath,
        orX,
        orY,
        psx,
        psy,
        epsg)

    return new_ds

def create(
    array,
    path = "",
    orX = 0.0,
    orY = 0.0,
    xPixSize = 1,
    yPixSize = 1,
    crs = 4326):

    """
    create osgeo.gdal.Dataset from an array and geographic informations (optionnals)

    - PARAMETERS - 
    array : numpy.ndarray - raster pixels values
    
    path  : str - default : "" - function return a gdal.Dataset store in memory
                  facultative : an outpath to write the raster somewhere
    
    orX   : float - raster x coordinate origin
    orY   : float - raster y coordinate origin
    
    xPixSize : float - the x length of the space portion represented by a pixel
    yPixSize : float - the y length of the space portion represented by a pixel

    crs : int - epsg of the target CRS
          str - WKT string representing the target CRS

    - RETURNS -
        an osgeo.gdal.Dataset
    """

    # Get raster driver name
    driverName = getDriverNameFromPath(path)

    # Get raster encoding from the array
    try:
        ds_enc = npdtype_gdalconst[array.dtype.name]
    except KeyError:
        print("@TELENVI INFOS : WARNING : \nno gdalconst encoding found for this array - default assignation : gdalconst.GDT_Float64 - the heaviest")
        ds_enc = gdalconst.GDT_Float64

    # Get raster size from the array
    nBands, nRows, nCols = getShape(array)

    # Create raster driver
    newds_driver = gdal.GetDriverByName(driverName)
    newds = newds_driver.Create(path, nCols, nRows, nBands, ds_enc)

    # Dataset geodata setup
    newds.SetGeoTransform((orX, xPixSize, 0.0, orY, 0.0, yPixSize))
    if type(crs) == int:
        crs = getWktFromEpsg(crs)
    newds.SetProjection(crs)

    # Load data into the dataset
    if nBands > 1:
        for band in range(1, nBands+1):
            newds.GetRasterBand(band).WriteArray(array[band-1])
    else:
        newds.GetRasterBand(1).WriteArray(array)

    return newds

def write(target, outpath, ndValue=None):
    """
    create osgeo.gdal.Dataset from an array and geographic informations (optionnals)

    - PARAMETERS -
    target : osgeo.gdal.Dataset - a raster represented by a gdal.Dataset
    outpath  : str - default : "" - function return a gdal.Dataset store in memory
                  facultative : an outpath to write the raster somewhere
    """
    
    # Create driver
    driverName = getDriverNameFromPath(outpath)
    driver = gdal.GetDriverByName(driverName)

    # Assign noData value if required
    if ndValue!= None:
        for band in range(1, getShape(target)[0]+1):
            target.GetRasterBand(band).SetNoDataValue(ndValue)

    # Create target copy
    driver.CreateCopy(outpath, target, 1)

    if gdal.Open(outpath) != None:
        print(f"{os.path.basename(outpath)} OK")
        return True
    else:
        print(f"error during {os.path.basename(outpath)} creation")
        return False    

def vectorize(target, mode='points'):

    # Read the target
    target = geoim.Geoim(target)

    if mode == 'points':

        # extract raster metadata
        x_origin, y_origin = getOrigin(target)
        x_pixel_size, y_pixel_size = getPixelSize(target)
        nBands, nCols, nRows = getShape(target)

        # Adjust the origins to put the vector point on the center of their corresponding pixels
        x_grid_origin = x_origin + (x_pixel_size - (x_pixel_size/2))
        y_grid_origin = y_origin + (y_pixel_size - (y_pixel_size/2)) 

        # Create coordinate arrays
        x_coords = np.arange(
            start= x_grid_origin,
            stop = x_grid_origin + x_pixel_size * nRows,
            step = x_pixel_size)

        y_coords = np.arange(
            start= y_grid_origin,
            stop = y_grid_origin + y_pixel_size * nCols,
            step = y_pixel_size)

        # A 3 dimensionals arrays. We use it like coords[:, posY, posX]
        coords = np.array(np.meshgrid(x_coords, y_coords))

        # Make a list with each 2D arrays of the images - if multispectral
        if nBands > 1:
            values = [target.array[band] for band in range(0,nBands)]
        else:
            values = [target.array]

        # Build a 3D array with X and Y coordinates of each pixel + their values
        combo = np.array(values + [coords[1], coords[0]])

        # Reverse it. Now, combo[posX][poxY] return at least 3 values : b1, b2... bn, coordx, coordy
        combo = combo.T
        
        # Build a list of DataFrames, one for each row because pd.DataFrame(array) must be used with 2D array
        protoVectorLayer = []
        for row in tqdm(range(nRows)):
            rowDf = pd.DataFrame(combo[row], columns = [f"b{nBand}" for nBand in range(nBands)] + ['cy','cx'])

            # Add geometry column with shapely
            with warnings.catch_warnings(): # Avoid inunderstandable shapely depreciation warning
                warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
                rowDf['geometry'] = rowDf.apply(lambda row: shapely.geometry.Point(row.cx, row.cy), axis=1)

            # Drop useless columns
            rowDf = rowDf.drop(['cx', 'cy'], axis=1)

            protoVectorLayer.append(rowDf)

        # Concat all the row DataFrames into one
        vectorLayer = gpd.GeoDataFrame(pd.concat(protoVectorLayer, ignore_index=True))

    return vectorLayer

    """
    elif mode == 'polygons':
        
        target = getRasterioDs(target)
        with rasterio.Env():
            with rasterio.open(str(Path(self.session.p_raster_data, self.pz_name, 'displacements', f"{self.pz_name}_moving-areas_{n_clusters}_{mode}.tif"))) as src:
                image = src.read(1) # first band
                results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    shapes(image, mask=mask, transform=src.transform)))
        geoms = list(results)
        gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms).set_crs(epsg=2154)
        """

def cropFromIndexes(target, indexes):

    # Unpack indexes
    firstCol, firstRow, lastCol, lastRow = indexes

    # Get target metadata
    target = getDs(target)
    crs = target.GetProjection()
    xRes, yRes = getPixelSize(target)
    orX, orY = getOrigin(target)
    _, nRows, nCols = getShape(target)

    # Compute the spatial origin of the area to extract
    new_orX = orX + (firstCol * xRes)
    new_orY = orY + (firstRow * yRes)

    # Compute X size and Y size in pixels
    xSize = lastCol - firstCol
    ySize = lastRow - firstRow

    # Make extraction
    extraction = target.ReadAsArray(firstCol, firstRow, xSize, ySize)

    # Create a new dataset from the new geodata and the new array
    newDs = create(
        extraction,
        "",
        new_orX,
        new_orY,
        xRes,
        yRes,
        crs)
    
    return newDs, extraction

def Open(
    target,
    nRes       = None,
    epsg       = None,
    clip       = None,
    nBands     = None,
    featureNum = 0,
    verbose    = False,
    geoExtent  = None,
    layername  = '',
    featureCondition  = {},
    arrExtent  = None,
    load_pixels  = False,
    resMethod  = "near"):

    """
    Function to load metadata and data, after applying 
    all kind of geo-pre-processings on a georaster

    - PARAMETERS -
        target (mandatory) : str - the path to the raster file to open
        geoim : boolean - if True, the function return a telenvi.geoim object. Else, a gdal.Dataset
        nBands : int or list - the band number to load, if target is a multispectral raster
        geoExtent - describe a spatial extent to crop the target
            str : a path to a shapefile or to a rasterfile
            tuple : (xMin, yMin, xMax, yMax)
        arrExtent : tuple - describe a matrixian portion to crop the target
            (firstCol, firstRow, lastCol, lastRow)
        nRes : int - the new resolution or pixel size to give to the target by resampling
        epsg : int - the epsg of a Coordinates System Reference. Target is reprojected in it.
        clip : str - A path to an other raster file or a geoim object.
                This case is different than just "crop" according to another
                raster file because here, the resolution, the xOrigin and yOrigin and the spatial 
                projection are setup according to this other raster file.
        nFeature : int - the id of the shapefile feature to use for the geo crop
        resMethod : str - the resampling ethod to use. default : "near"
        verbose : boolean - write informations during the pre-processings

    - RETURNS -
        if load_pixels : a telenvi.geoim instance
        if not load_pixels : a gdal.Dataset instance
    """

    if verbose: 
        print("\n")
    
    # Get target
    inDs = getDs(target)

    # Clip check
    if clip != None:
        model_ds = getDs(clip)

        # Get target (input) and clip (model) pixel size
        input_resX, _  = getPixelSize(inDs)
        model_resX, _ = getPixelSize(model_ds)

        # If the input pixel size is higher than the model pixel size,
        # we order a resample by setting res argument
        if model_resX < input_resX:
            nRes = model_resX

        # SCR check
        if inDs.GetProjection() != model_ds.GetProjection():
            epsg = model_ds.GetProjection()

        # Crop check
        geoExtent = model_ds

    # Reprojection
    if epsg != None:
        if verbose: print(f"reprojection\n---\nin  : {getCrsEpsg(target)}\nout : {epsg}\n---\n")
        inDs=reproj(inDs, epsg)

    def _geoCrop(inDs, geoExtent, featureNum, verbose, featureCondition=featureCondition):
        if type(geoExtent) == str:
            if geoExtent[-4:].lower() == ".shp":
                inDs=cropFromVector(inDs, geoExtent, featureNum=featureNum, verbose=verbose, featureCondition=featureCondition)
            elif geoExtent.endswith('.gpkg'):
                inDs=cropFromVector(inDs, geoExtent, layername, featureNum=featureNum, verbose=verbose, featureCondition=featureCondition)
            elif geoExtent == '':
                pass
            else :
                try:
                    geoExtent = getDs(geoExtent)
                except ValueError:
                    print("the raster to use for the crop is not valid")

        if type(geoExtent) == gdal.Dataset:
            inDs = cropFromRaster(inDs, geoExtent, verbose=verbose)

        if type(geoExtent) in [list, tuple]: # xMin, yMin, xMax, yMax
            if verbose: print(f"crop\n---\nxMin : {geoExtent[0]}\nyMin : {geoExtent[1]}\nxMax : {geoExtent[2]}\nyMax : {geoExtent[3]}\n---\n")
            inDs=cropFromVector(inDs, geoExtent)

        if type(geoExtent) in [shapely.geometry.multipolygon.MultiPolygon, shapely.geometry.polygon.Polygon]:
            inDs=cropFromVector(inDs, geoExtent)

        return inDs

    def _extractBands(inDs, nBands):
        if nBands != None:
            if verbose: print(f"extract bands\n---\n{nBands}\n---\n")
            inDs = pickBands(inDs, nBands)
        return inDs

    def _resample(inDs, nRes, resMethod):
        if nRes != None:
            if verbose: print(f"resample\n---\nin     : {getPixelSize(inDs)[0]}\nout    : {nRes}\nmethod : {resMethod}\n---\n")
            inDs=resize(inDs, xRes=nRes, yRes=nRes, method=resMethod)
        return inDs

    # Clip is effective 
    if clip != None:
        inDs = _extractBands(inDs, nBands)
        inDs = _resample(inDs, nRes, resMethod)
        inDs = _geoCrop(inDs, geoExtent, featureNum, verbose)

    # Clip is not effective, so it don't matter if the images not have exactly the same number of rows / columns
    else :
        inDs = _geoCrop(inDs, geoExtent, featureNum, verbose)
        inDs = _extractBands(inDs, nBands)
        inDs = _resample(inDs, nRes, resMethod)

    # Matrixian crop
    if arrExtent != None:
        inDs, inArray = cropFromIndexes(inDs, arrExtent)
        if load_pixels:
            return geoim.Geoim(inDs, inArray)

    # Returns
    if load_pixels:
        return geoim.Geoim(inDs)
    else:
        return inDs

def getRasterioDs(target):
    """
    Send a rasterio dataset from a path or a gdal dataset
    """
    pass

def getSlope(dem):
    """
    Compute slope in degrees from a dem 
    """

    # Load the data
    if not type(dem) == geoim.Geoim:
        dem = Open(dem, load_pixels=True)

    # Convert the GeoIm numpy.ndarray to richdem.rdarray (pre-processing before slope computing)
    dem_rdarray = rd.rdarray(dem.array, no_data=-9999)
    dem_rdarray.geotransform = dem.ds.GetGeoTransform()
    dem_rdarray.projection = dem.ds.GetProjection()

    # Compute slope and conert in degrees unit
    slope_deg = np.array(np.degrees(np.arctan(rd.TerrainAttribute(dem_rdarray, attrib = "slope_riserun"))))

    # Create a geoim with this array and the same dataset than the initial dem
    slope = geoim.Geoim(dem.ds, slope_deg)

    return slope

def getAspect(dem):
    """
    Compute aspect in degrees from a dem 
    """

    # Load the data
    if not type(dem) == geoim.Geoim:
        dem = Open(dem, load_pixels=True)

    # Convert the GeoIm numpy.ndarray to richdem.rdarray (pre-processing before aspect computing)
    dem_rdarray = rd.rdarray(dem.array, no_data=-9999)
    dem_rdarray.geotransform = dem.ds.GetGeoTransform()
    dem_rdarray.projection = dem.ds.GetProjection()

    # Compute aspect and conert in degrees unit
    aspect_deg = np.array(rd.TerrainAttribute(dem_rdarray, attrib = "aspect"))

    # Create a geoim with this array and the same dataset than the initial dem
    aspect = geoim.Geoim(dem.ds, aspect_deg)

    return aspect

def getRastermap(target_dir_path, epsg=4326, extensions=['tif', 'jp2', 'hgt']):
    """
    Map the extents of the rasters contained in a directory
    """

    target_dir_path = Path(target_dir_path)

    # Track the rasters compatible with each extensions 
    targets_paths = []
    for xt in extensions:
        xt = xt.removeprefix('.')
        targets_paths += list(target_dir_path.glob(f'*{xt}'))

    # Write their filepaths in a geodataframe
    geo_extents = pd.DataFrame([{'filepath':str(path)} for path in targets_paths])

    # Map their extents
    geo_extents['geometry'] = geo_extents.apply(lambda row: drawGeomExtent(row.filepath, 'shly'), axis=1)

    # Convert into GeoDataFrame
    geo_extents = gpd.GeoDataFrame(geo_extents).set_crs(epsg)

    # geo_extents = gpd.GeoDataFrame([{'geometry':drawGeomExtent(track, 'shly'), 'dataset':str(track)} for track in targets]).set_crs(epsg=epsg)
    return geo_extents

def OpenFromMultipleTargets(
    target_source : str | gpd.GeoDataFrame,
    layername : str = '',
    area_of_interest : shapely.Polygon = None,
    load_pixels : bool = False,
    nRes : numbers.Real = None,
    ):

    """
    Make Geoprocessings to open one raster from a directory containing many, only on a selected area    
    """

    # Open the metadata of the targets as vector files
    if type(target_source) == gpd.GeoDataFrame or (type(target_source) == str and target_source.endswith(('.gpkg', '.shp'))):
        tracks_metamap = vt.Open(target_source)
    else:
        tracks_metamap = getRastermap(target_source)

    # Here we find the tracks intersecting the rock glacier outlines
    tracks_metadata = tracks_metamap[tracks_metamap.intersects(area_of_interest)==True]

    # Crope each dems intersecting the rock glacier
    tracks_ds = [Open(track.filepath, load_pixels=False, geoExtent=area_of_interest) for track in tracks_metadata.iloc]

    # Merge
    tracks_merged = merge(tracks_ds)

    # return tracks_merged

    # Resampling if needed
    if nRes is not None:
        tracks_merged = resize(tracks_merged, xRes=nRes)
    
    # Loading
    return Open(tracks_merged, load_pixels=load_pixels)

#TODO if __name__ == "__main__":
# 
#     # There is a lot of work to do here
#     # To use this toolbox from command line