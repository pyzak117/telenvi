# Third-Party librairies
import numpy as np
from osgeo import gdal, gdalconst, osr, ogr
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
import geopandas as gpd
import shapely
import re

# Standard librairies
import os

def openManyGeoRaster(
    directory,
    pattern,
    endKeyPos,
    crop = None,
    pol = 0,
    clip = None,
    numBand = None,
    epsg = None,
    res = None,
    resMethod = "near",
    ar_encoding = np.float32,
    ds_encoding = gdalconst.GDT_Float32
    ):

    # Compile pattern with regular expression
    rpattern = re.compile(pattern.upper())

    x = {}
    for fileName in sorted(os.listdir(directory)):

        try : # Get pattern start position in fileName
            startKeyPos = re.search(rpattern, fileName.upper()).span()[0]

        except AttributeError: # Pattern not find in fileName
            continue
        
        fileBandName = os.path.join(directory, fileName)
        
        # Get the key corresponding to the pattern in the fileName
        bandId = fileName[startKeyPos:endKeyPos]

        # Extract and pack all the data in a lovely dictionnary with bandId as key
        x[bandId] = openGeoRaster(
            rasterPath = fileBandName,
            crop = crop,
            pol = pol,
            clip = clip,
            epsg = epsg,
            res = res,
            resMethod = resMethod,
            ar_encoding = ar_encoding,
            ds_encoding = ds_encoding)

    return x

def openGeoRaster(
    rasterPath,
    crop = None,
    pol = 0,
    clip = None,
    numBand = None,
    epsg = None,
    res = None,
    resMethod = "near",
    ar_encoding = np.float32,
    ds_encoding = gdalconst.GDT_Float32,
    verbose=True
    ):

    # check target path validity
    if not os.path.exists(rasterPath):
        raise ValueError("error 1 : the path doesn't exist")
    
    # Input dataset
    inDs = gdal.Open(rasterPath)
    if inDs == None:
        raise ValueError("error 2 : the file is not a valid raster")

    if clip != None:

        # Open the clip as gdal dataset
        master_ds = gdal.Open(clip)

        # Get input and clip resolutions
        master_resX,_ = getPixelSize(master_ds)
        input_resX,_ = getPixelSize(inDs)

        # If they're different, we order a resample by setting res argument
        if master_resX != input_resX:
            res = master_resX

        # Order the reproject the input image into the SCR of the master image
        epsg = master_ds.GetProjection()

        # Then we set the crop argument to crop the input image on the extent of clip image
        crop = clip

    # Crop
    if type(crop) == str:
        if crop[-4:].lower() == ".shp":
            inDs = cropDsFromVectorFileOrShapelyGeom(inDs, crop, polygon=pol)
        elif gdal.Open(crop) != None:
            inDs = cropDsFromAntoher(inDs, crop)
    
    elif type(crop) in [list, tuple]:
        # xMin, yMin, xMax, yMax
        inDs = cropDsFromIndex(inDs, crop)

    # Reprojection
    if epsg != None:
        inDs = reprojDs(inDs, epsg)

    # Resample
    if res != None:
        inDs = resizeDs(inDs, xRes=res, yRes=res, resMethod=resMethod)
    
    # Extract interest band
    if numBand != None:
        inDs = chooseBandFromDs(inDs, numBand, ar_encoding, ds_encoding)

    # Switch
    if clip != None:
        in_orX, in_orY = getOriginPoint(inDs) # After the resampling and the crop
        ma_orX, ma_orY = getOriginPoint(master_ds)
        gapX = in_orX - ma_orX
        gapY = in_orY - ma_orY
        inDs = setOriginPoint(inDs, gapX, gapY, ds_encoding, ar_encoding)

    geoim = GeoIm(inDs, ds_encoding, ar_encoding)
    
    if verbose: print(f"{os.path.basename(rasterPath)} loaded")
    return geoim

def makeDs(
    path,
    array,
    orX,
    xRes,
    orY,
    yRes,
    crs,
    driverName = "MEM",
    ds_encoding = gdalconst.GDT_Float32):

    nBands, nRows, nCols = getBandsRowsColsFromArray(array)

    # Dataset metadata
    newds_driver = gdal.GetDriverByName(driverName)
    newds = newds_driver.Create(path, nCols, nRows, nBands, ds_encoding)

    # Dataset geodata setup
    newds.SetGeoTransform((orX, xRes, 0.0, orY, 0.0, yRes))
    newds.SetProjection(crs)

    # Load data into the dataset
    if nBands > 1:
        for band in range(1, nBands+1):
            newds.GetRasterBand(band).WriteArray(array[band-1])
    else:
        newds.GetRasterBand(1).WriteArray(array)

    return newds

def getOriginPoint(ds):
    return (ds.GetGeoTransform()[0], ds.GetGeoTransform()[3])

def getPixelSize(ds):
    return (ds.GetGeoTransform()[1], ds.GetGeoTransform()[5])

def getCoordsExtent(ds):
    nRows, nCols = ds.RasterYSize, ds.RasterXSize
    xMin, yMax = getOriginPoint(ds)
    xRes, yRes = getPixelSize(ds)
    xMax = xMin + xRes * nCols
    yMin = yMax + yRes * nRows
    return xMin, yMin, xMax, yMax

def getGeomExtent(ds = None, coords = None, mode="OGR"):

    # Extract instance extent coordinates
    if ds != None:
        xMin, yMin, xMax, yMax = getCoordsExtent(ds)
    
    if coords != None:
        xMin, yMin, xMax, yMax = coords

    # Compute OGR geometry
    if mode.upper() == "OGR":

        # Create a ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xMin, yMax)
        ring.AddPoint(xMin, yMin)
        ring.AddPoint(xMax, yMin)
        ring.AddPoint(xMax, yMax)
        ring.AddPoint(xMin, yMax)

        # Assign this ring to a polygon
        polygon_env = ogr.Geometry(ogr.wkbPolygon)
        polygon_env.AddGeometry(ring)

    # Compute Shapely geometry
    elif mode.upper() == "SHAPELY":
        polygon_env = shapely.geometry.Polygon(
            [(xMin, yMax),
             (xMin, yMin),
             (xMax, yMin),
             (xMax, yMax),
             (xMin, yMax)])

    return polygon_env

def getBandsRowsColsFromDs(ds):
    nBands = 0
    while ds.GetRasterBand(nBands+1) != None:
        nBands += 1
    nRows = ds.RasterYSize
    nCols = ds.RasterXSize
    return nBands, nRows, nCols

def getBandsRowsColsFromArray(array):
    ar_dims = len(array.shape)
    if ar_dims == 2:
        nBands, nRows, nCols = (1,) + array.shape
    elif ar_dims == 3:
        nBands, nRows, nCols = array.shape
    else:
        print("Array have more than 3 dimensions")
        return None
    return nBands, nRows, nCols

def getDsArrayIndexesFromSpatialExtent(ds, BxMin, ByMin, BxMax, ByMax):

    # Get initial image resolution
    xRes, yRes = getPixelSize(ds)

    # Get initial image extent
    A = getGeomExtent(ds = ds, mode = "SHAPELY")
    AxMin = A.bounds[0]
    AyMax = A.bounds[3]

    # Get argument extent
    B = getGeomExtent(coords = (BxMin, ByMin, BxMax, ByMax), mode = "SHAPELY")

    # Get intersection extent
    C = A.intersection(B)
    CxMin, CyMin, CxMax, CyMax = C.bounds

    # Transform geographic to matrixian coordinates
    # distance between the top edge of A and the top edge of C = CyMax - AyMax
    # to find the row of the input image integrating the C line
    # number_of_pixels_between_A_top_edge_and_C_top_edge = dist_between_A_top_edge_and_C_top_edge / yRes

    row1 = int((CyMax - AyMax) / yRes)
    col1 = int((CxMin - AxMin) / xRes)
    row2 = int((CyMin - AyMax) / yRes)
    col2 = int((CxMax - AxMin) / xRes)

    return row1, col1, row2, col2

def cropDsFromVectorFileOrShapelyGeom(ds, vector, ds_encoding = gdalconst.GDT_Float32, ar_encoding = np.float32, polygon=0):

    # If vector argument is a path to a shapefile,
    # here we extract only one polygon of this shapefile
    if type(vector) == str:
        layer = gpd.read_file(vector)
        vector = layer["geometry"][polygon]

    # Extract Coordinates extent
    xMin, yMin, xMax, yMax = vector.bounds
    row1, col1, row2, col2 = getDsArrayIndexesFromSpatialExtent(ds, xMin, yMin, xMax, yMax)

    # Crop the array
    custom_array = ds.ReadAsArray(col1, row1, col2-col1, row2-row1).astype(ar_encoding)
    custom_orX = xMin
    custom_orY = yMax
    nBands, nRows, nCols = getBandsRowsColsFromArray(custom_array)
    xRes, yRes = getPixelSize(ds)
    crs = ds.GetProjection()

    # Create a new dataset with the array cropped
    return makeDs("", custom_array, custom_orX, xRes, custom_orY, yRes, crs, "MEM", ds_encoding)

def cropDsFromAntoher(slave_ds, master_ds):
    if type(slave_ds) == str:
        slave_ds = gdal.Open(slave_ds)
    if type(master_ds) == str:
        master_ds = gdal.Open(master_ds)

    # Extract geometries
    slave_extent = getGeomExtent(slave_ds, mode="shapely")
    master_extent = getGeomExtent(master_ds, mode="shapely")

    # Intersect themselves
    inter_extent = slave_extent.intersection(master_extent)

    # Get data on the intersection area
    return cropDsFromVectorFileOrShapelyGeom(slave_ds, inter_extent)

def cropDsFromIndex(ds, index, ds_encoding = gdalconst.GDT_Float32, ar_encoding = np.float32):
    col1, row1, col2, row2 = index
    ls_stack = []
    num_band = 0
    while ds.GetRasterBand(num_band+1) != None:
        num_band += 1
    if num_band > 1:
        for band in range(0,num_band):
            ar_band = ds.GetRasterBand(num_band)
            ar_custom = ar_band.ReadAsArray(col1, row1, col2-col1, row2-row1).astype(ar_encoding)
            ls_stack.append(ar_custom)
        im_array = np.array(ls_stack)
    else:
        im_array = ds.GetRasterBand(1).ReadAsArray(col1, row1, col2-col1, row2-row1).astype(ar_encoding)

    # Get input dataset metadata
    xRes, yRes = getPixelSize(ds)
    old_orX, old_orY = getOriginPoint(ds)

    # Compute new origin point
    new_orX = old_orX + (col1 * xRes)
    new_orY = old_orY + (row1 * yRes)

    # Update the dataset's instance
    newDs = makeDs(
        "",
        im_array,
        new_orX,
        xRes,
        new_orY,
        yRes,
        ds.GetProjection(),
        "MEM",
        ds_encoding)

    return newDs

def resizeDs(ds, xRes, yRes, resMethod="near"):

    ds_resized = gdal.Warp(
        destNameOrDestDS="",
        srcDSOrSrcDSTab = ds,
        format = "VRT",
        xRes = xRes,
        yRes = yRes,
        resampleAlg = resMethod)
    
    return ds_resized

def reprojDs(ds, epsg):
    if type(epsg) == int:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        crs = srs.ExportToWkt()
    else:
        crs = epsg
    return gdal.Warp("", ds, format = "VRT", dstSRS = crs)

def chooseBandFromDs(ds, index, ar_encoding=np.float32, ds_encoding=gdalconst.GDT_Float32):
    orX, orY = getOriginPoint(ds)
    nBands, nRows, nCols = getBandsRowsColsFromDs(ds)
    xRes, yRes = getPixelSize(ds)
    crs = ds.GetProjection()
    array = ds.GetRasterBand(index).ReadAsArray(ar_encoding)
    return makeDs("", array, orX, xRes, orY, yRes, crs, "MEM", ds_encoding)

def setOriginPoint(ds, offsetX, offsetY, ds_encoding, ar_encoding):

    # Get metadata
    xRes, yRes = getPixelSize(ds)
    orX, orY = getOriginPoint(ds)

    # shift
    ds.SetGeoTransform((orX + offsetX, xRes, 0.0, orY + offsetY, 0.0, yRes))

    return ds

def stackGeoIms(geoims):
    return geoims[0].stack(geoims[1:], inplace=False)

def stackMonoDs(ls_ds, ar_encoding=np.float32):
    stack_ds = ls_ds[0]
    i = 0
    for ds in ls_ds[1:]:
        i+=1
        print(f"{i}/{len(ls_ds)}")
        stack_ds.GetRasterBand(1).WriteArray(ds.ReadAsArray().astype(ar_encoding))
    return stack_ds

class GeoIm:

    """
    """

    def __init__(self, GDALdataset, ds_encoding = gdalconst.GDT_Float32, array_encoding = np.float32):
        self.array_encoding = array_encoding
        self.ds_encoding = ds_encoding
        self.ds = GDALdataset
        self.array = self.ds.ReadAsArray().astype(array_encoding)

    def _updateArray(self):
        """
        Update instance's array from his dataset
        """
        self.array = self.ds.ReadAsArray().astype(self.array_encoding)

    def _updateDs(self, geodata = None):
        """
        Update instance's dataset from his array
        """

        # Get geographic informations
        if geodata == None:
            xRes, yRes = self.getPixelSize()
            orX, orY = self.getOriginPoint()
            crs = self.ds.GetProjection()
        else:
            orX, xRes, orY, yRes, crs = geodata

        # Get array dimensions
        nBands, nRows, nCols = self.getArraySize()

        # Make a new memory dataset
        newds = makeDs(
            "",
            self.array,
            orX,
            xRes,
            orY,
            yRes,
            crs,
            "MEM",
            self.ds_encoding)

        # Update instance dataset
        self.ds = newds

    def __repr__(self):
        self.quickVisual()
        return ""

    def __getitem__(self, index):
        print(type(index))
        return self.array[index]

    def copy(self):
        return GeoIm(self.ds)

    def getArraySize(self):
        return getBandsRowsColsFromArray(self.array)

    def getOriginPoint(self):
        return getOriginPoint(self.ds)

    def getPixelSize(self):
        return getPixelSize(self.ds)

    def getGeomExtent(self, mode="OGR"):
        return getGeomExtent(self.ds)

    def getCoordsExtent(self):
        return getCoordsExtent(self.ds)

    def setOriginPoint(self, offsetX, offsetY, inplace=True):
        """
        Move the raster by offsetX coordinates system reference unity, same for offsetY
        """
        shiftedDs = setOriginPoint(self.ds, offsetX, offsetY, self.ds_encoding, self.array_encoding)
        if inplace:
            self.ds = shiftedDs
            self._updateArray()
        else:    
            return GeoIm(shiftedDs)

    def cropFromVectorFileOrShapelyGeom(self, vector, polygon=0, inplace=True):
        crop_ds = cropDsFromVectorFileOrShapelyGeom(self.ds, vector, ar_encoding=self.array_encoding, ds_encoding=self.ds_encoding, polygon=polygon)
        if inplace:
            self.ds = crop_ds
            self._updateArray()
        else:    
            return GeoIm(crop_ds)
        
    def cropFromDataset(self, master_ds, inplace=True):
        crop_ds = cropDsFromAntoher(self.ds, master_ds)
        if inplace:
            self.ds = crop_ds
            self._updateArray()
        else:
            return GeoIm(crop_ds)        

    def cropFromIndex(self, crop_row1, crop_row2, crop_col1, crop_col2, inplace=True):

        if inplace: 
            target = self
        else:
            target = self.copy()
        
        # Crop the array
        nBands = target.getArraySize()[0]
        if nBands == 1:
            target.array = target.array[crop_row1:crop_row2, crop_col1:crop_col2]
        else:
            target.array = target.array[0:nBands, crop_row1:crop_row2, crop_col1:crop_col2]

        # Get Metadata
        xRes, yRes = target.getPixelSize()
        old_orX, old_orY = target.getOriginPoint()

        # Compute new origin point
        new_orX = old_orX + (crop_col1 * xRes)
        new_orY = old_orY + (crop_row1 * yRes)

        # Update the dataset's instance
        target._updateDs((new_orX, xRes, new_orY, yRes, target.ds.GetProjection()))

        if not inplace: return target

    def resize(self, xRes, yRes, method="near", inplace=True):
        res_ds = resizeDs(self.ds, xRes, yRes, method)
        if inplace: 
            self.ds = res_ds
            self._updateArray()
        else:
            return res_ds

    def stack(self, ls_geoim, inplace=True):

        if inplace:
            target = self.copy()
        else:
            target = self

        ls_ar = [self.array] + [geoim.array for geoim in ls_geoim]
        stack_ar = np.array(ls_ar)
        target.array = stack_ar
        target._updateDs()

        return target

    def save(self, outpath, driverName="GTiff"):
        """
        Create a raster file from the instance
        """

        # Get dimensions
        nBands, nRows, nCols = self.getArraySize()

        # Get geographic informations
        xRes, yRes = self.getPixelSize()
        orX, orY = self.getOriginPoint()
        crs = self.ds.GetProjection()

        # Create a new dataset
        outds = makeDs(
            outpath,
            self.array,
            orX,
            xRes,
            orY,
            yRes,
            crs,
            driverName,
            self.ds_encoding)

        # Write on the disk
        outds.FlushCache()

    def quickVisual(self, index = None, band = 0, colors = "viridis"):

        # Compute nCols and nRows
        nBands, nRows, nCols = self.getArraySize()
        if index == None:
            a,b,c,d = 0, nRows-1, 0, nCols-1
        else:
            a,b,c,d = index

        # Plot
        if nBands > 1:
            plt.imshow(self.array[band][a:b, c:d], cmap = colors)

        else:
            plt.imshow(self.array[a:b, c:d], cmap = colors)

        plt.show()
        plt.close()
        return None

    def rgbVisual(self, colorMode=[0,1,2], resize_factor=1, brightness=1, show=False, path=None):

        _, nRows, nCols = self.getArraySize()

        if len(self.array.shape) != 3:
            raise AttributeError("You need a GeoIm in 3 dimensions to display a GeoIm in RGB")

        if self.array.shape[0] < 3:
            raise AttributeError("The GeoIm have only {} channel and we need 3 channels to display it in RGB")

        # Convert array into RGB array

        # Unpack the RGB components is separates arrays
        r = self.array[colorMode[0]]
        g = self.array[colorMode[1]]
        b = self.array[colorMode[2]]

        # data normalization between [0-1]
        r_norm = (r - r[r!=0].min()) / (r.max() - r[r!=0].min()) * 255
        g_norm = (g - g[g!=0].min()) / (g.max() - g[g!=0].min()) * 255
        b_norm = (b - b[b!=0].min()) / (b.max() - b[b!=0].min()) * 255

        # RGB conversion
        # --------------

        # Create a target array
        rgb_ar = np.zeros((nRows, nCols, 3))

        # For each cell of the "board"
        for row in range(nRows):
            for col in range(nCols):

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
        rgb = rgb.resize((nCols * resize_factor, nRows * resize_factor))

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