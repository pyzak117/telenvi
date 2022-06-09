# Standard librairies
from multiprocessing.sharedctypes import Value
import os

# Third-Party librairies
import numpy as np
from osgeo import gdal, gdalconst, osr
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
import geopandas as gpd

def makeDs(
    path,
    array,
    nBands,
    nCols,
    nRows,
    orX,
    resX,
    orY,
    resY,
    crs,
    driverName,
    ds_encoding):

    # Dataset metadata
    newds_driver = gdal.GetDriverByName(driverName)
    newds = newds_driver.Create(path, nCols, nRows, nBands, ds_encoding)

    # Dataset geodata setup
    newds.SetGeoTransform((orX, resX, 0.0, orY, 0.0, resY))
    newds.SetProjection(crs)

    # Load data into the dataset
    if nBands > 1:
        for band in range(1, nBands+1):
            newds.GetRasterBand(band).WriteArray(array[band-1])
    else:
        newds.GetRasterBand(1).WriteArray(array)

    return newds

def openGeoRaster(
    target_path,
    indexToLoad = None,
    roi = None,
    clip = None,
    ft = 0,
    crs = None,
    res = None,
    algo = "near",
    array_encoding = np.float32,
    ds_encoding = gdalconst.GDT_Float32):

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

    # Check CLIP mode
    CLIP = False
    if clip != None:

        # Check argument validity
        if type(clip) != str:
            raise ValueError("error 3.0 : Clip argument must be a path to a raster file")

        # Extract geo informations and switch on the crop and resample arguments
        roi = clip
        res = gdal.Open(clip).GetGeoTransform()[1]    

        CLIP = True

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
                    print("error 6.2 : invalid crop raster")

            """
            AJOUTER L'INTERSECTION DU RASTER D'ENTREE
            ET DE LA ZONE DEFINIE PAR CE CROP
            """

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
    nBands = 1
    if BANDSMODE == 0:
        array = inDs.ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(array_encoding)

    elif BANDSMODE == 1:
        array = inDs.GetRasterBand(indexToLoad).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(array_encoding)

    elif BANDSMODE == 2:
        band1 = inDs.GetRasterBand(indexToLoad[0]).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(array_encoding)
        array = np.array([band1])
        for index in indexToLoad[1:]:
            band = inDs.GetRasterBand(index).ReadAsArray(col1, row1, col2-col1+1, row2-row1+1).astype(array_encoding)
            array = np.append(array, [band], axis=0)

    # Get custom arrays dimensions
    ar_dims = len(array.shape)
    if ar_dims == 2:
        nBands, nRows, nCols = (1,) + array.shape
    elif ar_dims == 3:
        nBands, nRows, nCols = array.shape

    if CLIP:
        """
        DECALER LES PIXELS
        """
        pass

    customDs = makeDs(
        path = "",
        array = array,
        nBands = nBands,
        nCols = nCols,
        nRows = nRows,
        orX = orX,
        orY = orY,
        resY = yRes,
        resX = xRes,
        crs = projection,
        driverName = "MEM",
        ds_encoding = ds_encoding)
    
    return GeoIm(customDs, ds_encoding, array_encoding)

class GeoIm:

    """
    ATTENTION : PARTOUT OU LE DATASET EST MODIFIE, L'ARRAY DOIT L'ETRE AUSSI
        self.array = self.ds.ReadAsArray().astype(self.array_encoding)

    ATTENTION : PARTOUT OU L'ARRAY EST MODIFIE, LE DATASET DOIT L'ETRE AUSSI
        self.ds = gdal.Driver.Create(new_dataset)
    
    """

    def __init__(self, GDALdataset, ds_encoding = gdalconst.GDT_Float32, array_encoding = np.float32):
        self.array_encoding = array_encoding
        self.ds_encoding = ds_encoding
        self.ds = GDALdataset
        self.array = self.ds.ReadAsArray().astype(self.array_encoding)

    def __repr__(self):
        self.quickVisual()
        return ""

    def __getitem__(self, index):
        return self.array[index]

    def copy(self):
        return GeoIm(self.ds)

    def getArraySize(self):
        ar_dims = len(self.array.shape)
        if ar_dims == 2:
            nBands, nRows, nCols = (1,) + self.array.shape
        elif ar_dims == 3:
            nBands, nRows, nCols = self.array.shape

        return nBands, nRows, nCols

    def getOriginPoint(self):
        return (self.ds.GetGeoTransform()[0], self.ds.GetGeoTransform()[3])

    def getPixelSize(self):
        return (self.ds.GetGeoTransform()[1], self.ds.GetGeoTransform()[5])

    def moveRaster(self, offsetX, offsetY, inplace=True):
        """
        Move the raster by offsetX coordinates system reference unity, same for offsetY
        """

        # Copy instance
        if inplace: 
            target = self
        else:
            target = self.copy()

        # Get metadata
        nBands, nRows, nCols = target.getArraySize()
        resX, resY = target.getPixelSize()
        orX, orY = target.getOriginPoint()

        # Create a copy of the dataset but by changing the orX and orY
        newds = makeDs(
            "",
            target.array,
            nBands,
            nCols,
            nRows,
            orX + offsetX,
            resX,
            orY + offsetY,
            resY,
            target.crs,
            "MEM",
            target.ds_encoding)

        # Update
        target.ds = newds

        # Return
        if not inplace: return target

    def crop(self, crop_row1, crop_row2, crop_col1, crop_col2, inplace=True):

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
        resX, resY = target.getPixelSize()
        old_orX, old_orY = target.getOriginPoint()

        # Compute new origin point
        new_orX = old_orX + (crop_col1 * resX)
        new_orY = old_orY + (crop_row1 * resY)

        # Update the dataset's instance
        target._updateDs((new_orX, resX, new_orY, resY, target.ds.GetProjection()))

        if not inplace: return target

    def resize(self, resX, resY, method="near", inplace=True):

        if inplace: 
            target = self
        else:
            target = self.copy()

        ds_resized = gdal.Warp(
            destNameOrDestDS="",
            srcDSOrSrcDSTab = target.ds,
            format = "VRT",
            xRes = resX,
            yRes = resY,
            resampleAlg = method)
        
        target.ds = ds_resized
        target._updateArray()

        if not inplace: return target

    def save(self, outpath, driverName="GTiff"):
        """
        Create a raster file from the instance
        """

        # Get dimensions
        nBands, nRows, nCols = self.getArraySize()

        # Get geographic informations
        resX, resY = self.getPixelSize()
        orX, orY = self.getOriginPoint()
        crs = self.ds.GetProjection()

        # Create a new dataset
        outds = makeDs(
            outpath,
            self.array,
            nBands,
            nCols,
            nRows,
            orX,
            resX,
            orY,
            resY,
            crs,
            driverName,
            self.ds_encoding)

        # Write on the disk
        outds.FlushCache()
  
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
            resX, resY = self.getPixelSize()
            orX, orY = self.getOriginPoint()
            crs = self.ds.GetProjection()
        else:
            orX, resX, orY, resY, crs = geodata

        # Get array dimensions
        nBands, nRows, nCols = self.getArraySize()

        # Make a new memory dataset
        newds = makeDs(
            "",
            self.array,
            nBands,
            nCols,
            nRows,
            orX,
            resX,
            orY,
            resY,
            crs,
            "MEM",
            self.ds_encoding)

        # Update instance dataset
        self.ds = newds

    def quickVisual(self, index = None, band = 0, colors = "viridis"):
        """
        plot a band of the raster

        :params:
        --------
            band : int - if the array of pixels values represent a 
                        multispectral image, with 3 dimensions, 
                        you can choose the band than you want to show here.

            colors : str - a string describing the color-range to use 
                        to show the image
        """

        nBands, nRows, nCols = self.getArraySize()
        if index == None:
            a,b,c,d = 0, nRows-1, 0, nCols-1
        else:
            a,b,c,d = index

        if nBands > 1:
            plt.imshow(self.array[band][a:b, c:d], cmap = colors)

        else:
            plt.imshow(self.array[a:b, c:d], cmap = colors)

        plt.show()
        plt.close()
        return None

    def rgbVisual(self, colorMode=[0,1,2], resize_factor=1, brightness=1, show=False, path=None):

        _, nnRows, nnCols = self.getArraySize()

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
        rgb_ar = np.zeros((nnRows, nnCols, 3))

        # For each cell of the "board"
        for row in range(nnRows):
            for col in range(nnCols):

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
        rgb = rgb.resize((nnCols * resize_factor, nnRows * resize_factor))

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