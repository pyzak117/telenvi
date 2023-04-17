module_description = """
--- telenvi.GeoIm ---
Use GeoIm objects to read only once a raster file array.
Then make crops, compute optical indexes or statisticals
data from this array, with geo attributes under the hand
"""

# telenvi modules
import telenvi.raster_tools as rt
import telenvi.vector_tools as vt

# Standard libraries
import os

# Data libraries
import numpy as np
import pandas as pd
import numpy.ma as ma
from matplotlib import pyplot as plt

# Geo libraries
import shapely

# import richdem as rd
import geopandas as gpd
from geocube.api.core import make_geocube

# Image processing libraries
from PIL import Image, ImageEnhance

class GeoIm:

    def __init__(self, target, array=None):

        if type(array) not in [np.ndarray] :
            self.ds = rt.getDs(target)
            self._array = self.ds.ReadAsArray()

        elif type(array) in [np.ndarray]:
            ds = rt.getDs(target)

            # First we check dataset and array compatibility
            if not rt.getShape(ds)[1:] == rt.getShape(array)[1:]:
                print(f"dataset shape {rt.getShape(ds)[1:]} != array shape {rt.getShape(array)[1:]}")
                return None

            # Then we assign ds and array as geoim instance attributes
            self.ds = ds
            self._array = array
            self.updateDs()

        self.mask_value = 0

    def getArray(self):
        return self._array

    def delArray(self):
        pass

    def setArray(self, newArray):
        """
        this method is called each time we change the geoim instance array
        because if the geoim array change, the array contained in the
        instance.Dataset must change to
        """

        # If the new array is a masked_array
        if type(newArray) == ma.core.MaskedArray:
            
            # First we assign the masked array to the instance
            self._array = ma.copy(newArray)

            # Then we "smash" the masked array into a normal one-dimensionnal
            # array to store them into the dataset
            newArray.data[newArray.mask == True] = self.mask_value
            newArray = newArray.data

        elif type(newArray) in [np.ndarray]:
            self._array = np.copy(newArray)

        # In both cases, we can now update the Dataset with the new array
        self.updateDs(newArray)

    array = property(getArray, setArray, delArray)

    def updateDs(self, newArray=None):

        if type(newArray) not in [np.ndarray]:
            newArray = self.array

        new_ds = rt.create(
            newArray,
            "",
            self.getOrigin()[0],
            self.getOrigin()[1],
            self.getPixelSize()[0],
            self.getPixelSize()[1],
            self.ds.GetProjection())
        self.ds = new_ds

    def updateArrayFromDs(self):
        self.array = self.ds.ReadAsArray()

    def __add__(self, n):

        # Create an instance copy
        nGeoIm = self.copy()

        # If it's a geoim we extract is array
        if type(n) == GeoIm:
            n = n.array

        # Then we make an operation between copy instance array and n - which can be a int, or a float...
        # The array contained in the n.array Dataset is automatically updated thanks to the array setter
        nGeoIm.array += n

        return nGeoIm

    def __sub__(self, n):
        nGeoIm = self.copy()
        if type(n) == GeoIm:
            n = n.array
        nGeoIm.array -= n
        return nGeoIm
    
    def __mul__(self, n):
        nGeoIm = self.copy()
        if type(n) == GeoIm:
            n = n.array
        nGeoIm.array *= n
        return nGeoIm
    
    def __truediv__(self, n):
        nGeoIm = self.copy()
        if type(n) == GeoIm:
            n = n.array
        nGeoIm.array /= n
        return nGeoIm

    def __pow__(self, n):
        nGeoIm = self.copy()
        if type(n) == GeoIm:
            n = n.array
        nGeoIm.array = nGeoIm.array ** n
        return nGeoIm

    def __repr__(self):
        print(
f"""pixel size : {self.getPixelSize()}
origin     : {self.getOrigin()}
bands      : {self.getShape()[0]}
rows       : {self.getShape()[1]}
columns    : {self.getShape()[2]}
SCR epsg   : {self.getEpsg()}
SCR name   : {self.getProjName()}
array type : {self.array.dtype}""")
        return ""

    def copy(self):
        return GeoIm(self.ds, self.array)

    def getOrigin(self):
        return rt.getOrigin(self.ds)

    def getGeoBounds(self):
        return rt.getGeoBounds(self.ds)

    def getPixelSize(self):
        return rt.getPixelSize(self.ds)
    
    def getShape(self):
        return rt.getShape(self.ds)
    
    def drawGeomExtent(self, geomType="ogr"):
        return rt.drawGeomExtent(self.ds, geomType)
    
    def getEpsg(self):
        return rt.getEpsg(self.ds)
    
    def getProjName(self):
        return rt.getProjName(self.ds)
    
    def getJsonProj(self):
        return rt.getJsonProj(self.ds)

    def resize(self, xRes = None, yRes = None, model = None, method = "near", inplace=False):

        target = self
        if not inplace :
            target = self.copy()
        
        ds_resized = rt.resize(target, "", xRes, yRes, model, method)
        target.ds = ds_resized
        target.updateArrayFromDs()

        return target

    def cropFromVector(self, vector, polygon = 0, verbose = False, inplace=False):
        """
        vector : shapely.geometry.polygon.Polygon or str - path to a shapefile
        polygon : id of the feature, if vector is a shapefile
        """
        
        # We get the polygon geo extent
        if type(vector) == str:
            layer = gpd.read_file(vector)
            bounds = layer["geometry"][polygon].bounds

        if type(vector) == gpd.GeoDataFrame:
            bounds = vector.iloc[0]["geometry"].bounds

        elif type(vector) == tuple:
            bounds = vector

        elif type(vector) in (shapely.geometry.polygon.Polygon, shapely.geometry.multipolygon.MultiPolygon):
            bounds = vector.bounds

        # And we cut the geoim on it
        return self.cropFromBounds(bounds, verbose = verbose, inplace = inplace)

    def cropFromRaster(self, model, verbose = False, inplace=False):

        # We get the croper dataset and his geo extent
        modelBounds = rt.getGeoBounds(rt.getDs(model))

        # And we cut the geoim on it
        return self.cropFromBounds(modelBounds, verbose = verbose, inplace = inplace)

    def cropFromBounds(self, bounds, verbose = False, inplace=False):

        # We get the matrixian coordinates of the intersection area between the raster and the box
        crop_indexes = rt.spaceBox_to_arrayBox(bounds, self.ds, self.array)

        # And we cut the geoim on it
        return self.cropFromIndexes(crop_indexes, verbose = verbose, inplace = inplace)

    def cropFromIndexes(self, indexes, inplace=False, verbose = False):
        """
        indexes : tuple - (row1, col1, row2, col2)
        """        

        # We create a copy of the geoim instance if not inplace arg
        target = self.copy()

        # Get metadata
        xRes, yRes = target.getPixelSize()
        orX, orY = target.getOrigin()

        # Give a name to the indexes
        row1, col1, row2, col2 = indexes

        # Extract the array part between thoses indexes
        new_array = target.array[row1:row2, col1:col2]

        # Assign this new array to the geoim
        target.setArray(new_array)

        # Compute new origin point
        new_orX = orX + (col1 * xRes)
        new_orY = orY + (row1 * yRes)

        # Build a new geotransform
        new_geotransform = (new_orX, xRes, 0.0, new_orY, 0.0, yRes)
        
        # Set the target's geotransform
        target.ds.SetGeoTransform(new_geotransform)

        if inplace: 
            self = target
        
        return target        

    def maskFromThreshold(self, threshold, greater = True, opening_kernel_size = None):
        """
        Change the instance array into numpy.ma.masked_array according to a threshold apply on the array instance values.

        Args:
            threshold (float): each value of the array is compared to this threshold.
            greater (bool, optional): if True, the valid pixels have them with a greater value than the threshold. If False, it's the pixels with a lower value than the threshold. Defaults: True.
            opening_kernel_size (float, optional): kernel size which is used to reduce small and isolated pixels. Defaults to None.

        Returns:
            numpy.ma.masked_array: a 2 band array. First band is the data itself and the second one is a binary array representing the mask. 0 : mask is unactive; 1 : mask is active
        """

        # 0 : MASK IS UNACTIVE - DATA IS TO SEE
        # 1 : MASK IS ACTIVE   - DATA IS TO MASK

        # Instance's array binary classification - b for 'binary'
        b = np.copy(self.array)
        
        # The mask must be UNACTIVE on the pixels which respect the condition
        if greater == True:
            b[ b > threshold] = 0 # If greater, unactive mask is apply on pixels GREATER than the threshold
        else:
            b[ b < threshold] = 0 # Else, unactive mask is apply on pixels LOWER than the threshold

        # Now, all the valid pixels are transformed in 0
        # So, we can mask everything else
        b[ b != 0] = 1

        # Apply an opening operator
        if opening_kernel_size != None:
            import cv2 as cv
            kernel = np.ones((opening_kernel_size, opening_kernel_size))
            b = cv.morphologyEx(b, cv.MORPH_OPEN, kernel)

        # Change the instance's array into masked_array
        self.array = ma.masked_array(data = self.array, mask = b)

        return self.array

    def maskFromVector(self, vector, inside=True, verbose=False, epsg=2154):
        """
        change the instance array into masked_array.
        According to the 'inside' argument, the masked vectors 
        are either inside or outside the vector outlines.
        
        - PARAMETERS -
        vector : str or a geopandas.GeoDataFrame or geopandas.GeoSeries or pandas.Series
        a shapefile containing one or many geometric objects

        inside : boolean - describe if the data to keep unmasked 
        is inside (True) or outside (False) the vector outlines.

        verbose : boolean - to see how many features intersect the GeoIm

        - RETURNS -
        masked_array : numpy.ma.masked_array - an array of 2 dimensions.
        the first array is the normal array
        the second is a binary array representing the mask. 
        0 : mask is unactive
        1 : mask is active
        """

        # Get a GeoDataFrame from a geofile path
        vector_name = ""
        if type(vector) == str:
            vector_name = os.path.basename(vector)
            vector = gpd.read_file(vector)

        # Or from a GeoSerie
        elif type(vector) in (gpd.GeoSeries, pd.core.series.Series):
            vector = gpd.GeoDataFrame([vector])

        elif type(vector) in (shapely.geometry.multipolygon.MultiPolygon, shapely.geometry.polygon.Polygon):
            vector = gpd.GeoDataFrame([{'geometry':vector}]).set_crs(epsg=epsg)

        # Then set is crs to be the same than the geoim
        # vector.set_crs(epsg=epsg, allow_override=True, inplace=True)
        # This line raise a warning

        # Find the vector epsg
        epsg = vector.crs.to_epsg()

        # Select the features intersecting the image geo-extent
        vector = vector[vector["geometry"].intersects(self.drawGeomExtent(geomType="shly")) == True].copy()
        if verbose and vector_name != "":
            print(f"{vector_name} : {len(vector)} polygon intersecting the geoim")
        if len(vector) == 0:
            return None

        # Affect a value to the pixels inside the vector features outlines
        if inside :
            vector["rValue"] = 0
        else:
            vector["rValue"] = 1

        geomjson = self.drawGeomExtent().ExportToJson()[:-1] + ', "crs": {"properties": {"name": "EPSG:' + str(epsg) + '"}}}'

        # Rasterization
        mask = make_geocube(
            vector,
            measurements=["rValue"],
            geom = geomjson,
            resolution = self.getPixelSize()[0])

        mask = np.array(mask.rValue)

        # Affect a value to the pixels outside the shapefile features outlines
        if inside :
            mask[np.isnan(mask)] = 1
        else :
            mask[np.isnan(mask)] = 0

        # Clean the raster edges : make_geocube add sometimes a line and a colum
        _, nRows, nCols = self.getShape()
        mask = mask[0:nRows, 0:nCols]

        # Adapt the mask to the array if it's a multispectral raster
        nBands = self.getShape()[0]
        if nBands > 1:
            o = []
            [o.append(mask) for i in range(nBands)]
            mask = np.array([o])

        self.array = ma.masked_array(data = self.array, mask = mask)
        return self.array

    def maskZeros(self):
        """
        change the instance array into masked_array.
        
        - PARAMETERS -
        value : float or int - the array value to mask

        - RETURNS -
        masked_array : numpy.ma.masked_array - an array of 2 dimensions.
        the first array is the normal array
        the second is a binary array representing the mask. 
        0 : mask is unactive
        1 : mask is active        
        """
        self.array = ma.masked_where(self.array == 0, self.array)
        return self.array

    def unmask(self):
        """
        Get off the mask on the instance's array
        """
        if type(self.array) == ma.core.MaskedArray:
            self.array = self.array.data

    def median(self, band=0):
        """
        compute the raster median or a band median if multispectral. 
        The argument 'band' is refering to matrixian indexes, so the
        band 1 have the index 0.
        """
        if type(self.array) in [np.ndarray]:
            return np.median(self.array)
        elif type(self.array) == ma.core.MaskedArray:
            return ma.median(self.array)

    def mean(self, band=0):
        """
        compute the raster mean or a band mean if multispectral. 
        The argument 'band' is refering to matrixian indexes, so the
        band 1 have the index 0.
        """
        if type(self.array) in [np.ndarray] :
            return np.mean(self.array)
        elif type(self.array) == ma.core.MaskedArray:
            return ma.mean(self.array)

    def makeMosaic(self, thumbsY=2, thumbsX=2):
        """
        build many geoims side by side from the instance
        :params:
            nbSquaresByAx : int
                default : 2
                the number of cells to cells along the X size and the Y size
                from the current instance. 2 means you will have 4 GeoIms in
                return. The current instance will be split in 2 lines and 2 cols.

        :returns:
            mosaic : list
                a list of GeoIms
        """

        cells_nRows=int(self.getShape()[1]/thumbsY)
        cells_nCols=int(self.getShape()[2]/thumbsX)

        mosaic=[]
        for row in range(thumbsY):
            for col in range(thumbsX):
                row1=cells_nRows * row
                col1=cells_nCols * col
                row2=row1 + cells_nRows
                col2=col1 + cells_nCols
                mosaic.append(self.cropFromIndexes((row1, col1, row2, col2)))

        return mosaic

    def splitBands(self):
        """
        send a list of geoims monospectral for each band in the current instance
        """
        nBands=self.getShape()[0]

        if nBands == 1:
            return [self.copy()]

        elif nBands > 1:
            bands=[]
            for band in self.array:
                new=GeoIm(self.ds, band)
                new.updateDs(band)
                bands.append(new)

            return bands

    def show(self, index=None, band=0, colors="viridis", bar=True):

        """
        :descr:
            display one band of the GeoIm
        
        :params:
            index : tuple
                default : None - all the image is displayed.
                alternative : (firstRow, firstColumn, lastRow, lastColumn)
                described a matrixian area to display

            band : int
                default=0
                the index of the band to display if the geoim is multispectral

        :returns:
            None
        """

        # Compute nCols and nRows
        nBands, nRows, nCols=self.getShape()
        if index == None:
            row1, col1, row2, col2 = 0, 0, nRows-1, nCols-1
        else:
            row1, col1, row2, col2 = index

        # Plot
        if nBands > 1:
            plt.imshow(self.array[band][row1:row2, col1:col2], cmap=colors)

        else:
            plt.imshow(self.array[row1:row2, col1:col2], cmap=colors)

        if bar:
            plt.colorbar()

        plt.show()
        plt.close()
        return None

    def save(self, outpath, mask = False):
        if mask:
            rt.write(self.ds, outpath, self.mask_value)
        else:
            rt.write(self.ds, outpath)

    def rgbVisual(self, colorMode=[0,1,2], resize_factor=1, brightness=1, show=False, path=None):

        """
        :descr:
            display 3 bands of the GeoIm in RGB mode
        
        :params:
            colorMode : list or tuple
                the order of the 3 bands to display

            resize_factor : int
                default : 1
                allow to "zoom" on the image if the area is to 
                small to be correctly visualized

            brightness : int
                default : 1
                allow to improve the RGB composition brightness. 

            show : boolean
                default : False
                if True,the image is displayed in the os system image reader.
                when this method is called from a Jupyter Notebook, 
                there's no need to set it on True
            
            path : str
                default : None
                if not None, the image is not displayed but saved to this path

        :returns:
            rgb : PIL.Image
                a RGB image        
        """

        _, nRows, nCols=self.getShape()

        if len(self.array.shape) != 3:
            raise AttributeError("You need a GeoIm in 3 dimensions to display a GeoIm in RGB")

        if self.array.shape[0] < 3:
            raise AttributeError("The GeoIm have only {} channel and we need 3 channels to display it in RGB")

        # Convert array into RGB array

        # Unpack the RGB components is separates arrays
        r=self.array[colorMode[0]]
        g=self.array[colorMode[1]]
        b=self.array[colorMode[2]]

        # data normalization between [0-1]
        r_norm=(r - r[r!=0].min()) / (r.max() - r[r!=0].min()) * 255
        g_norm=(g - g[g!=0].min()) / (g.max() - g[g!=0].min()) * 255
        b_norm=(b - b[b!=0].min()) / (b.max() - b[b!=0].min()) * 255

        # RGB conversion
        # --------------

        # Create a target array
        rgb_ar= np.zeros((nRows, nCols, 3))

        # For each cell of the "board"
        for row in range(nRows):
            for col in range(nCols):

                # We get the separate RGB values in each band
                r=r_norm[row][col]
                g=g_norm[row][col]
                b=b_norm[row][col]

                # We get them together in little array
                rgb_pixel= np.array([r,g,b])

                # And we store this little array on the board position
                rgb_ar[row][col]=rgb_pixel

        rgb=Image.fromarray(np.uint8(rgb_ar))

        # Adjust size
        rgb=rgb.resize((nCols * resize_factor, nRows * resize_factor))

        # Adjust brightness
        enhancer=ImageEnhance.Brightness(rgb)
        rgb=enhancer.enhance(brightness)

        # Display
        if show:
            rgb.show()

        # Save
        if path != None:
            rgb.save(path)

        # Return PIL.Image instance
        return rgb

    def getCentroids(self):
        pX, pY = self.getPixelSize()
        _, nRows, nCols = self.getShape()
        orX, orY = self.getOrigin()
        centroids=[]
        for row in range(nRows):
            for col in range(nCols):
                x = orX + (col * pX)
                y = orY + (row * pY)
                centroids.append((x,y))
        return centroids

    def inspectGeoPoint(self, geoPoint):
        """
        extract the pixel(s) values under a geoPoint
        """

        # Get Matrixian coordinates of the geographic point
        row, col = rt.spaceCoord_to_arrayCoord(geoPoint, self)

        # Just a block to process GeoIm with many bands
        if self.getShape()[0] > 1:
            values = []
            for band in self.array:
                values.append(band[row][col])
            return values
        else:
            return self.array[row][col]

    def inspectGeoLine(self, geoLine):
        """
        extract the pixels values under a geoLine
        """

        # Create a point on each pixel along the line
        geoPoints = vt.getGeoPointsAlongGeoLine(geoLine, self.getPixelSize()[0])

        # Extract pixel values on each point
        values = [self.inspectGeoPoint(p) for p in geoPoints]
        return values

    def inspectGeoPolygon(self, geoPolygon):
        """
        extract pixels values contained in the geoPolygon
        """

        # Define intervals between points
        xGap = self.getPixelSize()[0]
        yGap = self.getPixelSize()[0]

        # Get the pixels values inside the polygon
        geoPoints = vt.getGridInGeoPolygon(geoPolygon, xGap, yGap)
        values = [self.inspectGeoPoint((p.x, p.y)) for p in geoPoints]
        return values

    def inspectRibsAlongSpine(self, spine, ribLength, ribStep, ribOrientation='v'):
        """
        extract pixels values along 
        multiple perpendicular lines (ribs)
        regularly sampled along another line (the spine)  
        """

        # Get the ribs
        ribs = vt.serializeGeoLines(spine, ribLength, ribStep, ribOrientation)

        # And now, extract pixels values along each rib
        values = [self.inspectGeoLine(rib) for rib in ribs]
        return values

# %%
