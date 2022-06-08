# Standard librairies
import os

# Third-Party librairies
import numpy as np
from osgeo import gdal, gdalconst
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
from geogrid import createGeoGridFromArray

class GeoIm:

    """
    Describe a georeferenced image. 
    A pixel represent a square part of the space.
    Each pixel must represent a space of the same size.

    attributes
    ----------

        name  | type       | short description
        ----------------------------------------------------------------
        array | np.ndarray | each pixel have one or many values. 
                              Here, they are stored in an array or a matrix. 
                              This array have 2 dimensions (x,y) if the image 
                              represented is monospectral (only one channel).
                              or 3 dimensions (x, y, channel) if the image is 
                              multispectral (just as the color images, with a
                              Red, a Green and a Blue channel)
        ----------------------------------------------------------------
        _geogrid | GeoGrid  | describe the spatial position of the image,
                             in a Spatial Coordinates System given

    methods
    -------
        [...]

    """

    def __init__(self, array, geogrid, resize_method = "near", numFormat = np.float32):

        # Attribute initialisation
        self.array = array
        self._geogrid = geogrid
        self.resize_method = resize_method
        self.numFormat = numFormat

        # Check the compatibility of _geogrid and the data array
        if not self.isValid():
            del self.array
            del self._geogrid
            raise AttributeError("The array and the _geogrid are not compatible")

    def __add__(self, neighboor):
        output = self.array + neighboor.array
        x = GeoIm(output, self._geogrid, self.crs)
        return x
    
    def __sub__(self, neighboor):
        output = self.array - neighboor.array
        x = GeoIm(output, self._geogrid, self.crs)
        return x

    def __mul__(self, neighboor):
        output = self.array * neighboor.array
        x = GeoIm(output, self._geogrid, self.crs)
        return x

    def __truediv__(self, neighboor):
        output = self.array / neighboor.array
        x = GeoIm(output, self._geogrid, self.crs)
        return x

    def __repr__(self):
        self.quickVisual()
        return ""

    def copy(self):
        return GeoIm(self.array, self._geogrid)

    def getGeoGrid(self):
        return self._geogrid

    def setGeoGrid(self, new_grid):
        """
        When the _geogrid of a GeoIm is modified, his data have to be aligned to the new_grid grid.
        The new_grid must be a part of the old. 
        """

        # Check the CRS compatibility between the 2 grids
        if self.geogrid.crs != new_grid.crs :
            raise AttributeError("The new_grid and self.geogrid _geogrid have not the same CRS")

        # Intersect the 2 grids
        inter_grid = self.geogrid.intersect(new_grid)

        # Get bounds of the intersection area
        firstRow= int((inter_grid.yMax - self.geogrid.yMax) / inter_grid.yRes)
        firstCol= int((inter_grid.xMin - self.geogrid.xMin) / inter_grid.xRes)
        lastRow = int((inter_grid.yMin - self.geogrid.yMax) / inter_grid.yRes)-1
        lastCol = int((inter_grid.xMax - self.geogrid.xMin) / inter_grid.xRes)

        print(firstRow, lastRow, firstCol, lastCol)

        if inter_grid.xRes != self.geogrid.xRes or inter_grid.yRes != self.geogrid.yRes:                

            # Make a resample with gdal warp
            # Transform the current GeoIm as a gdal.Dataset
            ds = self.toDs()

            print("DS BEFORE RESAMPLE : ")
            print(ds.RasterXSize)
            print(ds.RasterYSize)

            # Resample with gdal.Warp()
            ds = gdal.Warp(
                "",
                ds,
                format="VRT",
                xRes = inter_grid.xRes,
                yRes = inter_grid.yRes,
                resampleAlg = self.resize_method)

            print("DS AFTER RESAMPLE : ")
            print(ds.RasterXSize)
            print(ds.RasterYSize)

            # Get back to array with crop
            self.array = ds.ReadAsArray(firstCol, firstRow, lastCol-firstCol, lastRow-firstRow).astype(self.)

        # If the resolution is different, the following instructions will not be ran
        elif inter_grid.nRows != self.geogrid.nRows or inter_grid.nCols != self.geogrid.nCols:
            self.array = self.array[firstRow:lastRow, firstCol:lastCol]

        # Update the instance geogrid
        self._geogrid = inter_grid

        # Check the compatibility between the new grid and the new array
        # if not self.isValid():
        #     raise AttributeError("The array size and the _geogrid size are not compatible")

    def resample(self, new_xRes, new_yRes):
        new_grid = createGeoGridFromArray(
            self.array,
            self.geogrid.xMin,
            self.geogrid.yMax,
            new_xRes,
            new_yRes,
            self.geogrid.crs)

        self.setGeoGrid(new_grid)

    def crop(self, firstRow, lastRow, firstCol, lastCol):
        self.array = self.array[firstRow:lastRow, firstCol:lastCol]
        self._geogrid = createGeoGridFromArray(
            self.array,
            self.geogrid.xMin,
            self.geogrid.yMax,
            self.geogrid.xRes,
            self.geogrid.yRes,
            self.geogrid.crs)


    def rgbVisual(self, colorMode=[0,1,2], resize_factor=1, brightness=1, show=False, path=None):

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
        rgb_ar = np.zeros((self._geogrid.nRows, self._geogrid.nCols, 3))

        # For each cell of the "board"
        for row in range(self._geogrid.nRows):
            for col in range(self._geogrid.nCols):

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
        rgb = rgb.resize((self._geogrid.nCols * resize_factor, self._geogrid.nRows * resize_factor))

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


    def getCoordsExtent(self):
        """
        Voué à disparaître -- En attente d'homogénéisation avec les scripts utilisant cette fonctionnalité

        :return:
        --------
            bounding-box coordinates of the GeoIm's spatial extent
        """
        return (self._geogrid.xMin, self._geogrid.yMax, self._geogrid.xMax, self._geogrid.Min)

    def getGeomExtent(self, mode = "ogr"):
        """
        the geometry of the image spatial extent
        """
        if mode == "ogr":
            return self._geogrid.getOGRextent()

        elif mode == "shply":
            return self._geogrid.getShapelyExtent()
 
    def toDs(
        self,
        outpath = "",
        f_format = gdalconst.GDT_Float32,
        driverName = "MEM"):

        """
        build a gdal.Dataset from a GeoIm instance

        :params:
        --------
            outpath : str - the path where you want to save the raster - default : ""
            f_format : gdalconst - the numeric pixel values format - default : float32
            driverName : str - the extension of the raster file - default : "MEM" for 'memory'
        """

        driver = gdal.GetDriverByName(driverName)

        # Check if the array is 2D or 3D
        dim = len(self.array.shape)

        if dim == 2:
            nb_bands = 1
            rows, cols = self.array.shape

        elif dim == 3:
            nb_bands, rows, cols = self.array.shape

        else:
            raise ValueError("Array must be in 2 or 3 dimensions")

        # gdal.Dataset creation
        outDs = driver.Create(outpath, cols, rows, nb_bands, f_format)
        outDs.SetGeoTransform((self._geogrid.xMin, self._geogrid.xRes, 0.0, self._geogrid.yMax, 0.0, self._geogrid.yRes))
        outDs.SetProjection(self._geogrid.crs)

        # Export each band of the 3D array
        if dim == 3:
            for band in range(1, nb_bands+1):
                outDs.GetRasterBand(band).WriteArray(self.array[band-1])

        # Export the unique band
        else:
            outDs.GetRasterBand(1).WriteArray(self.array)

        if driverName != "MEM":
            outDs.FlushCache()
            return None

        return outDs
        
 

    def isValid(self):

        """
        Check the compatibility of the _geogrid with the array of the instances
        """

        # Get the dimensions of the self.array self.array
        dims = len(self.array.shape)

        if dims == 2:
            data_nRows, data_nCols = self.array.shape

        elif dims == 3:
            data_nRows = self.array.shape[1]
            data_nCols = self.array.shape[2]

        else:
            raise ValueError("A self.array self.array must have 2 or 3 dimensions")

        # Compare _geogrid attributes and self.array attributes
        cols_compatibility = data_nCols == self._geogrid.nCols
        rows_compatibility = data_nRows == self._geogrid.nRows

        # return a boolean
        return cols_compatibility == rows_compatibility == True
    
    geogrid = property(fget=getGeoGrid, fset=setGeoGrid)