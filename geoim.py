# Standard librairies
import os

# Third-Party librairies
import numpy as np
from osgeo import gdal, gdalconst
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

class GeoIm:

    """
    Describe a georeferenced image. 
    A pixel represent a square part of the space.
    Each pixel must represent a space of the same size.

    attributes
    ----------

        name   | type       | short description
        ----------------------------------------------------------------
        pxData | np.ndarray | each pixel have one or many values. 
                              Here, they are stored in an array or a matrix. 
                              This array have 2 dimensions (x,y) if the image 
                              represented is monospectral (only one channel).
                              or 3 dimensions (x, y, channel) if the image is 
                              multispectral (just as the color images, with a
                              Red, a Green and a Blue channel)
        ----------------------------------------------------------------
        geogrid | GeoGrid  | describe the spatial position of the image,
                             in a Spatial Coordinates System given

    methods
    -------
        [...]

    """

    def geogrid_fit_on_pxData(self):
        return geogrid_is_valid_from_pxData(self.geogrid, self.pxData)

    def __init__(self, pxData, geogrid):        

        # Attribute initialisation
        self.pxData = pxData
        self.geogrid = geogrid

        # Check the compatibility of geogrid and the data array
        if not self.geogrid_fit_on_pxData():
            raise AttributeError("The array size and the geogrid size are not compatible")

    def __add__(self, neighboor):
        output = self.pxData + neighboor.pxData
        x = GeoIm(output, self.geogrid, self.crs)
        return x
    
    def __sub__(self, neighboor):
        output = self.pxData - neighboor.pxData
        x = GeoIm(output, self.geogrid, self.crs)
        return x

    def __mul__(self, neighboor):
        output = self.pxData * neighboor.pxData
        x = GeoIm(output, self.geogrid, self.crs)
        return x

    def __truediv__(self, neighboor):
        output = self.pxData / neighboor.pxData
        x = GeoIm(output, self.geogrid, self.crs)
        return x

    def __repr__(self):
        self.quickVisual()
        return ""
    
    def changeGeoGrid(self, new_grid, inplace=False):
        """
        When the geogrid of a GeoIm is modified, his data have to be aligned to the new_grid grid.
        The new_grid must be a part of the old. 
        """
        
        # Rename current instance attributes
        old_grid = self.geogrid
        old_data = self.pxData
    
        # Check the CRS compatibility between the 2 grids
        if old_grid.crs != new_grid.crs :
            raise AttributeError("The new_grid and old_grid geogrid have not the same CRS")

        # Get bounds of the new geogrid
        firstRow= int((new_grid.yMax - old_grid.yMax) / old_grid.yRes)
        firstCol= int((new_grid.xMin - old_grid.xMin) / old_grid.xRes)
        lastRow = int((new_grid.yMin - old_grid.yMax) / old_grid.yRes)
        lastCol = int((new_grid.xMax - old_grid.xMin) / old_grid.xRes)

        print(firstRow, firstCol, lastRow, lastCol)

        # Create PIL.Image instance from the data
        im_data = Image.fromarray(old_data)

        # Crop the data and change his resolution with pillow
        im_data_resize = im_data.resize(
            size = (new_grid.nCols, new_grid.nRows),
            resample = Image.Resampling.NEAREST,
            box = (firstCol, firstRow, lastCol, lastRow),
        )

        # Get back to array
        new_pxData = np.asarray(im_data_resize)

        # Verify the compatibility between the new grid and the new pxData
        if not self.geogrid_fit_on_pxData():
            raise AttributeError("The array size and the geogrid size are not compatible")

        if inplace:

            # Change instance attributes
            self.pxData = new_pxData
            self.geogrid = new_grid
    
        else:
            return GeoIm(new_pxData, new_grid)

    def getCoordsExtent(self):
        """
        Voué à disparaître -- En attente d'homogénéisation avec les scripts utilisant cette fonctionnalité

        :return:
        --------
            bounding-box coordinates of the GeoIm's spatial extent
        """
        return (self.geogrid.xMin, self.geogrid.yMax, self.geogrid.xMax, self.geogrid.Min)

    def getGeomExtent(self, mode = "ogr"):
        """
        the geometry of the image spatial extent
        """
        if mode == "ogr":
            return self.geogrid.getOGRextent()

        elif mode == "shply":
            return self.geogrid.getShapelyExtent()
 
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
        dim = len(self.pxData.shape)

        if dim == 2:
            nb_bands = 1
            rows, cols = self.pxData.shape

        elif dim == 3:
            nb_bands, rows, cols = self.pxData.shape

        else:
            raise ValueError("Array must be in 2 or 3 dimensions")

        # gdal.Dataset creation
        outDs = driver.Create(outpath, cols, rows, nb_bands, f_format)
        outDs.SetGeoTransform((self.geogrid.xMin, self.geogrid.xRes, 0.0, self.geogrid.yMax, 0.0, self.geogrid.yRes))
        outDs.SetProjection(self.geogrid.crs)

        # Export each band of the 3D array
        if dim == 3:
            for band in range(1, nb_bands+1):
                outDs.GetRasterBand(band).WriteArray(self.pxData[band-1])

        # Export the unique band
        else:
            outDs.GetRasterBand(1).WriteArray(self.pxData)

        # outDs.FlushCache()
        return outDs
        
    def quickVisual(self, band = 0, colors = "viridis"):
        """
        Show the data as an image

        :params:
        -------
            band : int - if the array of pixels values represent a 
                         multispectral image, with 3 dimensions, 
                         you can choose the band than you want to show here.

            colors : str - a string describing the color-range to use 
                           to show the image
        """

        if len(self.pxData.shape) == 2:
            plt.imshow(self.pxData, cmap = colors)

        elif len(self.pxData.shape) == 3:
            plt.imshow(self.pxData[band], cmap = colors)

        plt.show()
        plt.close()
        return None

    def rgbVisual(self, colorMode=[0,1,2], resize_factor=1, brightness=1, show=False, path=None):
        """
        show the image in colors
        """

        if len(self.pxData.shape) != 3:
            raise AttributeError("You need a GeoIm in 3 dimensions to display a GeoIm in RGB")

        if self.pxData.shape[0] < 3:
            raise AttributeError("The GeoIm have only {} channel and we need 3 channels to display it in RGB")

        # Convert pxData into RGB array

        # Unpack the RGB components is separates arrays
        r = self.pxData[colorMode[0]]
        g = self.pxData[colorMode[1]]
        b = self.pxData[colorMode[2]]

        # data normalization between [0-1]
        r_norm = (r - r[r!=0].min()) / (r.max() - r[r!=0].min()) * 255
        g_norm = (g - g[g!=0].min()) / (g.max() - g[g!=0].min()) * 255
        b_norm = (b - b[b!=0].min()) / (b.max() - b[b!=0].min()) * 255

        # RGB conversion
        # --------------

        # Create a target array
        rgb_ar = np.zeros((self.geogrid.nRows, self.geogrid.nCols, 3))

        # For each cell of the "board"
        for row in range(self.geogrid.nRows):
            for col in range(self.geogrid.nCols):

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
        rgb = rgb.resize((self.geogrid.nCols * resize_factor, self.geogrid.nRows * resize_factor))

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

def geogrid_is_valid_from_pxData(geogrid, array):
    """
    Check the compatibility of a geogrid with an array
    """

    # Get the dimensions of the array array
    dims = len(array.shape)

    if dims == 2:
        data_nRows, data_nCols = array.shape

    elif dims == 3:
        data_nRows = array.shape[1]
        data_nCols = array.shape[2]

    else:
        raise ValueError("A array array must have 2 or 3 dimensions")

    # Compare geogrid attributes and array attributes
    cols_compatibility = data_nCols == geogrid.nCols
    rows_compatibility = data_nRows == geogrid.nRows

    # return a boolean
    return cols_compatibility == rows_compatibility == True