# Third-Party librairies
import numpy as np
import geopandas as gpd
from osgeo import gdal, gdalconst, ogr, osr
from matplotlib import pyplot as plt

class GeoIm:
    """
    Describe a georeferenced raster.
    
    attributes
    ----------
        pxData (np.ndarray) : an array representing the pixels values
        geoGrid (tuple) : a GeoGrid object representing the pixels grid
    """

    def __init__(self, pxData, geoGrid):
        self.pxData = pxData
        self.geoGrid = geoGrid

    def __add__(self, neighboor):
        res = self.pxData + neighboor.pxDatapytpython
        x = GeoIm(res, self.geoGrid)
        return x
    
    def __sub__(self, neighboor):
        res = self.pxData - neighboor.pxData
        x = GeoIm(res, self.geoGrid)
        return x

    def __mul__(self, neighboor):
        res = self.pxData * neighboor.pxData
        x = GeoIm(res, self.geoGrid)
        return x

    def __truediv__(self, neighboor):
        res = self.pxData / neighboor.pxData
        x = GeoIm(res, self.geoGrid)
        return x

    def __repr__(self):
        self.quickVisual()
        return ""

    def exportAsRasterFile(
        self,
        outP,
        format = gdalconst.GDT_Float32,
        driverName = "GTiff"):

        """
        Export a GeoIm object into a raster file georeferenced.

        :param:
        -------
            outP (str) : the path where you want to save the raster
            format (gdalconst) : the image format
            driverName (str) : the extension of the raster file
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
        outDs = driver.Create(outP, cols, rows, nb_bands, format)
        outDs.SetGeoTransform((self.geoGrid.xLeft, self.geoGrid.cellLengthX, 0.0, self.geoGrid.yTop, 0.0, self.geoGrid.cellLengthY))
        outDs.SetProjection(self.geoGrid.crs)

        # Export each band of the 3D array
        if dim == 3:
            for band in range(1, nb_bands+1):
                outDs.GetRasterBand(band).WriteArray(self.pxData[band-1])

        # Export the unique band
        else:
            outDs.GetRasterBand(1).WriteArray(self.pxData)

        outDs.FlushCache()
        print("\n" + os.path.basename(outP) + " OK")
        return None

    def quickVisual(self, band = 0, colors = "viridis"):
        """
        Show the pixels values of a GeoIm object.

        :param:
        -------
            band (int) : if the array of pixels values represent a multispectral image, with 3 dimensions, you can choose the band than you want to show here.
            colors (str) : a string describing the color-range you want to use to show the image
        """

        if len(self.pxData.shape) == 2:
            plt.imshow(self.pxData, cmap = colors)

        elif len(self.pxData.shape) == 3:
            plt.imshow(self.pxData[band], cmap = colors)

        plt.show()
        plt.close()
        return None