# Standard librairies
import os

# Third-Party librairies
import numpy as np
from osgeo import gdal, gdalconst
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

def makeDs(path, array, nb_bands, cols, rows, orX, resX, orY, resY, crs, driverName, encoding):
    newds_driver = gdal.GetDriverByName(driverName)
    newds_format = encoding
    newds = newds_driver.Create(path, cols, rows, nb_bands, newds_format)
    newds.SetGeoTransform((orX, resX, 0.0, orY, 0.0, resY))
    newds.SetProjection(crs)
    if nb_bands > 1:
        for band in range(1, nb_bands+1):
            newds.GetRasterBand(band).WriteArray(array[band-1])
    else:
        newds.GetRasterBand(1).WriteArray(array)
    return newds

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

    def crop(self, crop_row1, crop_row2, crop_col1, crop_col2, inplace=True):

        if inplace: 
            target = self
        else:
            target = self.copy()
        
        # Crop the array
        target.array = target.array[crop_row1:crop_row2, crop_col1:crop_col2]

        # Compute new geographic informations
        old_orX, resX, _, old_orY, __, resY = target.ds.GetGeoTransform()
        new_orX = old_orX + (crop_col1 * resX)
        new_orY = old_orY + (crop_row1 * resY)

        # Update the dataset's instance
        target.updateDs((new_orX, resX, new_orY, resY, target.ds.GetProjection()))

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
        target.updateArray()

        if not inplace: return target

    def save(self, outpath, driverName="GTiff"):
        """
        Create a raster file from the instance
        """

        # Get dimensions
        nb_bands, rows, cols = self.array_dimensions()

        # Get geographic informations
        orX, resX, _, orY, __, resY = self.ds.GetGeoTransform()
        crs = self.ds.GetProjection()

        # Create a new dataset
        outds = makeDs(
            outpath,
            self.array,
            nb_bands,
            cols,
            rows,
            orX,
            resX,
            orY,
            resY,
            crs,
            driverName,
            self.ds_encoding)

        # Write on the disk
        outds.FlushCache()
        
    def array_dimensions(self):
        # Get array dimensions
        ar_dims = len(self.array.shape)
        if ar_dims == 2:
            nb_bands, rows, cols = (1,) + self.array.shape
        elif ar_dims == 3:
            nb_bands, rows, cols = self.array.shape

        return nb_bands, rows, cols

    def updateArray(self):
        """
        Update instance's array from his dataset
        """
        self.array = self.ds.ReadAsArray().astype(self.array_encoding)

    def updateDs(self, geodata = None):
        """
        Update instance's dataset from his array
        """

        # Get geographic informations about the new dataset
        if geodata == None:
            orX, resX, _, orY, _, resY = self.ds.GetGeoTransform()
            crs = self.ds.GetProjection()
        else: 
            orX, resX, orY, resY, crs = geodata

        # Get array dimensions
        nb_bands, rows, cols = self.array_dimensions()

        # Make a new memory dataset
        newds = makeDs(
            "",
            self.array,
            nb_bands,
            cols,
            rows,
            orX,
            resX,
            orY,
            resY,
            crs,
            "MEM",
            self.ds_encoding)

        # Update instance dataset
        self.ds = newds

    def quickVisual(self, band = 0, colors = "viridis"):
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

        if len(self.array.shape) == 2:
            plt.imshow(self.array, cmap = colors)

        elif len(self.array.shape) == 3:
            plt.imshow(self.array[band], cmap = colors)

        plt.show()
        plt.close()
        return None


