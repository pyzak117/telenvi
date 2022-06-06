#%%
from osgeo import gdal, gdalconst
import raster_tools as rt

def dsFromGeoIm(
    geoim,
    f_format = gdalconst.GDT_Float32,
    driverName = "MEM"):

    """
    Export a GeoIm object into a raster file georeferenced.

    :params:
    --------
        outP (str) : the path where you want to save the raster
        f_format (gdalconst) : the image file format
        driverName (str) : the extension of the raster file
    """

    driver = gdal.GetDriverByName(driverName)

    # Check if the array is 2D or 3D
    dim = len(geoim.pxData.shape)

    if dim == 2:
        nb_bands = 1
        rows, cols = geoim.pxData.shape

    elif dim == 3:
        nb_bands, rows, cols = geoim.pxData.shape

    else:
        raise ValueError("Array must be in 2 or 3 dimensions")

    # gdal.Dataset creation
    outDs = driver.Create("", cols, rows, nb_bands, f_format)
    outDs.SetGeoTransform((geoim.geogrid.xMin, geoim.geogrid.xRes, 0.0, geoim.geogrid.yMax, 0.0, geoim.geogrid.yRes))
    outDs.SetProjection(geoim.geogrid.crs)

    # Export each band of the 3D array
    if dim == 3:
        for band in range(1, nb_bands+1):
            outDs.GetRasterBand(band).WriteArray(geoim.pxData[band-1])

    # Export the unique band
    else:
        outDs.GetRasterBand(1).WriteArray(geoim.pxData)

    # outDs.FlushCache()
    return None

fox1_path = r"C:\Users\Eudes\Desktop\tests_GeoGrid\partie_2\A1\A1.tif"
fox1 = rt.openGeoRaster(fox1_path)