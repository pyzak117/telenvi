#%%
from geoim2 import openGeoRaster
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt

path = "c:/users/eudes/desktop/rgb.tif"
s = openGeoRaster(path, roi=r"C:\Users\Eudes\Desktop\tests_GeoGrid\partie_2\cuts\cuts.shp", res=4)