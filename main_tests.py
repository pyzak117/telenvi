#%%
from geoim2 import GeoIm
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt

# On ouvre un geoim
path = r"C:\Users\Eudes\Desktop\tests_GeoGrid\partie_2\A3\A3.tif"
ds = gdal.Open(path)
fox0 = GeoIm(ds)
new = fox0.crop(10, 70, 20, 50, False)