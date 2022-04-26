# telenvi
Some remote sensing tricks from telenvi master students.
Contain tools to work on satellites images. You can easily load georeferenced raster files in python, compute indices on them or extract their values, then make crops, resample and reprojection.

## GitHub repo
The code is available here : github.com/pyzak117/telenvi

## Dependancies
to work with telenvi you need thoses libraries:
  - gdal
  - numpy
  - geopandas
  - matplotlib

## Installation
pip install telenvi

## Use from a python script

```  
from telenvi import raster_tools
```  

#### open a raster, mono or multi-spectral
```
target = raster_tools.openGeoRaster("aGeoreferencedRaster.tif")
```

#### open a part of raster
```
target = raster_tools.openGeoRaster(
  "aGeoreferencedRaster.tif",
  roi = "aShapefileContainingOneOrManyPolygons.shp",
  ft = 0 # the id of the polygon into the attribute table of the shapefile
  )
```

#### pixel-values extraction
array = target.pxlV

#### export a raster
```
tarrget.exportAsRasterFile("pathToANewGeoTiff.tif")
```


You can download a tutorial to learn how to use the telenvi package here :
https://mega.nz/file/voNTUYrR#WHtp_stDGi-p3_TfUj9K_K76H56n0B1L0hGixMvVzkY
You will find a .rar archive containing jupyter notebooks and data to test the package.

# Contact
You can contact me at thibaut.duvanel@univ-savoie.fr if you have any questions.
