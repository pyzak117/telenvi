# telenvi
Some remote sensing tricks from telenvi master students.
Contain tools to work on satellites images. You can easily load georeferenced raster files in python, compute indices on them or extract their values, then make crops, resample and reprojection.

## GitHub repo
The code is available here : github.com/pyzak117/telenvi

## Dependancies
to work with telenvi you need the following libraries:
  - gdal
  - numpy
  - geopandas
  - matplotlib
  - pillow (PIL)
  - shapely

## Installation
pip install telenvi

## Use from a python script
```
from telenvi import raster_tools as rt
```
