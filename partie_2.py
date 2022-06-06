"""
TESTS SYNERGIE GEOGRID / GEOIM - PARTIE 2
recalage de rasters de résolution différente
"""
#%%
import raster_tools as rt

# Fabrication des inputs : Raster A et B de même emprise, une seule bande, de résolutions différentes
base_path = r"C:\Users\Eudes\Desktop\tests_GeoGrid\partie_2\05-20090805-0965-6445-LA93-0M50-C070.tif"
sub_area_path = r"C:\Users\Eudes\Desktop\tests_GeoGrid\partie_2\sub_area.shp"

# Ouverture du raster de base
base = rt.openGeoRaster(base_path, roi=sub_area_path)

# Extraction d'une seule bande
b1 = rt.GeoIm(base.pxData[0], base.geogrid)
b1.exportAsRasterFile(r"C:\Users\Eudes\Desktop\tests_GeoGrid\partie_2\base.tif")

#%%
import raster_tools as rt

# Tests de crop et rééchantillonage à partir du changement de GeoGrid

# Définition des chemins
fox1_path = r"C:\Users\Eudes\Desktop\tests_GeoGrid\partie_2\A1\A1.tif"
fox3_path = r"C:\Users\Eudes\Desktop\tests_GeoGrid\partie_2\A3\A3.tif"

# Ouverture de 2 rasters d'emprise et résolution différente
fox1 = rt.openGeoRaster(fox1_path)
fox3 = rt.openGeoRaster(fox3_path)

# Application de la grille de fox3 sur fox1
fox3.changeGeoGrid(fox1.geogrid, inplace=True)