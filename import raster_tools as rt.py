import raster_tools as rt
path = r"H:\REMOTE_S\TELENVI_PACKAGE\tests\data\S2A_L1C_20220603\T44TMM_20220603T052651_B03.jp2"
s = rt.openGeoRaster(path)
indexes = [
    (0,0,1000,1000),
    (0,1000,1000,2000),
    (1000,0,2000,1000),
    (1000,1000,2000,2000)
]

ls_geoims = []
i = 0

for index in indexes:
    ls_geoims.append(s.cropFromIndex(index, inplace=False))

for geoim in ls_geoims:
    geoim.save(f"c:/users/eudes/desktop/part_{i}.tif")
    i+=1

merge = rt.mergeGeoIms(ls_geoims)

#%%
import raster_tools as rt
path = r"c:/users/eudes/desktop/part_0.tif"
s = rt.openGeoRaster(path)
mos = s.makeMosaic()