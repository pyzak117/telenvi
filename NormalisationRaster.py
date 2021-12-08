# -*- coding: UTF-8 -*-

from PIL import Image
import os
from osgeo import gdal, gdalconst, gdal_array
import numpy as np
import matplotlib.pyplot as plt
import time
import sys


# ================================ Donnée =====================================

zone = [570812, 625175, 5310543, 5350353]

#================================== Fonction ===================================

def ConversionGIF (pathdossier, ImageSeul = False):
    """

    :param pathdossier: Chemin d'accès 
    :param ImageSeul:
    :return:
    """
    try:
        if ImageSeul == False :
            nameImg = [fichier for fichier in os.listdir(pathdossier) if fichier[-4 :] == ".tif"]
            path = [os.path.join(pathdossier, path) for path in nameImg]
            for e in path :
                Image.open(e).save('{}.gif'.format(e[:-4]), format='GIF')
        elif ImageSeul == True:
            os.chdir(pathdossier)
            e = input("Rentrer le chemin d'accès de l'image : ")
            Image.open(e).save('{}.gif'.format(e[:-4]), format='GIF')

    except NotADirectoryError:
        print("Le chemin d'accès vers l'image n'est pas conforme, \n par exemple sous Linux : '/home/user/Bureau/répertoire_avec_série_d'image'")

def Resample(NameFile, Xres, Yres , algo = gdalconst.GRA_NearestNeighbour):
    """

    :param NameFile: Le chemin d'accès vers l'image à rééchantillonné
    :param Xres: La largeur voulu qui sera donné à un pixel
    :param Yres: La longueur  voulu qui sera donné à un pixel
    :param algo: Le type de rééchenatillonage
    par défaut sur le plus proche voisin
    :return:
    """
    try:
        ds = gdal.Open(NameFile)
        option = gdal.WarpOptions(xRes=Xres, yRes=Yres, resampleAlg=algo)
        ds_resample = gdal.Warp(NameFile[:-4]+"_Resample.tif", ds, options=option)

    except NotADirectoryError:
        print("Le chemin d'accès vers l'image n'est pas conforme, \n par exemple sous Linux : '/home/user/Bureau/Image.tif'")
        sys.exit(1)

def Crop(pathdossier, zone):
    try:

        xMin, xMax, yMin, yMax = zone

        nameImg = [fichier for fichier in os.listdir(pathdossier) if fichier[-4:] == ".tif"]

        path_band20 = [os.path.join(pathdossier, band) for band in nameImg]

        #exemple = gdal.Open(path_band20[0])

        l=[]
        newkey = []

        for e in path_band20:
            ds = gdal.Open(e)
            ds_gt = ds.GetGeoTransform()
            row1 = int((yMax - ds_gt[3]) / abs(ds_gt[5]))
            col1 = int((xMin - ds_gt[0]) / abs(ds_gt[1]))
            row2 = int((yMin - ds_gt[3]) / abs(ds_gt[5]))
            col2 = int((xMax - ds_gt[0]) / abs(ds_gt[1]))
            nArray = np.array(ds.ReadAsArray(col1, row1, col2 - col1 + 1, row2 - row1 + 1))
            l.append(nArray)
            #print(e)
            newkey.append(e[:-4])

        if l[0].shape == l[1].shape or l[1].shape == l[2].shape:
            count = 0
            for narray in l:
                print(narray)
                Image.fromarray(narray).save("{}.tif".format(newkey[count]),format='GIF')
                count+=1
        else:
            print("NOP")

        """
        driver = gdal.GetDriverByName('GTiff')
        count = 0
        for e in l:

            p = driver.Create(newkey[count]+ '.tif' , exemple.RasterXSize, exemple.RasterYSize, 1, gdal.GDT_Float32)
            p.SetGeoTransform(exemple.GetGeoTransform())
            p.SetProjection(exemple.GetProjection())
            p.GetRasterBand(1).WriteArray(e)
            p.FlushCache()
            count+=1
        """

    except NotADirectoryError:
        print('CHEH')
        sys.exit(1)


def Reproj (NameFile,ProjVoulu):
    try:
        ds =gdal.Open(NameFile)
        ProjSource = ds.GetProjection()
        option = gdal.WarpOptions(dstSRS= ProjVoulu)
        ds_reproj = gdal.Warp(NameFile[:-4]+"Reproj.tif",ds, options=option)
        

    except ValueError:
        print("CHEH")
        pass



if __name__ == '__main__':

    debut = time.time()

    Reproj('/home/ju/Bureau/Pente.tif', "EPSG:2154")
    Reproj('/home/ju/Bureau/rennes.roads.tif', "EPSG:2154")
    Reproj('/home/ju/Bureau/rennes.urban_7m.tif', "EPSG:2154")
    Reproj('/home/ju/Bureau/rennes.urban_7m.tif', "EPSG:2154")
    Reproj('/home/ju/Bureau/rennes.urban_7m.tif', "EPSG:2154")

    Resample('/home/ju/Bureau/PenteReproj.tif', 60, 60)
    Resample('/home/ju/Bureau/rennes.roadsReproj.tif', 60, 60)
    Resample('/home/ju/Bureau/rennes.urban_7mReproj.tif', 60, 60)
    Resample('/home/ju/Bureau/rennes.urban_7mReproj.tif', 60, 60)
    Resample('/home/ju/Bureau/rennes.urban_7mReproj.tif', 60, 60)

    #Crop("/home/ju/Bureau/M2/Atelier Prospective - Laurence Moy/Input/Resample_Fait",zone)

    #ConversionGIF('/home/ju/Bureau/M2/AtelierProspective_LaurenceMoy/Input/Resample_Fait/Reproj/Reproj2')

    fin = time.time()
    print("Temps de Calcul : ", fin - debut)

"""

    ds = gdal.Open('/home/ju/Bureau/M2/Atelier Prospective - Laurence Moy/Input/Resample_Fait/Reproj/rennes.urban_7m_ResampleReproj.tif')
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    print(geo)
    #print(proj)

    ds2 = gdal.Open('/home/ju/Bureau/M2/Atelier Prospective - Laurence Moy/Input/Resample_Fait/Reproj/MNT_rennes_ResampleReproj.tif')
    geo2 = ds2.GetGeoTransform()
    proj2 = ds2.GetProjection()
    print(geo2)
    #print(proj2)

    ds3 = gdal.Open('/home/ju/Bureau/M2/Atelier Prospective - Laurence Moy/Input/Resample_Fait/Reproj/rennes.roads_ResampleReproj.tif')
    geo3 = ds3.GetGeoTransform()
    proj3 = ds3.GetProjection()
    print(geo3)
    #print(proj3)
    #================================= Image Référence ====================================
ds=gdal.Open('/home/ju/Bureau/M2/Atelier Prospective - Laurence Moy/Input/rennes.roads.big.tif')
option = gdal.WarpOptions(xRes=10,yRes=10,resampleAlg=gdalconst.GRA_NearestNeighbour)



ds_resample = gdal.Warp('rennes_roads_big_resample.tif',ds,options=option)
NP_roads = np.array(ds_resample)
print(NP_roads.shape)

optionResize = gdal.WarpOptions(width=np.shape(NP_roads)[0],height=np.shape(NP_roads)[1])

#====================================================================================================


urban = gdal.Open('/home/ju/Bureau/M2/Atelier Prospective - Laurence Moy/Input/rennes.urban_7m.tif')
geotra = urban.GetGeoTransform()
resampurban = gdal.Warp('Urban_resample.tif',urban,options=option)
print('Cest le geotra, ds2 : ',geotra)



MNT = gdal.Open('/home/ju/Bureau/M2/Atelier Prospective - Laurence Moy/Input/MNT_rennes.tif')
geo3 = MNT.GetGeoTransform()
print('cest le geo3',geo3)
resamplMNT = gdal.Warp('MNT_2021_Resampl.tif',MNT, options=option)



pathdossier ='/home/ju/Bureau/M2/Atelier Prospective - Laurence Moy/GIF'

nameImg = [fichier for fichier in os.listdir(pathdossier) if fichier[-4 :] == ".tif"]
path= [os.path.join(pathdossier,path) for path in nameImg ]

for e in path :
    ds = Image.open(e).save('{}.gif'.format(e[:-4]),format='GIF')
"""