# telenvi modules
import telenvi.raster_tools as rt
import telenvi.geoim as geoim

# Standard libraries
from tqdm import tqdm

# Data libraries
import numpy as np
from sklearn import cluster

# Image processing
import cv2

# Geo Libraries
import shapely
from osgeo import gdal
import geopandas as gpd

def get_array(input_target):
    """
    extract np.array from different geodata containers
    """
    input_is_geoim = False

    if type(input_target) == gdal.Dataset:
        output_array = input_target.ReadAsArray()
    
    elif type(input_target) == rt.geoim.Geoim:
        output_array = input_target.array
        input_is_geoim = True
    
    elif type(input_target) == np.ndarray:
        output_array = input_target

    return output_array, input_is_geoim

def get_auto_clusters(input_target, n_clusters=2, n_init=10):
    """
    Apply a KMeans clusterization on the geoim
    """

    # Extract array
    input_array, input_is_geoim = get_array(input_target)

    # Reshape the array for the clustering
    input_reshaped = input_array.reshape(-1,1)

    # Create the classifier
    k_means_classifier = cluster.KMeans(
        n_clusters=n_clusters, 
        n_init=n_init)

    # Fit to the data
    k_means_classifier.fit(input_reshaped)

    # Get the labels by prediction
    labels = k_means_classifier.predict(input_reshaped)

    # Get the value of the barycentre of each label
    input_segmented = k_means_classifier.cluster_centers_[labels].reshape(input_array.shape)

    if input_is_geoim:

        # write the new array in a new geoim
        out_geoim = input_target.copy()
        out_geoim.array = input_segmented
        return out_geoim

    return input_segmented

def shift_hist(input_target, breakpoint):

    input_array, input_is_geoim = get_array(input_target)

    # Change data type to allowed place for negative values
    if input_array.dtype == np.uint8:
        input_array = input_array.astype(np.int16)

    # Premiere chose, créer une matrice de maximum
    max_array = np.zeros_like(input_array) + input_array.max()

    # Maintenant, calculons les distance de chaque valeur au maximum
    dist_to_max = max_array - input_array

    # Crééons un masque booléen pour identifier les pixels
    # que l'on souhaite shifter en dessous de 0.
    mask = input_array > breakpoint

    # Maintenant, soustrayons à 0 nos valeurs, 
    # seulement sur les pixels où le masque est True
    input_array[mask] = np.zeros_like(input_array[mask])-1 - dist_to_max[mask]

    # Maintenant, réhaussons le tout pour repasser avec un minimum à zéro
    shifted_array = input_array + (0 - input_array.min())

    # Range la matrice dans un nouveau geoim si l'input en était un
    if input_is_geoim:

        # write the new array in a new geoim
        out_geoim = input_target.copy()
        out_geoim.array = shifted_array
        return out_geoim

    return shifted_array

def get_manual_clusters(input_target, thresholds):

    if type(thresholds) == list:
        thresholds = np.array(thresholds)

    # Extract array
    input_array, input_is_geoim = get_array(input_target)

    # Define thresholds array
    thresholds = [float('-inf')] + sorted(thresholds) + [float('inf')]

    # Classify
    bins = np.digitize(input_array, bins=thresholds).astype(input_array.dtype)

    # Put the array in a geoim
    if input_is_geoim:
        out_geoim = input_target.copy()
        out_geoim.array = bins
        return out_geoim

    return bins

def get_binary_contours(binary_target):

    # Load data in a Geoim
    if type(binary_target) != geoim.Geoim:
        binary_target = rt.Open(binary_target, load_pixels=True)

    # Extract dataset and array
    target_ds = binary_target.ds
    binary_ar = binary_target.array.astype(np.uint8)

    # Load geographic metadata
    or_x, or_y = rt.getOrigin(target_ds)
    x_pixel_size, y_pixel_size = rt.getPixelSize(target_ds)

    # Detect binary contours in the array
    contours, _ = cv2.findContours(
        binary_ar,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour
    geometries = []
    print('Vectorization...')
    for in_contour in tqdm(contours):

        # For each point
        out_points = []
        for point_coords in in_contour:

            # Extract his image referential coordinates 
            x_im_ref_coord, y_im_ref_coord = point_coords.flatten()

            # Convert them in the epsg of the image
            x = or_x + (x_im_ref_coord * x_pixel_size)
            y = or_y + (y_im_ref_coord * y_pixel_size)

            # Create a point
            out_point = shapely.Point((x,y))
            out_points.append(out_point)

        # Bring the all in a shaped polygon
        out_polygon = shapely.Polygon(out_points)
        geometries.append(out_polygon)

    # Build a geodataframe
    geooutput = gpd.GeoDataFrame().set_geometry(geometries)
    print('Done')
    return geooutput