# telenvi modules
import telenvi.raster_tools as rt
import telenvi.geoim as geoim

# Standard libraries
from tqdm import tqdm

# Data libraries
import numpy as np
from skimage import morphology
from skimage import measure
from sklearn import cluster

# Image processing
import cv2
from PIL import Image, ImageFilter, ImageEnhance

# Geo Libraries
import shapely
from osgeo import gdal
import geopandas as gpd

def geo_monoband_to_pil(mono_target):
    """
    Convert a monoband geo dataset to rgb Pillow.Image object
    """
    mono_target = rt.Open(mono_target, load_pixels=True)
    return Image.fromarray(mono_target.array).convert('L')

def geo_rgb_to_im_rgb(rgb_target):
    """
    Convert a rgb geo dataset to a rgb Pillow.Image object
    """
    assert rgb_target.getShape()[0] == 3, 'not a rgb target'

    # Extract 3 independant bands
    red_ar, green_ar, blue_ar = rgb_target.array
    
    # Convert them in 3 different Pillow Images objects
    red_im, green_im, blue_im = [Image.fromarray(array).convert('L') for array in (red_ar, green_ar, blue_ar)]

    # Stack them into one Pillow Image object
    rgb_im_stack = Image.merge(mode='RGB', bands = (red_im, green_im, blue_im))

    return rgb_im_stack
    
def mono_im_to_geo_mono(mono_im, geo_template):
    im_array = get_array(mono_im)[0]
    geomono = geo_template.copy()
    geomono.array = im_array
    return geomono
    
def rgb_im_to_geo_rgb(rgb_im, geo_template):

    rgb_im_array = get_array(rgb_im)[0]
    
    # Builds empty geoims
    geo_r_band = geo_template.copy()
    geo_g_band = geo_template.copy()
    geo_b_band = geo_template.copy()

    # Extract RGB arrays
    r_band = rgb_im_array[:, :, 0]
    g_band = rgb_im_array[:, :, 1]
    b_band = rgb_im_array[:, :, 2]

    # Put them into geoims
    geo_r_band.array = r_band
    geo_g_band.array = g_band
    geo_b_band.array = b_band

    # Stack
    georgb = rt.Open(rt.stack((geo_r_band, geo_g_band, geo_b_band)), load_pixels=True)

    return georgb

def blur(target, r):

    # Create a gaussian blur filter
    gaussian_filter = ImageFilter.GaussianBlur(radius=r)

    # Apply it to the rgb image
    target_blurred = target.filter(gaussian_filter)
    return target_blurred

def contrast(target, c):

    # Create an enhancer
    contrast_enhancer = ImageEnhance.Contrast(target)

    # Apply it with a given factor
    target_contrasted = contrast_enhancer.enhance(c)
    return target_contrasted

def canny(target, l, h):

    # Convert our image into numpy array
    target_array = np.array(target)

    # Apply the Open CV Canny algorithm
    edges_array = cv2.Canny(target_array, l, h)

    # Re-build an PIL.Image object
    return Image.fromarray(edges_array)

def edges_detection_chain(target, r, c, l, h):
    step_1 = blur(target, r)
    step_2 = contrast(step_1, c)
    step_3 = canny(step_2, l, h)
    return (step_1, step_2, step_3)
    
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
    
    elif type(input_target) == Image.Image:
        output_array = np.array(input_target)
        
    elif type(input_target) == np.ndarray:
        output_array = input_target

    return output_array, input_is_geoim

def get_auto_clusters(input_target, n_clusters, n_init=10, to_exclude=-999, mode=''):
    """
    input_target : 2D array
    n_clusters   : int, number of clusters to create
    n_init       : int
    to_exclude   : this value will not be taken in account for the clustering
                   and the pixels with this value will keep the same value
    mode         : 'labels', 'barycentres' or '' for both (default behavior)
    """

    # Extract array if the input is a geoim or a dataset
    input_array, input_is_geoim = get_array(input_target)

    # Exclusion des valeurs à éviter
    valid_array  = input_array[input_array != to_exclude]

    # Transformation de la matrice d'entrée pour qu'elle soit valide vis à vis du k-means
    linear_valid_array = valid_array.reshape(-1,1)

    # This will be useful later, for the reshape
    input_linear = input_array.reshape(-1,1)

    # Créée un estimateur KMeans vide
    estimator = cluster.KMeans(
        n_clusters=n_clusters, 
        n_init=n_init)

    # Charge les données dans l'estimateur
    estimator.fit(linear_valid_array)

    # Extrait les labels
    linear_valid_labels = estimator.labels_

    # Réintègre les valeurs initiales
    output_labels_linear = input_linear + 0
    output_labels_linear[output_labels_linear != to_exclude] = linear_valid_labels

    # Pareil pour les barycentres
    linear_valid_barycentres = estimator.cluster_centers_[linear_valid_labels].flatten()
    output_barycentres_linear = input_linear + 0
    output_barycentres_linear[output_barycentres_linear != to_exclude] = linear_valid_barycentres

    # Retransformation matricielle en deux dimensions
    output_labels      = output_labels_linear.reshape(input_array.shape)
    output_barycentres = output_barycentres_linear.reshape(input_array.shape)

    # Combinaison des deux... Ou pas
    if mode == '':
        output_array = np.array((output_labels, output_barycentres))
    elif mode == 'labels':
        output_array = output_labels
    elif mode == 'barycentres':
        output_array = output_barycentres

    # Intégration de la nouvelle matrice dans un geoim
    if input_is_geoim:
        out_geoim = input_target.copy()
        out_geoim.array = output_array
        return out_geoim

    # Ou pas
    return output_array

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

def denoise_binary_image(binary_target, small_objects_min_size = 150, morpho_operator_size = 1, value_to_keep='highest'):

    # Extract array from input target
    binary_array, input_is_geoim = get_array(binary_target)

    # Create a mask to binarize the array
    if value_to_keep[0].lower() == 'h':
        mask = (binary_array > binary_array.min())
    else:
        mask = (binary_array < binary_array.max())

    # Dilatation to connect pixels
    mask_dilated = morphology.dilation(mask, morphology.square(morpho_operator_size))

    # Erosion to delete the noise (isolated pixels)
    mask_eroded = morphology.erosion(mask_dilated, morphology.square(morpho_operator_size))

    # Labeled the regions
    labeled_regions = measure.label(mask_eroded)

    # Delete small regions
    filtered_regions = morphology.remove_small_objects(
        labeled_regions, 
        min_size=small_objects_min_size)

    # Create final output
    if value_to_keep[0].lower() == 'h':
        filtered_regions[filtered_regions > 0] = 1
    else:
        filtered_regions[filtered_regions == 0] = 1

    # Put the array in a geoim
    if input_is_geoim:
        out_geoim = binary_target.copy()
        out_geoim.array = filtered_regions
        return out_geoim
    
    return filtered_regions

def get_binary_contours_bis(binary_target, epsg=''):

    """
    Create a vector layer from a binary raster. Resulting polygons have an attribute 1 or 0.
    binary_target : raster path or gdal.dataset or telenvi.Geoim, the raster to vectorize
    epsg : int, the epsg of the output layer coordinates reference system
    """
    
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

    # Get only the contours with more than 2 segments to build polygons :
    contours = filter(lambda x: len(x) > 2, contours)

    # For each contour
    geometries = []
    print('Vectorisation...')
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

        # Bring them all in a shaped polygon
        out_polygon = shapely.Polygon(out_points)
        geometries.append(out_polygon)

    # Build a geodataframe
    out_gdf = gpd.GeoDataFrame().set_geometry(geometries)
    print('Done')

    # Set a CRS
    if epsg != '':
        out_gdf = out_gdf.set_crs(epsg=epsg)

    return out_gdf

def degrees_to_unit_vector(angle_degrees):
    """
    Convert an angle in degrees to a unit vector (x, y).
    
    Parameters:
    angle_degrees (float): The angle in degrees.
    
    Returns:
    vector (np.ndarray): A 2D unit vector corresponding to the given angle.
    """
    # Convert degrees to radians
    angle_radians = np.deg2rad(angle_degrees)
    
    # Calculate the x and y components of the unit vector
    x = np.cos(angle_radians)
    y = np.sin(angle_radians)
    
    return np.array([x, y])

def measure_direction_homogeneity(df, column_name='direction'):
    """
    Measure the homogeneity of directions given as angles in a DataFrame column 'direction'.
    
    Parameters:
    df (pd.DataFrame): A DataFrame where each row contains a direction (angle in degrees) in the 'direction' column.
    
    Returns:
    homogeneity (float): A measure of directional homogeneity (1.0 means all directions are the same).
    """
    # Convert angles (in degrees) to unit vectors
    vectors = np.stack(df[column_name].apply(degrees_to_unit_vector).values)
    
    # Compute the mean direction vector
    mean_direction = np.mean(vectors, axis=0)
    
    # Normalize the mean direction
    mean_direction /= np.linalg.norm(mean_direction)
    
    # Compute cosine similarity between each vector and the mean direction
    cosine_similarities = np.dot(vectors, mean_direction)
    
    # Calculate the average cosine similarity (homogeneity measure)
    homogeneity = np.mean(cosine_similarities)
    
    return homogeneity