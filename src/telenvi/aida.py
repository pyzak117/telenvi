#%%
# telenvi modules
import telenvi.raster_tools as rt
import telenvi.geoim as geoim

# Standard libraries
from tqdm import tqdm
import string

# Data libraries
import pandas as pd
import numpy as np
from skimage import morphology
from skimage import measure
from sklearn import cluster
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix

# Visualisation libraries
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# Image processing
import cv2
from PIL import Image, ImageFilter, ImageEnhance

# Geo Libraries
import shapely
from osgeo import gdal
import geopandas as gpd
from matplotlib.patches import Patch

def geo_monoband_to_pil(mono_target):
    """
    Convert a monoband geo dataset to rgb Pillow.Image object
    """
    mono_target = rt.Open(mono_target, load_pixels=True)
    mono_target.array = mono_target.array.astype(np.float32)
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

def sharp(target, radius=2, percent=150, threshold=3):

    # Create filter
    print(radius, percent, threshold)
    sharp_filter = ImageFilter.UnsharpMask(radius, percent, threshold)

    # Apply it to the image
    target_sharpened = target.filter(sharp_filter)
    return target_sharpened

def canny(target, l, h):
    """
    Perform Canny edge detection on a grayscale image.

    Detects edges by computing image gradients and applying hysteresis
    with two thresholds to classify strong and weak edges.

    Parameters
    ----------
    h : float
        Lower threshold for edge linking (weak edge threshold).
    l : float
        Upper threshold for strong edge detection.

    Notes
    -----
    - Pixels with gradient ≥ l are considered strong edges.
    - Pixels with gradient between h and l are kept
      only if connected to strong edges.
    - Pixels below h are discarded.
    - For best results, apply Gaussian blur before edge detection.
    """

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

def get_clusters_kmeans(
    *ys,
    df=None,
    columns=None,
    n_clusters=3,
    n_init=10,
    random_state=None
    ):
    """
    Flexible KMeans clustering

    Option 1:
        get_clusters_kmeans(y1, y2, y3, ...)

    Option 2:
        get_clusters_kmeans(df=df, columns=["col1", "col2", ...])

    Returns:
        labels, barycentres, estimator
    """

    # --- Case 1: DataFrame input ---
    if df is not None and columns is not None:
        sub = df.dropna(subset=columns)
        input_array = sub[columns].values

    # --- Case 2: raw arrays ---
    elif len(ys) > 0:
        input_array = np.column_stack(ys)

    else:
        raise ValueError("Provide either arrays (*ys) or (df + columns)")

    # --- KMeans ---
    estimator = cluster.KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state
    )

    estimator.fit(input_array)

    labels = estimator.labels_
    barycentres = estimator.cluster_centers_

    return labels, barycentres, estimator

def predict_hue_from_y1_y2_from_arrays(y1, y2, model):
    """
    Predict the hue (categorical variable) from two quantitative variables y1 and y2 using a trained model.
    
    Parameters:
    y1 (array-like): First quantitative variable.
    y2 (array-like): Second quantitative variable.
    model: A trained classification model with a predict method.
    
    Returns:
    predictions (np.ndarray): Predicted hue values.
    """
    # Ensure y1 and y2 are numpy arrays
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    # Reshape y1 and y2 to be 2D arrays with one column
    if len(y1.shape) == 1:
        y1 = y1.reshape(-1, 1)
    if len(y2.shape) == 1:
        y2 = y2.reshape(-1, 1)
    
    # Combine y1 and y2 into a single feature matrix
    X = np.hstack((y1, y2))
    
    # Use the model to predict the hue
    predictions = model.predict(X)
    
    return predictions

def get_auto_clusters_image(input_target, n_clusters, n_init=10, to_exclude=-999, mode=''):
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

def get_manual_clusters(input_target, thresholds, values_style='median'):

    if type(thresholds) == list:
        thresholds = np.array(thresholds)

    # Extract array
    input_array, input_is_geoim = get_array(input_target)

    # Define thresholds array
    thresholds = [float('-inf')] + sorted(thresholds) + [float('inf')]

    # Classify
    bins = np.digitize(input_array, bins=thresholds).astype(input_array.dtype) - 1

    # Write the median values of the input_array

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

# =====================================================
# Main function
# =====================================================

def explore_linear_relation(
    x=None,
    y=None,
    title='linear data visualisation',
    x_label='predictor',
    y_label='dependant',
    figsize=None,
    get_mad=False,
    data=None,
    hue=None,
    palette_dict=None,
    hue_order=None,
    s=1,
    alpha=None,
    ax=None,
    pts_color='black',
    reg_line_color='red',
    reg_line_alpha=1,
    reg_line_width=1,
    reg_line_style='solid',
    scores_text_color='black',
    xbound=None,
    ybound=None,
    show_score=True,
    show_legend=True,
    mad_lines_color=None,
    x_units=None,
    y_units=None,
    show_reg_line_label=True,
    visualize_relation=True,
    reg_line_label_note='',
    pts_label_field=None
):

    # -----------------------------
    # Preprocess
    # -----------------------------
    X, y_true, x_label, y_label, is_multivariate, df_out = preprocess_data(
        x, y, data, x_label, y_label
    )

    # -----------------------------
    # Fit model
    # -----------------------------
    model = LinearRegression().fit(X, y_true)
    y_pred = model.predict(X)

    # -----------------------------
    # Output dataframe
    # -----------------------------
    pred_col = f"pred_{y_label.lower().replace(' ', '_')}"
    df_out[pred_col] = y_pred

    # -----------------------------
    # Visualization
    # -----------------------------
    if visualize_relation:

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.set_title(title)

        if is_multivariate:
            x_vis = y_pred
            y_vis = y_true
            x_vis_label = pred_col
            y_vis_label = y_label
        else:
            x_vis = X.flatten()
            y_vis = y_true
            x_vis_label = x_label
            y_vis_label = y_label

        # Scatter
        plot_scatter(
            x_vis, y_vis, ax,
            hue=hue, palette_dict=palette_dict,
            s=s, alpha=alpha, pts_color=pts_color,
            show_legend=show_legend, data=df_out
        )

        # Regression line
        plot_regression_line(
            x_vis, y_pred if not is_multivariate else x_vis,
            model if not is_multivariate else None,
            ax, x_vis_label, y_vis_label,
            reg_line_color, reg_line_width,
            reg_line_alpha, reg_line_style,
            show_reg_line_label, reg_line_label_note,
            x_units, y_units
        )

        # MAD lines
        if get_mad:
            plot_mad_lines(x_vis, y_vis, y_pred if not is_multivariate else x_vis,
                           ax, mad_lines_color, reg_line_color)

        # Score
        if show_score:
            add_scores_text(x_vis, y_vis, y_pred if not is_multivariate else x_vis,
                            ax, scores_text_color)

        # Points labels
        if pts_label_field is not None :
            
            # Use coordinates of x variable
            if not is_multivariate:
                df_out=data.copy()
                x_coords_label = x_label
                y_coords_label = y_label

            # Use the coordinates of the prediction
            else:
                x_coords_label = f"pred_{y_label}"
                y_coords_label = y_label 

            for x, y, label in zip(df_out[x_coords_label], df_out[y_coords_label], df_out[pts_label_field]):
                ax.annotate(
                    label,
                    (x,y),
                    xytext=(4,4),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )

        ax.set_xlabel(x_vis_label)
        ax.set_ylabel(y_vis_label)

        if xbound is not None:
            ax.set_xbound(xbound)
        if ybound is not None:
            ax.set_ybound(ybound)

        return ax, model, df_out

    return model, df_out


# =====================================================
# Preprocessing
# =====================================================

def preprocess_data(x, y, data, x_label, y_label):

    if data is not None:
        if isinstance(x, list):
            X = data[x].values
            is_multivariate = True
        else:
            X = data[x].values.reshape(-1, 1)
            is_multivariate = False

        y_vals = data[y].values
        df_out = data.copy()

        if x_label == 'predictor':
            x_label = " + ".join(x) if isinstance(x, list) else x
        if y_label == 'dependant':
            y_label = y

    else:
        X = np.array(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            is_multivariate = False
        else:
            is_multivariate = True

        y_vals = np.array(y)
        df_out = pd.DataFrame({'y': y_vals})

    return X, y_vals, x_label, y_label, is_multivariate, df_out


# =====================================================
# Plot helpers
# =====================================================

def plot_scatter(
    x, y, ax, hue, palette_dict,
    s, alpha, pts_color, show_legend, data
):
    if hue is not None and hue in data.columns:
        sns.scatterplot(
            x=x, y=y, ax=ax, hue=data[hue],
            palette=palette_dict, s=s, alpha=alpha, legend=show_legend
        )
    else:
        sns.scatterplot(
            x=x, y=y, ax=ax, color=pts_color, s=s, alpha=alpha
        )


def plot_regression_line(
    x, y_pred, model, ax, x_label, y_label,
    color, width, alpha, style,
    show_label, label_note, x_units, y_units
):
    if model is not None and show_label:
        if x_units is None: x_units = x_label
        if y_units is None: y_units = y_label
        label = f"{label_note}{model.intercept_:.3f}{y_units} + {model.coef_[0]:.6f}*{x_units}"
    else:
        label = None

    sns.lineplot(
        x=x, y=y_pred, ax=ax,
        color=color, linewidth=width,
        alpha=alpha, linestyle=style, label=label
    )


def plot_mad_lines(x, y_true, y_pred, ax, mad_lines_color, default_color):
    color = mad_lines_color or default_color
    mad = 0
    for xi, yt, yp in zip(x, y_true, y_pred):
        sns.lineplot(x=[xi, xi], y=[yt, yp], ax=ax, color=color)
        mad += abs(yt - yp)
    mad /= len(x)
    print(f"MAD: {mad:.4f}")


def add_scores_text(x, y_true, y_pred, ax, scores_text_color):
    r2_val = r2_score(y_true, y_pred)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(
        xlim[1] - 0.02*(xlim[1]-xlim[0]),
        ylim[0] + 0.02*(ylim[1]-ylim[0]),
        f"r2: {r2_val:.3f}",
        fontsize=10, ha='right', va='bottom',
        color=scores_text_color,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )


def get_anova(df, hue, y, equal_var=True, nan_policy='raise'):
    """
    Perform one-way ANOVA on the given DataFrame `df`, grouping by `hue` and testing the means of `y`.
    """
    samples = [df[df[hue]==v][y].values for v in df[hue].unique()]
    samples = [s for s in samples if len(s) > 1]
    f, p = stats.f_oneway(*samples, equal_var=equal_var, nan_policy=nan_policy)
    return f, p, p < 0.05

def get_tukey(df, hue, y, v_order=None, ax=None):
    """
    Perform Tukey's HSD test on the given DataFrame `df`, grouping by `hue` and testing the means of `y`.
    """
    # Get the samples
    if v_order is None:
        v_order = df[hue].unique() # We need to know the order of the values

    all_samples = [df[df[hue]==v][y].values for v in v_order]
    valid_samples = [s for s in all_samples if len(s) > 1]
    wrong_samples = [s for s in all_samples if len(s) <= 1]
    valid_v_order = [v for i, v in enumerate(v_order) if len(all_samples[i]) > 1]

    # Perfor the Tukey HSD test
    res = stats.tukey_hsd(*valid_samples)

    # Visualise the results
    if ax is None:
        fig, ax = plt.subplots()

    # Binary mask : True (1) or False (0) for each pair of groups
    # True, 1 will be shown in green because higher than 0 on the cmap
    # False, 0, will be shown in red
    cmap = 'RdYlGn' 
    mask = res.pvalue <= 0.05

    # Set extent so grid matches cell edges
    ax.imshow(mask, cmap=cmap)

    # Add a lagend
    # Create a custom legend for the colors
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Not significantly different'),
        Patch(facecolor='green', edgecolor='black', label='Significantly different')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9, facecolor='white')

    print(wrong_samples)

    # Set ticks and labels
    ax.set_xticks(range(len(valid_v_order)))
    ax.set_xticklabels(valid_v_order)
    ax.set_yticks(range(len(valid_v_order)))
    ax.set_yticklabels(valid_v_order)
    return res.statistic, res.pvalue, res.confidence_interval, ax

def show_confusion_matrix(Y_test, Y_test_pred, class_names, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    # Get and reshape confusion matrix data
    matrix = confusion_matrix(Y_test, Y_test_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    sns.heatmap(matrix, annot=True, annot_kws={'size':10}, cmap='mako', linewidths=0.2, ax=ax)

    tick_marks = np.arange(len(class_names)) + 0.5
    ax.set_xticks(tick_marks, class_names, rotation=25)
    ax.set_yticks(tick_marks, class_names, rotation=0)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    return ax

def show_density_contours(
    x_col,
    y_col,
    data,
    figsize=None,
    linewidth=1,
    alpha=1,
    levels=[0.25, 0.5, 0.75, 1],
    fill=True,
    ax=None,
    color=None,
    cmap='Reds',
    **kwargs
    ):
    """
    Show density contours of two variables x and y, optionally grouped by a hue variable.
    If only one level is given, use the exact color (not a colormap) for the contour.
    """ 

    if type(levels) == int or type(levels) == float:
        levels = [levels]

    # Extract x and y
    x = data[x_col].values
    y = data[y_col].values

    # Pre-process
    if len(x.shape) != 2:
        x = x.reshape(-1, 1)
    if len(y.shape) != 2:
        y = y.reshape(-1, 1)

    # Create the figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # If only one level, set color directly (not cmap)
    if color is not None:
        sns.kdeplot(
            x=x.flatten(),
            y=y.flatten(),
            ax=ax,
            linewidth=linewidth,
            alpha=alpha,
            fill=fill,
            levels=levels,
            color=color,
            **kwargs
        )
    else:
        sns.kdeplot(
            x=x.flatten(),
            y=y.flatten(),
            ax=ax,
            linewidth=linewidth,
            alpha=alpha,
            fill=fill,
            levels=levels,
            cmap=cmap,
            **kwargs
        )

    return ax

def get_w(
    dem,
    dir_ins,
    e_w = 4,
    s_w = 1,
):
    """
    Compute an index (arbitrarily called "W) based on rasters of 2 drivers : direct insolation and altitude. 
            
    params:
        dem : Geoim, the digital elevation model
        dir_ins : Geoim, the direct insolation raster
        e_w : float, the weight of the elevation driver
        s_w : float, the weight of the direct insolation driver
        save_w : bool, whether to save the resulting raster
        out_path : str, the path to save the resulting directory where the raster will be saved
        out_path_note : str, a note to add to the name of the resulting raster

    return:
        w : Geoim, the resulting W index raster

    The W index is computed as follows:
    - Normalize the elevation and direct insolation rasters between 0 and 1 across the study area
    - Reverse the elevation values because higher elevation = lower temperature
    - Compute the W index as a weighted sum of the normalized elevation and direct insolation rasters
    - Normalize the W index between 0 and 1
    - Invert the W index so that high values correspond to cold ground (low direct insolation and high elevation)

    Note: This index is purely relative and has no physical meaning. It is only meant to be used as a relative index to compare different areas within the same study area.    
    """

    # Get the arrays from the geoims
    e = dem.array
    s = dir_ins.array

    # Normalize values between 0 and 1 across our study area
    e_norm = normalize_array(e)
    s_norm = normalize_array(s)

    # Reverse elevation because higher elevation = lower temperature
    e_norm_inv = 1-e_norm

    # Compute W index
    w = (e_norm_inv * e_w) + (s_norm * s_w)

    # Normalize it
    w_norm = normalize_array(w)

    # Invert it so that high value = cold ground
    w_norm_inv = 1-w_norm

    # Geometrize it
    w = dir_ins.copy()
    w.array = w_norm_inv

    return w

def normalize_array(x, A=0, B=1):
    """
    Min-max scaling of x to range [A, B].

    Parameters:
        x : array-like
        A : float, lower bound of target range
        B : float, upper bound of target range
    """

    x_min = np.min(x)
    x_max = np.max(x)

    # Edge case where all the values are identical
    # The, division by 0 (x_max - x_min) will raise an error
    if x_max == x_min:

        # Full_like return an array with the shape of x, where all the values are equals to our A argument
        return np.full_like(x, A, dtype=float)

    return A + (B - A) * (x - x_min) / (x_max - x_min)

def get_y_from_x_on_linear_regmodel(linear_regression_model):
    print(f"{linear_regression_model.coef_[0]}x + {linear_regression_model.intercept_}")
    return lambda x: linear_regression_model.intercept_ + linear_regression_model.coef_[0] * x

def cluster_df_on_quantiles(target_df, target_field, n_clusters=4, labels=None, out_field=None):
    """
    Automatically cluster a dataframe based on the values of it's rows of a given field
    """

    # Get the relative quantiles given the desired number of clusters
    rel_qs = np.linspace(0, 1, n_clusters+1)

    # Get the corresponding bounds in the data serie
    abs_qs = target_df[target_field].quantile(rel_qs)

    # Define the labels
    if labels is None:
        labels =list(string.ascii_letters.upper())
    labels = labels[:n_clusters]

    # Define the clusters field name
    if out_field is None:
        out_field = 'cluster'

    # Establish the clustering based on the values of each rows and the absolute bounds
    # Each label correspond to the features with a value lower than the bound and higher than the previous bound
    target_df[out_field] = pd.cut(target_df[target_field], abs_qs, include_lowest=True, labels=labels)

    return target_df

def assign_color_from_clusters(target_df, cluster_field, cmap='tab20'):

    # Get the total number of clusters
    n_clusters = len(target_df[cluster_field].unique())

    # For each cluster label
    for i, cluster_label in enumerate(target_df[cluster_field].unique()):

        # Get the color from the colormap
        color = plt.get_cmap(cmap)(i / n_clusters)

        # Converti  the color to hexadecimal
        hex_color = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))

        # Assign the hexadecimal colorcode to the rows belonging to the cluster
        target_df.loc[target_df[cluster_field] == cluster_label, f'{cluster_field}_color'] = hex_color

    return target_df

def from1D_to_2D_with_mask(data, mask, geotemplate, mask_value=-1):

    # # Création d'un geoim d'accueil de la future matrice 2D et instanciation de la matrice
    geodims = geotemplate.array.shape
    geo_foo = geotemplate.copy()
    
    # # Création d'une nouvelle matrice pleine de zéros aux dimensions de la matrice masquée
    foo_ar = np.zeros(mask.shape)
    
    # # Application du masque sur la matrice plate
    foo_ar[mask == False] = mask_value
    foo_ar[mask == True] = data
    
    # # Retransformation en 2D à partir des dimensions de la matrice originelle
    geo_foo_ar = foo_ar.reshape(geodims)
    geo_foo.array = geo_foo_ar
    return geo_foo

def add_groups_legend(ax, target_df, column, show_counts=True, legend_fontsize=8, legend_loc='upper right', show_bounds=None, bounds_field=None, rounder=2, labels_order=None):
    """
    Adds a legend to the map for the_groupss defined in the target_df.

    Parameters:
    - ax: Matplotlib axis object where the legend will be added.
    - target_df: GeoDataFrame containing the_groups ed data with colors.
    - column: The field in target_df that contains the_groups labels.
    - show_counts: Boolean indicating whether to show counts in the legend.
    - legend_fontsize: Font size for the legend text.
    - legend_loc: Location of the legend on the map.
    - show_bounds : 'mean', 'min_max'
    - bounds_field = name of the field to get the stats for each categoriy / bound

    Returns:
    - ax: Matplotlib axis object with the added legend.
    """

    # Define the dict of match between labels and colors
    category_colors_match = {}
    if labels_order is None:
        for label in target_df[column].unique():
            color = target_df[target_df[column] == label][f'{column}_color'].values[0]
            category_colors_match[label] = color
        
    else:
        for label in labels_order:
            color = target_df[target_df[column] == label][f'{column}_color'].values[0]
            category_colors_match[label] = color

    # Define the patches elements - First case : only the feature counts of each category
    if show_counts == True and show_bounds is None:
        legend_elements = [Patch(facecolor=color, label=f"{label} ({len(target_df[target_df[column] == label])})") for label, color in category_colors_match.items()]

    elif show_counts == True and show_bounds == 'min_max':
        legend_elements = [Patch(facecolor=color, label=f"{label} : {np.round(target_df[target_df[column] == label][bounds_field].min(), rounder)} ; {np.round(target_df[target_df[column] == label][bounds_field].max(), rounder)} ({len(target_df[target_df[column] == label])})")  for label, color in category_colors_match.items()]

    elif show_counts == True and show_bounds == 'mean':
        legend_elements = [Patch(facecolor=color, label=f"{label} : {np.round(target_df[target_df[column] == label][bounds_field].mean(), rounder)} ({len(target_df[target_df[column] == label])})")  for label, color in category_colors_match.items()]

    elif show_counts == False and show_bounds == 'min_max':
        legend_elements = [Patch(facecolor=color, label=f"{label} : {np.round(target_df[target_df[column] == label][bounds_field].min(), rounder)} ; {np.round(target_df[target_df[column] == label][bounds_field].max(), rounder)}")  for label, color in category_colors_match.items()]

    elif show_counts == False and show_bounds == 'mean':
        legend_elements = [Patch(facecolor=color, label=f"{label} : {np.round(target_df[target_df[column] == label][bounds_field].mean(), rounder)} ; {np.round(target_df[target_df[column] == label][bounds_field].max(), rounder)}")  for label, color in category_colors_match.items()]
    
    # Case : nothing to display as label
    else:
        legend_elements = [Patch(facecolor=color, label=f"{label}") for label, color in category_colors_match.items()]

    # Implement the legend
    ax.legend(handles=legend_elements, title='', fontsize=legend_fontsize, loc=legend_loc).set_zorder(100)

    return ax