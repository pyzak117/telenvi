module_description = """
--- telenvi.vector_tools ---
Functions to process vector geo data through geopandas
"""

# Geo libraries
from pathlib import Path
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import numbers
import contextily as cx
from matplotlib import pyplot as plt
from shapely.ops import polygonize
from osgeo import gdal, ogr, osr

import math

swissTopoMap = cx.providers.SwissFederalGeoportal.NationalMapColor
swissTopoMapGr = cx.providers.SwissFederalGeoportal.NationalMapGrey
swissIm = cx.providers.SwissFederalGeoportal.SWISSIMAGE
esriIm = cx.providers.Esri.WorldImagery
franceIm = cx.providers.GeoportailFrance.orthos
franceHistIm = cx.providers.GeoportailFrance.Orthoimagery_Orthophotos_1950_1965

def Open(layer_source, layer=None, set_crs_epsg=None, to_crs_epsg=None):
    """
    Return a GeoDataFrame from vector file (shapefile or geopackage)
    """

    # If it's a geometry we just create a gdf with 1 feature & 1 column
    if isGeometry(layer_source):
        layer = getGeoDf(layer_source)
        
    if type(layer_source) == gpd.GeoDataFrame:
        layer = layer_source

    if str(layer_source).lower().endswith('.shp'):
        layer = gpd.read_file(layer_source)
    
    elif str(layer_source).lower().endswith('.gpkg'):
        if layer is None:
            layer = gpd.read_file(layer_source)
        else:
            layer = gpd.read_file(layer_source, layer=layer)

    if set_crs_epsg is not None:
        layer = layer.set_crs(epsg=set_crs_epsg)
    
    if to_crs_epsg is not None:
        layer = layer.to_crs(epsg=to_crs_epsg)

    return layer

def share_same_geotype(targets):
    """
    Check if the objects described in targets have the same type
    """
    types_of_objects = [type(t) for t in targets]
    uniques_types = pd.Series(types_of_objects).drop_duplicates().tolist()
    return len(uniques_types) == 1

def getGeoSerie(targets):
    """
    Funnel between different geometric objects represented in tuples, arrays, shapely geoms, geodataframe... and a geoserie
    """
    if type(targets) in [gpd.GeoDataFrame, pd.DataFrame]:
        return targets.geometry

    shapely_objects = [getGeometry(t) for t in targets]

    assert share_same_geotype(shapely_objects), "Targets must have the same geotype"

    return gpd.GeoSeries(shapely_objects)

def getGeoDf(targets, epsg=None):
    """
    Funnel between different geometric objects represented in tuples, arrays, shapely geoms, geoseries... and a geodataframe
    """

    if type(targets) == gpd.GeoDataFrame:
        return targets

    if type(targets) not in [list, tuple, pd.DataFrame, gpd.GeoSeries, pd.Series, np.ndarray]:
        targets = [targets]

    shapely_objects = [getGeometry(t) for t in targets]

    assert share_same_geotype(shapely_objects), "Targets must have the same geotype"

    gdf = gpd.GeoDataFrame(shapely_objects, columns=['geometry'])

    if epsg is not None:
        gdf = gdf.set_crs(epsg=2056)
    return gdf

def isGeometry(target):
    return isinstance(target, shapely.geometry.base.BaseGeometry)
    
def getGeometry(target = None, x = None, y = None, geom_type='polygon'):
    """
    Return a shapely.geometry object from different cases
    examples :
        target = [x, y] -> send a point
        target = [(x1, y1), (x2, y2)], geom_type = 'line' -> send a line
        target = [(x1, y1), (x2, y2)], geom_type = 'line' -> send a polygon (default case)
        target = [(x, y)] -> send a point
        target = None, x = 5, y = 6 -> send a point (5,6)
        target = geoserie -> send geoserie.geometry
        target = geodataframe -> send the geometry of the first feature - we assume there is only one feature in all the gdf
        target = string -> the string refers to a gpkg or shp path -> send the geometry of the first feature
        """

    # We didn't get target so we need x and y
    if target is None:
        assert x is not None and y is not None, 'input arguments invalids'
        target = (x,y)
    
    # It's already a shapely.geometry (type could be shapely.Polygon, shapely.Point, shapely.LineString...)
    if isGeometry(target):
        return target
        
    # If it's a geoserie or a serie we extract the geometry column
    if type(target) in [pd.Series, gpd.GeoSeries]:
        geometry = target.geometry

    # If it's a (geo)dataframe 
    # Or a string - we assume it'as path to gpkg
    # We open it and we extract the geometry of the first feature
    elif type(target) in [str, gpd.GeoDataFrame, pd.DataFrame]:
        geometry = Open(target).iloc[0].geometry

    # If target is a container, we assume that there is x and y coordinates inside
    elif type(target) in [tuple, list]:

        # Here we only have a container with 1 object
        # We assume it'as sub-container like [(x, y)]
        if len(target) == 1:
            geometry = shapely.geometry.point.Point(target)

        elif len(target) > 1:
    
            # Here we have a container with 2 numbers --> it's a point
            if len(target) == 2 and (type(target[0]) == type(target[1]) == numbers.Number):
                geometry = shapely.geometry.point.Point(target)
    
            # Here we have more than 1 point point so, either a line, either a polygon
            else:
                if 'line' in geom_type.lower() or len(target) == 2:
                    print("geom_type 'pol' but the target have only 2 points so we return a line")
                    geometry = shapely.geometry.linestring.LineString(target)
                elif 'pol' in geom_type.lower():
                    geometry = shapely.geometry.polygon.Polygon(target)
        else:
            raise ValueError('empty target container')

    return geometry
    
def getMainAxes(polygon : shapely.Polygon | shapely.MultiPolygon):
    """
    Return 2 shapely.LineString objects, describing the major axes of the RGU extended outlines bounding box
    """
    
     # Get the rotated rectangle of the Extended outline
    geobox = polygon.minimum_rotated_rectangle

    # Box coords
    corners = np.array(geobox.boundary.coords)[:-1]
    
    # Split X and Y corners coordinates
    xa, xb, xc, xd = corners[:,0]
    ya, yb, yc, yd = corners[:,1]
    
    # Middle Points
    e = shapely.Point([(xa+xb)/2, (ya+yb)/2])
    f = shapely.Point([(xc+xd)/2, (yc+yd)/2])
    g = shapely.Point([(xa+xd)/2, (ya+yd)/2])
    h = shapely.Point([(xb+xc)/2, (yb+yc)/2])

    # Axis
    major_axis = shapely.LineString([e,f])
    minor_axis = shapely.LineString([g,h])

    return major_axis, minor_axis

def getGeoPointsAlongGeoLine(geoLine, step):
    """
    Return a list of geoPoints
    """

    # Extract shapely.geometry
    geoLine = getGeometry(geoLine)

    # Make array of distances to the origin
    distances = np.arange(0, geoLine.length, step)

    # Create a geoPoint for each distance
    geoPoints = [geoLine.interpolate(distance) for distance in distances]
    return geoPoints

def getGridInGeoPolygon(geoPolygon, xGap, yGap):
    """
    return a list of points regularly sampled in a geoPolygon
    """

    # Extract shapely.geometry
    geoPolygon = getGeometry(geoPolygon)

    # Create a grid of coordinates inside the polygon
    x_min, y_min, x_max, y_max = geoPolygon.bounds
    x_points = np.arange(x_min, x_max, xGap)
    y_points = np.arange(y_min, y_max, yGap)
    p_coords = np.array(np.meshgrid(x_points, y_points)).T.reshape(-1,2)

    # Create geoPoints for each coordinates inside the polygon
    geoPoints = [shapely.geometry.Point(point) for point in p_coords if geoPolygon.contains(shapely.geometry.Point(point))]
    return geoPoints

def serializeGeoLines(spine, ribLength, ribStep, ribOrientation='v'):
    """
    return a list of lines regularly sampled along a spine 
    """

    # Get the length to add from each side of the spine to have complete ribs
    r = ribLength / 2

    # Sampled the spine
    ribsOrigins = getGeoPointsAlongGeoLine(spine, ribStep)

    # If we want vertical ribs we increment the Y coordinates
    if ribOrientation.lower() == 'v':
        ribs = [getGeometry([(origin.x, origin.y + r),(origin.x, origin.y - r)]) for origin in ribsOrigins]

    # If we want horizontal ribs we increment X
    elif ribOrientation.lower() == 'h':
        ribs = [getGeometry([(origin.x + r, origin.y),(origin.x - r, origin.y)]) for origin in ribsOrigins]

    return ribs

def simplifyPolygons(polygons, rayon_buffer = 30, tolerance = 6):
    """
    Reduce the vertices of a set of polygons
    """
    # Make a copy to avoid to modify the polygons itself (weird, but when I noticed that
    # b = simplifyPolygons(a) change directly a
    new_polygons = polygons.copy(deep=True)

    # Smooth the new_polygons by first apply an erosion-dilatation
    new_polygons['geometry'] = new_polygons.apply(lambda row: row.geometry.buffer(distance=rayon_buffer).buffer(distance=-rayon_buffer),axis=1)

    # Then simplify them
    new_polygons['geometry'] = new_polygons.apply(lambda row: row.geometry.simplify(tolerance=tolerance), axis=1)

    return new_polygons

def cropLayerFromExtent(
        target_layer : str | gpd.GeoDataFrame,
        extent_feature : shapely.Polygon) -> gpd.GeoDataFrame :
    """
    Return a geodataframes with the features contained inside the extent
    """
    target_layer = Open(target_layer)
    return target_layer.sjoin(extent_feature, predicate='within', how='inner')

def getNeighbors(point, population, dist):
    z = point.buffer(dist)
    vs = population[(population.within(z)) & (~population.geom_equals(point))]
    return vs
    
def save(target, filepath, layer=None, epsg=None, driver='gpkg'):

    """
    Funnel between an object containing one or many geometric objects and a vector file .gpkg or .shp
    """

    # Build a geodataframe
    target = getGeoDf(target, epsg=epsg)

    # Check the consistency of 
    if not filepath.endswith(driver):
        if not driver.startswith('.'):
            driver = '.' + driver
        filepath += f"{driver}"

    if layer is None:
        target.to_file(filepath, layer=layer)
    else:
        target.to_file(filepath)

    if Path(filepath).exists():
        print(f"{Path(filepath).name} ok")

def show_polygons_pannel(
    polygons_layers,
    titles=None,
    linewidth=1.5,
    linecolor='red',
    map_background=cx.providers.SwissFederalGeoportal.SWISSIMAGE,
    figsize=((10,10)),
    facecolor="none",
    alpha=1,
    column_to_classify=None,
    epsg=2056,
    cmap="Reds",
    savepath=None,
    buffer_extent=30,
    share_geo_extent=False
    ):

    # TODO : be sure than we don't miss anything by taking the extent of the first result
    # Define extent which will be use for each map - from the first result
    
    # Prepare an empty figure
    fig, axes = plt.subplots(1, len(polygons_layers), figsize=figsize)

    # Prepare default titles if nothing is given
    if titles is None:
        titles = ['' for i in range(len(polygons_layers))]
    
    # Iterate on the vector layers to create a map for each
    for i, ax in enumerate(axes):
        
        # Draw a map
        polygons = polygons_layers[i]
        if column_to_classify is None:
            polygons.plot(ax=ax, linewidth=linewidth, facecolor=facecolor, alpha=alpha, cmap=cmap)
        else:
            polygons.plot(ax=ax, linewidth=linewidth, facecolor=facecolor, column=column_to_classify, alpha=alpha, cmap=cmap)

        if share_geo_extent:
            minx, miny, maxx, maxy = polygons_layers[0].dissolve().buffer(buffer_extent).total_bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
 
        ax.set_title(titles[i])

        # Add a background
        if map_background is not None:
            cx.add_basemap(ax=ax, source=map_background, crs=epsg)

    plt.tight_layout()
    
    # Write the figure in a png file
    if savepath is not None:
        if not savepath.endswith('.png'):
            savepath += '.png'
        plt.savefig(savepath)
        if Path(savepath).exists():
            print(f"{savepath} ok")

    return fig
    
def draw_geo_boundaries(geo_target, ax=None, epsg=3857, geo_target_color='black', geo_target_linestyle='dashed', geo_target_linewidth=0.5, geo_target_alpha=1, figsize=(5,5)):

    # Create an empty figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if type(geo_target) != gpd.GeoDataFrame:
        geo_target = getGeoDf([geo_target])
    
    # Draw the geo target on the figure
    geo_target.boundary.plot(ax=ax, linewidth=geo_target_linewidth, color=geo_target_color, alpha=geo_target_alpha, linestyle=geo_target_linestyle)
    return ax

def add_wmts_layer(geo_target, source=cx.providers.SwissFederalGeoportal.SWISSIMAGE, ax=None, epsg=3857, figsize=(5,5), geo_target_color='black', geo_target_linestyle='dashed', geo_target_linewidth=0.5, geo_target_alpha=1, expand_extent_x=0, expand_extent_y=0, mask_outside_geo_target=False, mask_color=None, mask_alpha=None):
    """
    Add a WMTS layer on a pyplot ax. Source is either a cx.providers object, or a URL string.
    Sources:
        - cx.providers.SwissFederalGeoportal.NationalMapColor
        - cx.providers.SwissFederalGeoportal.NationalMapGrey
        - cx.providers.SwissFederalGeoportal.SWISSIMAGE
        - ...
    Optionally expand the extent of the map on the East-West axis by a specified number of CRS units.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Draw the geo target
    draw_geo_boundaries(geo_target, ax=ax, geo_target_linewidth=geo_target_linewidth, geo_target_color=geo_target_color, geo_target_alpha=geo_target_alpha, geo_target_linestyle=geo_target_linestyle)
    
    # Adjust the extent if expand_extent_x is provided
    minx, miny, maxx, maxy = geo_target.total_bounds
    ax.set_xlim(minx - expand_extent_x, maxx + expand_extent_x)
    ax.set_ylim(miny - expand_extent_y, maxy + expand_extent_y)

    # Add the map background
    cx.add_basemap(ax=ax, source=source, crs=epsg)

    # Mask outside of the area
    if mask_outside_geo_target:
        add_white_mask_to_map(
            area_of_interest=geo_target,
            ax=ax,
            mask_color=mask_color,
            mask_alpha=mask_alpha)

    return ax

def anim_on_swiss_aerial_imagery(geo_target, epsg=3857, b_inf=None, b_sup=2024, step=5, years=None, figsize=(5,5), geo_target_color='black', geo_target_linestyle='dashed', geo_target_linewidth=0.5, geo_target_alpha=1):
    
    """
    User must give either b_inf or a list of years. By default the step is 5 years.
    """

    # Generate a list of years if required
    if years is None and b_inf is not None:
        years = range(b_inf, b_sup, step)
    
    # For each year
    for y in years:
        
        # Create an empty figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw the geo target on the figure
        draw_geo_boundaries(geo_target, ax=ax, geo_target_linewidth=geo_target_linewidth, geo_target_color=geo_target_color, geo_target_alpha=geo_target_alpha, geo_target_linestyle=geo_target_linestyle)

        # Request the wmts corresping to the year
        wms_src = f'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/' + str(year) +'/' + str(epsg) + '/{z}/{x}/{y}.jpeg'
        
        # Add it to the current figure
        ax = add_wmts_layer(source_url, ax, epsg)
        return ax

def count_overlap(gdf, epsg=2056):

    """
    Copy / paste from there 
    https://gis.stackexchange.com/questions/387773/count-overlapping-features-using-geopandas
    The post of ni1o1

    Warning : the post of sutan is fucked up. Do not use it.
    """

    #generating all of the split pieces
    import shapely
    bounds = gdf.geometry.exterior.unary_union
    new_polys = list(shapely.ops.polygonize(bounds))
    new_gdf = gpd.GeoDataFrame(geometry=new_polys)
    new_gdf['id'] = range(len(new_gdf))

    #count overlapping by sjoin between pieces representative point (point inside polygon) and the input gdf 
    new_gdf_centroid = new_gdf.copy()
    new_gdf_centroid['geometry'] = new_gdf.geometry.representative_point()
    overlapcount = gpd.sjoin(new_gdf_centroid,gdf)
    overlapcount = overlapcount.groupby(['id'])['index_right'].count().rename('overlap_score').reset_index()
    out_gdf = gpd.GeoDataFrame(pd.merge(new_gdf,overlapcount)).set_crs(epsg)
    return out_gdf

def spatial_selection(left, right, cols_to_keep=[], predicate='within'):
    """
    content : GeoDataFrame, points, lines or polygons
    container : GeoDataFrame, polygons 
    return the rows of content which are inside container
    """

    # If user just send a string for 1 column no keep and not a list
    if type(cols_to_keep) == str:
        cols_to_keep = [cols_to_keep]
    
    # Get the column names of the content geodataframe
    initial_columns = left.columns

    # Make a spatial join by keeping only content features with geometry within any container geometry
    # This add the columns and values of container beside of each feature initial attributes
    joined = left.sjoin(right, how='inner', predicate=predicate)    

    # Identification of the columns to drop : the ones which are in the joined dataframe but not in the initial
    # These has been added during the spatial join
    columns_to_drop = [colname for colname in joined.columns if colname not in initial_columns and colname not in cols_to_keep and f"{colname}_right" not in cols_to_keep]

    # Now we remove all the columns from the right layer (the container attributes)
    content_cleaned = joined.drop(columns_to_drop, axis=1)
    return content_cleaned

def get_geogrid(extent_layer, cell_width=100, cell_height=100, clip=False, epsg=2056):
    """
    Send a grid of cells based on the given extent
    extent should be a geodataframe containing one or many features
    """

    # Get a bounding box containing all the features of the extent layer
    xmin, ymin, xmax, ymax = extent_layer.total_bounds

    # Divide the cell_height in a number of rows
    rows = int(np.ceil((ymax-ymin) / cell_height))

    # Same for columns from cell_height
    cols = int(np.ceil((xmax-xmin) / cell_width))

    # Define incrementals variables
    XleftOrigin = xmin
    XrightOrigin = xmin + cell_width
    YtopOrigin = ymax
    YbottomOrigin = ymax- cell_height

    # Empty container
    polygons = []

    # For each column
    for i in range(cols):

        # Reset Ytop and Ybottom
        Ytop = YtopOrigin
        Ybottom =YbottomOrigin

        # For each line
        for j in range(rows):

            # Create a squared polygon
            polygons.append(shapely.Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 

            # Update Ys
            Ytop = Ytop - cell_height
            Ybottom = Ybottom - cell_height

        # Update Xs
        XleftOrigin = XleftOrigin + cell_width
        XrightOrigin = XrightOrigin + cell_width

    # Build a geodataframe
    geogrid = gpd.GeoDataFrame({'geometry':polygons}).set_crs(epsg=epsg)

    # Clip if needed
    if clip:
        return gpd.clip(geogrid, mask=extent_layer)

    return geogrid

def identify_features_within_area(features, areas, status_field_name='within_area'):
    """
    Add a column to say to the features layer to know if each of them is located within the areas layer
    NOTE : areas will be dissolved and threaten as one
    """

    # If we have snow in the displacement field
    if len (areas) > 0:
        
        # Dissolve all the areas
        areas = areas.dissolve()

        # Identify the vectors within the snow
        features[status_field_name] = features.apply(lambda row: row.geometry.within(areas.geometry), axis=1)

    # Else, all the vectors get False on the status_field_names status
    else:
        features[status_field_name] = False

def get_total_bounds_gdf(target_layer, epsg):
    """
    Send a geodataframe with one feature, the total extent of the target layer
    """
    
    # Numerical extent
    xmin, ymin, xmax, ymax = target_layer.geometry.total_bounds

    # Geometrical extent
    geom = shapely.box(xmin, ymin, xmax, ymax)

    # GeoDataFrame
    return gpd.GeoDataFrame({'geometry':[geom]}).set_crs(epsg=epsg)


def create_hex_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:29902"):
    """Hexagonal grid over geometry.
    See https://sabrinadchan.github.io/data-blog/building-a-hexagonal-cartogram.html
    """

    from shapely.geometry import Polygon
    import geopandas as gpd
    if bounds != None:
        xmin, ymin, xmax, ymax= bounds
    else:
        xmin, ymin, xmax, ymax= gdf.total_bounds

    unit = (xmax-xmin)/n_cells
    a = np.sin(np.pi / 3)
    cols = np.arange(np.floor(xmin), np.ceil(xmax), 3 * unit)
    rows = np.arange(np.floor(ymin) / a, np.ceil(ymax) / a, unit)

    #print (len(cols))
    hexagons = []
    for x in cols:
      for i, y in enumerate(rows):
        if (i % 2 == 0):
          x0 = x
        else:
          x0 = x + 1.5 * unit

        hexagons.append(Polygon([
          (x0, y * a),
          (x0 + unit, y * a),
          (x0 + (1.5 * unit), (y + unit) * a),
          (x0 + unit, (y + (2 * unit)) * a),
          (x0, (y + (2 * unit)) * a),
          (x0 - (0.5 * unit), (y + unit) * a),
        ]))

    grid = gpd.GeoDataFrame({'geometry': hexagons},crs=crs)
    grid["grid_area"] = grid.area
    grid = grid.reset_index().rename(columns={"index": "grid_id"})
    if overlap == True:
        cols = ['grid_id','geometry','grid_area']
        grid = grid.sjoin(gdf, how='inner').drop_duplicates('geometry')
    return grid

def rasterize(gdf, pixel_size=10, burn_value=1, out_dtype=gdal.GDT_Byte):
    """
    Rasterizes a GeoDataFrame of polygons into a numpy array.

    Parameters:
    - gdf: GeoDataFrame with polygon geometries (must be in EPSG:2056)
    - pixel_size: size of each output pixel (in map units, default 10m)
    - burn_value: value to burn into the raster (default: 1)
    - out_dtype: GDAL data type (default: gdal.GDT_Byte)

    Returns:
    - raster: 2D numpy array
    - transform: GDAL geotransform tuple
    """

    # Reproject to EPSG:2056 if necessary
    if gdf.crs.to_epsg() != 2056:
        gdf = gdf.to_crs(2056)

    # Get bounds
    minx, miny, maxx, maxy = gdf.total_bounds

    # Calculate raster dimensions
    cols = math.ceil((maxx - minx) / pixel_size)
    rows = math.ceil((maxy - miny) / pixel_size)

    # Create transform (geotransform)
    transform = (minx, pixel_size, 0, maxy, 0, -pixel_size)

    # Create in-memory raster
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create('', cols, rows, 1, out_dtype)
    mem_raster.SetGeoTransform(transform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2056)
    mem_raster.SetProjection(srs.ExportToWkt())

    # Create OGR layer
    drv = ogr.GetDriverByName('Memory')
    data_source = drv.CreateDataSource('memData')
    layer = data_source.CreateLayer('layer', srs, ogr.wkbPolygon)

    # Add features
    for _, row in gdf.iterrows():
        feature = ogr.Feature(layer.GetLayerDefn())
        geom = ogr.CreateGeometryFromWkb(row.geometry.wkb)
        feature.SetGeometry(geom)
        layer.CreateFeature(feature)

    # Rasterize all geometries with burn_value
    gdal.RasterizeLayer(mem_raster, [1], layer, burn_values=[burn_value])

    return mem_raster

def summarize_raster_values_into_vector_layer(vector_layer, raster, column_name = "raster_median"):
    """
    Apply a function for each vector feature to compute the median value of the pixels within the feature
    """
    vector_layer[column_name] = vector_layer.apply(lambda row: np.median(np.array(raster.inspectGeoPolygon(row.geometry))), axis=1)
    return vector_layer    

def add_white_mask_to_map(area_of_interest, ax, mask_color='white', mask_alpha=0.9):
    """
    Add a white mask outside of the area of interest, within the bounds of ax.

    Parameters:
    - area_of_interest: a GeoDataFrame or a shapely geometry defining the area of interest
    - ax: the matplotlib Axes object to apply the mask on

    Returns:
    - ax: the same matplotlib Axes object with the white mask applied
    """
    # Get the bounds of the current axis
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bounds = shapely.geometry.box(xlim[0], ylim[0], xlim[1], ylim[1])  # outer rectangle

    # Ensure area_of_interest is a GeoSeries
    if isinstance(area_of_interest, gpd.GeoDataFrame):
        aoi_geom = area_of_interest.unary_union
    else:
        aoi_geom = area_of_interest  # assume it's already a shapely geometry

    # Subtract the AOI from the full bounds to get the mask geometry
    mask_geom = bounds.difference(aoi_geom)

    # Create a GeoSeries for the mask and plot it
    mask = gpd.GeoSeries(mask_geom)
    mask.plot(ax=ax, color=mask_color, zorder=10, alpha=mask_alpha)

    return ax

def add_north_arrow(ax, size=0.1, location=(0.1, 0.9), color='black'):
    """
    Add a north arrow to a matplotlib Axes.

    Parameters:
    - ax: the matplotlib Axes object to add the north arrow to
    - size: size of the north arrow as a fraction of the Axes height (default: 0.1)
    - location: (x, y) tuple as fractions of Axes width/height (center of arrow)
    - color: color of the north arrow (default: 'black')

    Returns:
    - ax: the same matplotlib Axes object with the north arrow added
    """
    from matplotlib.patches import FancyArrow

    loc_x, loc_y = location
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    arrow_length = size * (ylim[1] - ylim[0])

    # Center the arrow at loc_x, loc_y
    x_pos = xlim[0] + loc_x * (xlim[1] - xlim[0])
    y_pos = ylim[0] + loc_y * (ylim[1] - ylim[0]) - arrow_length / 2

    arrow = FancyArrow(
        x_pos, y_pos, 0, arrow_length,
        width=arrow_length * 0.1,
        head_width=arrow_length * 0.2,
        head_length=arrow_length * 0.2,
        length_includes_head=True,
        color=color
    )
    ax.add_patch(arrow).set_zorder(101)
    ax.text(
        x_pos, y_pos + arrow_length + (arrow_length * 0.1), 'N',
        horizontalalignment='center', verticalalignment='bottom',
        fontsize=12, color=color
    ).set_zorder(101)
    return ax

def add_scale_bar(ax, length, location=(0.5, 0.05), linewidth=3, text_offset=0.02, units='m', color='black', fontsize=10):
    """
    Add a scale bar to a matplotlib Axes.

    Parameters:
    - ax: the matplotlib Axes object to add the scale bar to
    - length: length of the scale bar in map units (e.g., meters)
    - location: tuple (x, y) specifying the center of the scale bar as a fraction of the Axes width and height (default: (0.5, 0.05))
    - linewidth: thickness of the scale bar line (default: 3)
    - text_offset: vertical offset for the scale bar text as a fraction of the Axes height (default: 0.02)
    - units: units to display next to the scale bar length (default: 'm')
    - color: color of the scale bar and text (default: 'black')

    Returns:
    - ax: the same matplotlib Axes object with the scale bar added
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()        

    # Center the scale bar at location
    x_center = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y_pos = ylim[0] + location[1] * (ylim[1] - ylim[0])
    x_start = x_center - length / 2
    x_end = x_center + length / 2

    ax.hlines(y=y_pos, xmin=x_start, xmax=x_end, colors=color, linewidth=linewidth).set_zorder(101)
    if units == 'km':
        length = length / 1000
    ax.text(
        x_center, y_pos + text_offset * (ylim[1] - ylim[0]), f'{int(length)} {units}',
        horizontalalignment='center', verticalalignment='bottom',
        fontsize=fontsize, color=color
    ).set_zorder(101)
    return ax 

def add_scale_and_north(
    ax,
    x_loc=0.2, 
    y_loc=0.15, 
    gap_between_arrow_and_scale_bar=0.05, 
    scale_bar_length=5000, 
    scale_bar_units='m', 
    north_arrow_size=0.1, 
    north_arrow_color='black', 
    scale_bar_color='black', 
    scale_bar_fontsize=10):
    """
    Add a north arrow and a scale bar to a matplotlib Axes, vertically aligned together
    """

    ax = add_north_arrow(
        ax, 
        size=north_arrow_size, 
        location=(x_loc, y_loc + gap_between_arrow_and_scale_bar + north_arrow_size / 2), 
        color=north_arrow_color)

    ax = add_scale_bar(
        ax, 
        length=scale_bar_length, 
        location=(x_loc, y_loc), 
        units=scale_bar_units, 
        color=scale_bar_color, 
        fontsize=scale_bar_fontsize)

    return ax

def count_coalescent_systems(gdf):
    """
    Count coalescent systems and classify them as single vs multi.
    
    Returns:
        total_systems (int): Total number of systems
        single_systems (int): Systems from exactly one original polygon
        multi_systems (int): Systems formed by merging >1 polygon
    """
    # Assign a dummy column for dissolve
    gdf["_tmp"] = 1
    
    # Dissolve all into one union, then explode into separate systems
    dissolved = gdf.dissolve(by="_tmp")
    exploded = dissolved.explode(index_parts=False)
    
    # Spatial join to see how many original polygons fall in each system
    exploded = exploded.reset_index(drop=True)
    joined = gpd.sjoin(gdf.drop(columns="_tmp"), exploded, predicate="intersects")
    
    # Count how many original polygons in each exploded system
    counts = joined.groupby("index_right").size()
    
    total_systems = len(counts)
    single_systems = (counts == 1).sum()
    multi_systems = (counts > 1).sum()
    
    return total_systems, single_systems, multi_systems

def get_cell_surf_covered_by_hue_vals(gdf, grid, hue):
    """
    For each grid cell, compute the percentage of its area covered by each value of `hue` in gdf.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygons and a categorical column `hue`.
    grid : geopandas.GeoDataFrame
        GeoDataFrame containing the grid cells.
    hue : str
        The name of the column in gdf with the categories.
    
    Returns
    -------
    geopandas.GeoDataFrame
        Original grid with additional columns for each hue value (percent coverage).
    """
    # Ensure CRS match
    if gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    
    # Compute intersection
    intersections = gpd.overlay(grid.reset_index(), gdf[[hue, 'geometry']], how='intersection')
    
    # Compute area of intersection and original grid cells
    intersections["area_intersection"] = intersections.geometry.area
    grid_area = grid.geometry.area
    intersections = intersections.rename(columns={"index": "grid_index"})
    
    # Compute percentage of cell covered by each hue
    intersections["pct"] = intersections.apply(lambda row: row["area_intersection"] / grid_area[row["grid_index"]], axis=1)
    
    # Group by grid_index and hue
    coverage = intersections.groupby(["grid_index", hue])["pct"].sum().unstack(fill_value=0)
    
    # Merge back into grid
    result = grid.join(coverage, how="left").fillna(0)
    
    return result