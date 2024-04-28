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
import telenvi.node as node
import telenvi.segment as segment
import telenvi.multi_segment as multi_segment
import telenvi.geo_network as geo_network
import numbers

def save(target, filepath, layername=None, epsg=None, driver='gpkg'):

    # Build a geodataframe
    target = getGeoDf(target, epsg=epsg)

    # Check the consistency of 
    if not filepath.endswith(driver):
        if not driver.startswith('.'):
            driver = '.' + driver
        filepath += f"{driver}"

    if layername is None:
        target.to_file(filepath, layer=layername)
    else:
        target.to_file(filepath)

    if Path(filepath).exists():
        print(f"{Path(filepath).name} ok")

def Open(layer_source, layer_name=None, target_epsg=None):
    """
    Return a GeoDataFrame from vector file (shapefile or geopackage)
    """

    if type(layer_source) == gpd.GeoDataFrame:
        return layer_source

    if str(layer_source).lower().endswith('.shp'):
        layer = gpd.read_file(layer_source)
    
    elif str(layer_source).lower().endswith('.gpkg'):
        if layer_name is None:
            layer = gpd.read_file(layer_source)
        else:
            layer = gpd.read_file(layer_source, layer=layer_name)

    if target_epsg is not None:
        layer = layer.to_crs(target_epsg)

    return layer

def share_same_geotype(targets):
    types_of_objects = [type(t) for t in targets]
    uniques_types = pd.Series(types_of_objects).drop_duplicates().tolist()
    return len(uniques_types) == 1

def getGeoSerie(targets):
    """
    Send a geoserie from a bunch of geometric objects
    """
    shapely_objects = [getGeoThing(t) for t in targets]
    assert share_same_geotype(shapely_objects), "Targets must have the same geotype"
    return gpd.GeoSeries(shapely_objects)

def getGeoDf(targets, epsg=None):
    if type(targets) == geo_network.GeoNetwork:
        targets = targets.nodes
    shapely_objects = [getGeoThing(t) for t in targets]
    assert share_same_geotype(shapely_objects), "Targets must have the same geotype"
    gdf = gpd.GeoDataFrame(shapely_objects, columns=['geometry'])
    if epsg is not None:
        gdf = gdf.set_crs(epsg=2056)
    return gdf

def getGeoThing(target = None, x = None, y = None):
    """
    Return a shapely.geometry object from different cases 
    """

    # We didn't get target
    if target is None:
        assert x is not None and y is not None, 'input arguments invalids'
        target = (x,y)
    
    # It's already a shapely.geometry (type could be shapely.Polygon, shapely.Point, shapely.LineString...)
    try :
        _ = target.coords
        return target
    except AttributeError:
        pass

    if type(target) in [pd.Series, gpd.GeoSeries]:
        geoThing = target.geometry

    elif type(target) == node.Node:
        geoThing = target.shape

    elif type(target) in [gpd.GeoDataFrame, pd.DataFrame]:
        geoThing = target.iloc[0].geometry

    elif type(target) == str:
        geoThing = gpd.read_file(target).iloc[0].geometry

    elif type(target) in [tuple, list]:
        if len(target) == 1:
            geoThing = shapely.geometry.point.Point(target)

        elif len(target) == 2:

            # We have a tuple of numbers : a point
            if type(type(target[0]) == type(target[1]) == numbers.Number):
                x, y = target
                geoThing = shapely.geometry.point.Point(x, y)

            # We have a tuple of tuples : line
            else:
                geoThing = shapely.geometry.linestring.LineString(target)

        elif len(target) > 2:
            geoThing = shapely.geometry.polygon.Polygon(target)

    return geoThing
    
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
    geoLine = getGeoThing(geoLine)

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
    geoPolygon = getGeoThing(geoPolygon)

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
        ribs = [getGeoThing([(origin.x, origin.y + r),(origin.x, origin.y - r)]) for origin in ribsOrigins]

    # If we want horizontal ribs we increment X
    elif ribOrientation.lower() == 'h':
        ribs = [getGeoThing([(origin.x + r, origin.y),(origin.x - r, origin.y)]) for origin in ribsOrigins]

    return ribs

def simplifyPolygons(polygons, rayon_buffer = 30, tolerance = 6):

    # Smooth the polygons by first apply an erosion-dilatation
    polygons['geometry'] = polygons.apply(lambda row: row.geometry.buffer(distance=rayon_buffer).buffer(distance=-rayon_buffer),axis=1)

    # Then simplify them
    polygons['geometry'] = polygons.apply(lambda row: row.geometry.simplify(tolerance=tolerance), axis=1)

    return polygons

def cropLayerFromExtent(
        target_layer : str | gpd.GeoDataFrame,
        extent_feature : shapely.Polygon) -> gpd.GeoDataFrame :
    """
    Return a geodataframes with the row contained inside the extent
    """
    target_layer = Open(target_layer)
    return target_layer[target_layer.within(extent_feature)]