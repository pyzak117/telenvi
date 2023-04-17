module_description = """
--- telenvi.vector_tools ---
Functions to process vector geo data through geopandas
"""

# Geo libraries
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd

def getGeoThing(target):
    """
    Return a shapely.geometry object from different cases 
    """

    try :
        _ = target.coords
        return target
    except AttributeError:
        pass

    if type(target) in [pd.Series, gpd.GeoSeries]:
        geoThing = target.geometry

    elif type(target) in [gpd.GeoDataFrame, pd.DataFrame]:
        geoThing = target.iloc[0].geometry

    elif type(target) == str:
        geoThing = gpd.read_file(target).iloc[0].geometry

    elif type(target) in [tuple, list]:
        if len(target) == 1:
            geoThing = shapely.geometry.point.Point(target)

        if len(target) == 2:
            geoThing = shapely.geometry.linestring.LineString(target)

        if len(target) > 2:
            geoThing = shapely.geometry.polygon.Polygon(target)

    return geoThing

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
