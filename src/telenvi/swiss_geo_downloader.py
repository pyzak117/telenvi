import os
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from telenvi import raster_tools as rt
from telenvi import vector_tools as vt

adminch_metadata_path = Path(Path(__file__).with_name('ressources'), 'metadata_admin-ch_valais.gpkg')
swimage_metadata_layer = gpd.read_file(adminch_metadata_path, layer='cleaned-metadata-swiss-images-valais-detailled')
sw3d_metadata_layer = gpd.read_file(adminch_metadata_path, layer='metadata_swiss-surface-3D_valais')

def get_swimages(
    dest_repo,
    study_area_source,
    study_area_layername = None,
    metadata_source=None,
    metadata_layername=None,
    acq_year=None,
    verbose=True):

    """
    dest_repo : string or Path, where the images will be downloaded
    study_area_source : string or geodataframe in 2056
    study_area_layername : string if study_area refer to a geopackage file with many layers
    metadata_source : string or geodataframe in 2056
    metadata_layername : string if study_area refer to a geopackage file with many layers
    acq_year : int [2017;2024], if None, take all the available images on the server
    """

    # Define a default metadata file location :
    # the one for the swissimages in valais
    if metadata_source is None:
        metadata_source = swimage_metadata_layer
    
    # Open the metadata file
    metadata_df = vt.Open(metadata_source, metadata_layername, to_crs_epsg=2056)

    # Open the study area file
    study_area_df = vt.Open(study_area_source, study_area_layername, set_crs_epsg=2056)

    # Intersect both
    target_tiles = metadata_df.sjoin(study_area_df, how='inner').drop_duplicates('url')

    # Select from attributes if needed
    if acq_year is not None:
        target_tiles = target_tiles[target_tiles.acq_year == acq_year]

    if len(target_tiles) == 0:
        print('study_area not intersecting the metadata_file')
        return target_tiles
    
    # Build standardized names
    def write_target_output_name(row):
        adminch_name = row.url.split('/')[-1][:-4]
        thib_phd_name = f"{row.pseudo_d}_{row.delta_days}_{adminch_name}.tif"
        return thib_phd_name

    # Build destination filename through the standards defined just before
    target_tiles['dest_filename'] = target_tiles.apply(lambda row: write_target_output_name(row),axis=1)

    # Build destination complete filepath from the destination repo argument
    target_tiles['dest_filepath'] = target_tiles.apply(lambda row: Path(dest_repo, row.dest_filename), axis=1)

    # For each tile
    for tile in target_tiles.iloc:

        # Check if the destination path is not already existing
        if not tile.dest_filepath.exists():

            # Download the file
            dl_command = f"wget {tile.url} --output-document {tile.dest_filepath}"
            print(f'download {tile.url}')
            os.system(dl_command)
        
        else:
            if verbose:
                print(f"{tile['dest_filename']} already existing in the destination dir.")

    return target_tiles

def get_ss3d(
    dest_repo,
    study_area_source,
    study_area_layername = None,
    metadata_source=None,
    metadata_layername=None,
    verbose=True,
    blank=False):

    """
    study_area : string or geodataframe in 2056
    study_area_layername : string if study_area refer to a geopackage file with many layers
    metadata_source : string or geodataframe in 2056
    metadata_layername : string if study_area refer to a geopackage file with many layers
    dest_repo  : string or Path
    """

    # Define a default metadata file location :
    # the one for the swissimages in valais
    if metadata_source is None:
        metadata_source = sw3d_metadata_layer
    
    # Open the metadata file
    metadata_df = vt.Open(metadata_source, metadata_layername, to_crs_epsg=2056)

    # Open the study area file
    study_area_df = vt.Open(study_area_source, study_area_layername, set_crs_epsg=2056)

    # Intersect both
    target_tiles = metadata_df.sjoin(study_area_df, how='inner').drop_duplicates('url')

    if len(target_tiles) == 0:
        print('study_area not intersecting the metadata_file')
        return target_tiles
    
    # Build standardized names
    def write_target_output_name(row):
        adminch_name = row.url.split('/')[-1][:-4]
        thib_phd_name = f"{row.rtype}_{row.year}_{adminch_name}.tif"
        return thib_phd_name

    # Build destination filename through the standards defined just before
    target_tiles['dest_filename'] = target_tiles.apply(lambda row: write_target_output_name(row),axis=1)

    # Build destination complete filepath from the destination repo argument
    target_tiles['dest_filepath'] = target_tiles.apply(lambda row: Path(dest_repo, row.dest_filename), axis=1)

    # For each tile
    for tile in target_tiles.iloc:

        # Check if the destination path is not already existing
        if not tile.dest_filepath.exists():

            # Download the file
            dl_command = f"wget {tile.url} --output-document {tile.dest_filepath}"
            if blank:
                print(f'blank {tile.url}')
            else:
                print(f'download {tile.url}')
                os.system(dl_command)
        
        else:
            if verbose:
                print(f"{tile['dest_filename']} already existing in the destination dir.")

    return target_tiles