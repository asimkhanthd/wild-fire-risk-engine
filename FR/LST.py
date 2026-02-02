"""
Sentinel-3 SLSTR Land Surface Temperature (LST) Processor.

This module provides functions to:
- Read LST data from Sentinel-3 .SEN3.zip files
- Resample swath data to a regular grid
- Clip data using a shapefile
- Generate PNG visualizations

Author: User
Date: 2025
"""

import os
os.environ['GDAL_DATA'] = r'C:\Users\alvar\anaconda3\envs\storcito\Library\share\gdal'

import numpy as np
import zipfile
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Union

import netCDF4 as nc
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
import geopandas as gpd
import matplotlib.pyplot as plt
from pyresample import geometry, kd_tree


# =============================================================================
# STEP 1: LST DATA EXTRACTION FROM SENTINEL-3
# =============================================================================

def read_sentinel3_lst(zip_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read LST data and coordinates from a Sentinel-3 .SEN3.zip file.

    Parameters
    ----------
    zip_path : str or Path
        Path to the .SEN3.zip file

    Returns
    -------
    tuple
        (lst_data, lat_data, lon_data) as numpy arrays float32/float64
    """
    zip_path = Path(zip_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)

        # Find .SEN3 folder
        sen3_folder = next(f for f in tmpdir.iterdir() if f.suffix == '.SEN3')

        # Read LST data
        with nc.Dataset(sen3_folder / 'LST_in.nc', 'r') as ds_lst:
            lst_data = np.ma.filled(ds_lst.variables['LST'][:], np.nan).astype(np.float32)

        # Read coordinates
        with nc.Dataset(sen3_folder / 'geodetic_in.nc', 'r') as ds_geo:
            lat_data = np.ma.filled(ds_geo.variables['latitude_in'][:], np.nan).astype(np.float64)
            lon_data = np.ma.filled(ds_geo.variables['longitude_in'][:], np.nan).astype(np.float64)

    return lst_data, lat_data, lon_data


# =============================================================================
# STEP 2: SWATH TO REGULAR GRID CONVERSION
# =============================================================================

def resample_swath_to_grid(
    lst_data: np.ndarray,
    lat_data: np.ndarray,
    lon_data: np.ndarray,
    resolution: float = 0.01,
    radius_of_influence: float = 50000
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Resample satellite swath data to a regular grid using pyresample.

    Parameters
    ----------
    lst_data : np.ndarray
        LST data in swath format
    lat_data : np.ndarray
        Per-pixel latitudes
    lon_data : np.ndarray
        Per-pixel longitudes
    resolution : float
        Resolution in degrees (default: 0.01 ~ 1km)
    radius_of_influence : float
        Radius of influence in meters for resampling (default: 50000)

    Returns
    -------
    tuple
        (lst_gridded, bounds) where bounds = (min_lon, min_lat, max_lon, max_lat)
    """
    # Create valid data mask
    valid_mask = ~np.isnan(lst_data) & ~np.isnan(lat_data) & ~np.isnan(lon_data)

    print(f'Valid points for resampling: {np.sum(valid_mask):,}')

    if np.sum(valid_mask) == 0:
        raise ValueError("No valid points for resampling")

    # Calculate bounds
    min_lon = float(np.nanmin(lon_data[valid_mask]))
    max_lon = float(np.nanmax(lon_data[valid_mask]))
    min_lat = float(np.nanmin(lat_data[valid_mask]))
    max_lat = float(np.nanmax(lat_data[valid_mask]))
    bounds = (min_lon, min_lat, max_lon, max_lat)

    print(f'Bounds: lon=[{min_lon:.4f}, {max_lon:.4f}], lat=[{min_lat:.4f}, {max_lat:.4f}]')

    # Calculate target grid dimensions
    width = int(np.ceil((max_lon - min_lon) / resolution))
    height = int(np.ceil((max_lat - min_lat) / resolution))

    print(f'Resampling to {height}x{width} pixel grid...')

    # Define source swath
    swath_def = geometry.SwathDefinition(lons=lon_data, lats=lat_data)

    # Define target area (regular grid)
    area_def = geometry.AreaDefinition(
        area_id='lst_grid',
        description='LST regular grid',
        proj_id='longlat',
        projection='EPSG:4326',
        width=width,
        height=height,
        area_extent=(min_lon, min_lat, max_lon, max_lat)
    )

    # Resample using nearest neighbor with kd_tree
    lst_gridded = kd_tree.resample_nearest(
        swath_def,
        lst_data,
        area_def,
        radius_of_influence=radius_of_influence,
        fill_value=np.nan # type: ignore
    ).astype(np.float32)  # type: ignore

    return lst_gridded, bounds


# =============================================================================
# STEP 3: GEOTIFF EXPORT
# =============================================================================

def save_geotiff(
    data: np.ndarray,
    output_path: Union[str, Path],
    bounds: Tuple[float, float, float, float],
    crs: str = 'EPSG:4326',
    nodata: float = -9999
) -> Path:
    """
    Save an array as GeoTIFF.

    Parameters
    ----------
    data : np.ndarray
        2D data array to save
    output_path : str or Path
        Output path
    bounds : tuple
        (min_lon, min_lat, max_lon, max_lat)
    crs : str
        Coordinate reference system (default: EPSG:4326)
    nodata : float
        Value for invalid data

    Returns
    -------
    Path
        Path to saved file
    """
    output_path = Path(output_path)
    min_lon, min_lat, max_lon, max_lat = bounds

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, data.shape[1], data.shape[0])

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data, 1)

    print(f'GeoTIFF saved: {output_path}')
    return output_path


# =============================================================================
# STEP 4: SPATIAL CLIPPING OF LST
# =============================================================================

def load_shapefile(shp_path: Union[str, Path], target_crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """
    Load a shapefile and reproject to target CRS.

    Parameters
    ----------
    shp_path : str or Path
        Path to shapefile
    target_crs : str
        Target CRS (default: EPSG:4326)

    Returns
    -------
    GeoDataFrame
        Loaded and reprojected shapefile
    """
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(target_crs)
    return gdf


def clip_array_with_shapefile(
    data: np.ndarray,
    bounds: Tuple[float, float, float, float],
    gdf: gpd.GeoDataFrame,
    crs: str = 'EPSG:4326'
) -> Tuple[np.ndarray, Tuple[float, float, float, float], dict]:
    """
    Clip an array using a shapefile, adjusting extent to shapefile bounds.

    Parameters
    ----------
    data : np.ndarray
        2D array with data to clip
    bounds : tuple
        (min_lon, min_lat, max_lon, max_lat) of input array
    gdf : GeoDataFrame
        Shapefile for clipping
    crs : str
        Coordinate reference system (default: EPSG:4326)

    Returns
    -------
    tuple
        (clipped_data, new_bounds, metadata)
    """
    geometries = gdf.geometry.values
    shp_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    shp_minx, shp_miny, shp_maxx, shp_maxy = shp_bounds

    min_lon, min_lat, max_lon, max_lat = bounds
    src_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, data.shape[1], data.shape[0])

    # Calculate resolution
    res_x = src_transform.a  # type: ignore
    res_y = abs(src_transform.e)  # type: ignore

    # New dimensions based on shapefile bounds
    new_width = int(np.ceil((shp_maxx - shp_minx) / res_x))
    new_height = int(np.ceil((shp_maxy - shp_miny) / res_y))

    # New transform
    new_transform = from_bounds(shp_minx, shp_miny, shp_maxx, shp_maxy, new_width, new_height)

    # Destination array
    dest_data = np.full((new_height, new_width), np.nan, dtype=np.float32)

    # Reproject to new extent
    reproject(
        source=data,
        destination=dest_data,
        src_transform=src_transform,
        src_crs=crs,
        dst_transform=new_transform,
        dst_crs=crs,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan
    )

    # Apply shapefile mask
    shp_mask = geometry_mask(geometries, out_shape=(new_height, new_width),
                              transform=new_transform, invert=True)
    dest_data[~shp_mask] = np.nan

    # Build metadata
    clipped_meta = {
        'driver': 'GTiff',
        'height': new_height,
        'width': new_width,
        'count': 1,
        'dtype': np.float32,
        'crs': crs,
        'transform': new_transform,
        'nodata': np.nan
    }

    new_bounds = (shp_minx, shp_miny, shp_maxx, shp_maxy)

    valid_count = np.count_nonzero(~np.isnan(dest_data))
    print(f'Clipped raster: {valid_count:,} valid pixels')

    return dest_data, new_bounds, clipped_meta


# =============================================================================
# STEP 5: FINAL LST VISUALIZATION
# =============================================================================

def create_lst_png(
    data: np.ndarray,
    output_path: Union[str, Path],
    bounds: Tuple[float, float, float, float],
    title: str = 'Land Surface Temperature (°C)',
    gdf: Optional[gpd.GeoDataFrame] = None,
    vmin: float = 0,
    vmax: float = 50,
    convert_to_celsius: bool = True
) -> Path:
    """
    Create a PNG visualization of LST data.

    Parameters
    ----------
    data : np.ndarray
        LST data (in Kelvin if convert_to_celsius=True)
    output_path : str or Path
        Output path
    bounds : tuple
        (min_lon, min_lat, max_lon, max_lat) or (left, bottom, right, top)
    title : str
        Plot title
    gdf : GeoDataFrame, optional
        Shapefile for drawing boundary
    vmin, vmax : float
        Color scale limits
    convert_to_celsius : bool
        If True, converts from Kelvin to Celsius

    Returns
    -------
    Path
        Path to saved file
    """
    output_path = Path(output_path)

    # Convert to Celsius if needed
    if convert_to_celsius:
        data_plot = data - 273.15
    else:
        data_plot = data

    fig, ax = plt.subplots(figsize=(10, 10))

    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    im = ax.imshow(data_plot, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, extent=extent)

    if gdf is not None:
        gdf.boundary.plot(ax=ax, color='black', linewidth=1)

    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label='Temperature (°C)', shrink=0.7)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'PNG saved: {output_path}')
    return output_path


# =============================================================================
# FULL PIPELINE: SENTINEL-3 → CLIPPED FINAL LST
# =============================================================================

def process_sentinel3_lst(
    zip_path: Union[str, Path],
    shp_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    resolution: float = 0.01,
    generate_png: bool = True,
    save_tif: bool = True
) -> Tuple[np.ndarray, Tuple[float, float, float, float], dict]:
    """
    Process a complete Sentinel-3 LST file: read, resample, clip and return clipped array.

    Parameters
    ----------
    zip_path : str or Path
        Path to .SEN3.zip file
    shp_path : str or Path
        Path to shapefile for clipping
    output_dir : str or Path, optional
        Output directory (default: same directory as zip_path)
    resolution : float
        Resolution in degrees (default: 0.01 ~ 1km)
    generate_png : bool
        If True, generates PNG visualizations
    save_tif : bool
        If True, saves clipped GeoTIFF (_clip)

    Returns
    -------
    tuple
        (clipped_data, clipped_bounds, clipped_meta) - Clipped array, bounds and metadata
    """
    zip_path = Path(zip_path)
    shp_path = Path(shp_path)

    if output_dir is None:
        output_dir = zip_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    base_name = zip_path.stem.replace('.SEN3', '')
    output_tif_clip = output_dir / f'{base_name}_LST_clip.tif'
    output_png_clip = output_dir / f'{base_name}_LST_clip.png'

    # 1. Read data
    print("\n" + "="*60)
    print("1. READING SENTINEL-3 DATA")
    print("="*60)
    lst_data, lat_data, lon_data = read_sentinel3_lst(zip_path)

    # 2. Resample to regular grid
    print("\n" + "="*60)
    print("2. RESAMPLING TO REGULAR GRID")
    print("="*60)
    lst_gridded, bounds = resample_swath_to_grid(
        lst_data, lat_data, lon_data, resolution=resolution
    )

    # 3. Load shapefile and clip array
    print("\n" + "="*60)
    print("3. CLIPPING WITH SHAPEFILE")
    print("="*60)
    gdf = load_shapefile(shp_path)
    clipped_data, clipped_bounds, clipped_meta = clip_array_with_shapefile(
        lst_gridded, bounds, gdf
    )

    # 4. Save clipped GeoTIFF (optional)
    if save_tif:
        print("\n" + "="*60)
        print("4. SAVING CLIPPED GEOTIFF")
        print("="*60)
        save_geotiff(clipped_data, output_tif_clip, clipped_bounds)

    # 5. Generate PNG (optional)
    if generate_png:
        print("\n" + "="*60)
        print("5. GENERATING VISUALIZATION")
        print("="*60)
        create_lst_png(
            clipped_data, output_png_clip, clipped_bounds,
            title='Clipped LST (°C)', gdf=gdf
        )

    print("\n" + "="*60)
    print("PROCESSING COMPLETED")
    print("="*60)

    return clipped_data, clipped_bounds, clipped_meta


# =============================================================================
# STEP 6: TEMPORAL ANALYSIS - MULTI-LST AVERAGE
# =============================================================================

def calculate_tif_mean(tif_paths: list, output_path: Union[str, Path]) -> np.ndarray:
    """
    Calculate the temporal mean of multiple LST GeoTIFF files.

    Parameters
    ----------
    tif_paths : list
        List of paths to GeoTIFF files
    output_path : str or Path
        Output path for the mean raster

    Returns
    -------
    np.ndarray
        Mean array
    """
    with rasterio.open(tif_paths[0]) as src:
        meta = src.meta.copy()
        acum = np.zeros(src.shape, dtype=np.float64)
        count = np.zeros(src.shape, dtype=np.int32)

    for path in tif_paths:
        with rasterio.open(path) as src:
            data = src.read(1)
            valid = ~np.isnan(data)
            acum[valid] += data[valid]
            count[valid] += 1

    mean = np.where(count > 0, acum / count, np.nan).astype(np.float32)

    meta.update(dtype=rasterio.float32, nodata=-9999)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mean, 1)

    return mean


def reclassify_lst(data: np.ndarray, n_intervals: int = 5) -> np.ndarray:
    """
    Reclassify LST values into discrete categories.

    Parameters
    ----------
    data : np.ndarray
        Input LST data
    n_intervals : int
        Number of classification intervals (default: 5)

    Returns
    -------
    np.ndarray
        Classified array as int8
    """
    bins = np.linspace(np.nanmin(data), np.nanmax(data), num=n_intervals + 1)
    classified = np.digitize(data, bins)
    return classified.astype(np.int8)


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Usage example
    zip_path = Path(r'C:\Users\alvar\Desktop\primeros test\mini_test\S3A_SL_2_LST____20250804T110923_20250804T111223_20250805T222849_0179_129_037_2160_PS1_O_NT_004.SEN3.zip')
    shp_path = Path(r'C:\Users\alvar\Desktop\Comunidade_Autonoma (1)\Comunidade_Autonoma_IGN.shp')

    clipped_data, clipped_bounds, clipped_meta = process_sentinel3_lst(
        zip_path, shp_path, generate_png=True
    )
