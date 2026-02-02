# -*- coding: utf-8 -*-
"""
TWI calculation per region using WhiteboxTools + rasterio (no QGIS/SAGA).

Optimized version: improved readability and performance.
- Input: Regional DTM (100 m) and Slope (radians, 30 m).
- Output: flow (cells), sca (m), twi (dimensionless) on the DTM grid (100 m).
"""

from __future__ import annotations

import csv
import tempfile
import rasterio
import whitebox
import numpy as np

from pathlib import Path
from functools import partial
from typing import Any, Sequence
from concurrent.futures import ProcessPoolExecutor

from numpy.typing import NDArray
from dataclasses import dataclass, field
from rasterio.warp import Resampling, reproject

# =====================================================================
# PUBLIC API
# =====================================================================

__all__ = [
    # Configuration
    "ConfigTWI",
    # Data classes
    "RasterData",
    "RegionResult",
    # Main functions
    "run_twi_processing",
    "process_region",
    "find_dtm_files",
    "compute_twi_from_files",
    # Utilities
    "load_raster",
    "save_raster",
    "compute_flow_accumulation",
    "compute_twi",
]

# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass(frozen=True)
class ConfigTWI:
    """Configuration parameters for TWI calculation.

    This immutable dataclass holds all configuration options for TWI processing,
    including paths, numeric thresholds, parallelization settings, and output options.

    Attributes:
        base_path: Root directory containing regional data folders.
        subdirs: Tuple of subdirectory names to process. If None, auto-detects
            all folders in base_path.
        dtm_pattern: Glob pattern for finding DTM files (e.g., "dtm_*.tif").
        slope_prefix: Prefix for slope files (e.g., "slope_" -> "slope_vigo.tif").
        overwrite: If True, regenerate outputs even if they exist.
        nodata_out: NoData value for output rasters.
        resample_method: Resampling method for slope reprojection.
        eps_tan: Minimum value for tan(slope) to prevent division by zero.
        eps_sca: Minimum value for SCA to prevent log(0).
        twi_clip: Optional (min, max) tuple to clip TWI values.
        single_region: If set, process only this specific region.
        max_workers: Number of parallel processes for multiprocessing.
        verbose: If True, print progress messages to stdout.

    Example:
        >>> config = ConfigTWI(
        ...     base_path=Path("/data/twi"),
        ...     max_workers=8,
        ...     verbose=False
        ... )
        >>> config.directories  # Auto-detected folders
        [Path('/data/twi/region1'), Path('/data/twi/region2')]
    """

    base_path: Path = Path(r"C:\Users\rachs\Desktop\RAQUEL\FOREST FIRE RISK MAP\Forest Fire Risk Map Gal\TWI")
    subdirs: tuple[str, ...] | None = None  # If None, auto-detect folders in base_path

    dtm_pattern: str = "dtm_*.tif"
    slope_prefix: str = "slope_"

    overwrite: bool = False
    nodata_out: float = -9999.0
    resample_method: Resampling = Resampling.bilinear

    # Numeric thresholds
    eps_tan: float = 1e-12      # Minimum for tan(beta)
    eps_sca: float = 1e-6       # Minimum for SCA
    twi_clip: tuple[float, float] | None = None  # E.g.: (-5, 20)

    # Filters
    single_region: str | None = None  # E.g.: "vigo" to process only one

    # Parallelization
    max_workers: int = 4  # Number of parallel processes

    # Logging
    verbose: bool = True  # Show progress messages

    @property
    def directories(self) -> list[Path]:
        if self.subdirs is not None:
            return [self.base_path / subdir for subdir in self.subdirs]
        # Auto-detect folders in base_path
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")
        return sorted([p for p in self.base_path.iterdir() if p.is_dir()])

    @property
    def log_csv_path(self) -> Path:
        return self.base_path / "twi_log.csv"


# =====================================================================
# TYPES
# =====================================================================

ArrayF32 = NDArray[np.float32]


# =====================================================================
# LOGGING
# =====================================================================

# Global variable to control verbosity (updated from config)
_verbose: bool = True


def _log(msg: str) -> None:
    """Print message only if verbose is enabled."""
    if _verbose:
        print(msg)


def _set_verbose(verbose: bool) -> None:
    """Set global verbosity level."""
    global _verbose
    _verbose = verbose


# =====================================================================
# WHITEBOX INITIALIZATION
# =====================================================================

_wbt = whitebox.WhiteboxTools()
_wbt.set_verbose_mode(False)


# =====================================================================
# I/O FUNCTIONS
# =====================================================================

@dataclass
class RasterData:
    """Container for loaded raster data and metadata.

    This dataclass encapsulates all information loaded from a raster file,
    including the pixel values, geographic metadata, and nodata handling.

    Attributes:
        array: 2D array of pixel values as float32, with nodata converted to NaN.
        profile: Rasterio profile dictionary with file metadata.
        nodata: Original nodata value from the file, or None if not defined.
        nodata_mask: Boolean mask where True indicates nodata pixels.
        shape: Tuple of (height, width) in pixels.
        transform: Affine transformation matrix for georeferencing.
        crs: Coordinate Reference System of the raster.

    Example:
        >>> data = load_raster(Path("dem.tif"))
        >>> data.shape
        (1000, 1200)
        >>> data.array[data.nodata_mask].size  # Number of nodata pixels
        150
    """
    array: ArrayF32
    profile: dict
    nodata: float | None
    nodata_mask: NDArray[np.bool_]
    shape: tuple[int, int]
    transform: Any  # rasterio.Affine
    crs: Any  # rasterio.crs.CRS


def load_raster(path: Path) -> RasterData:
    """Load a raster file and normalize NODATA values to NaN.

    Reads the first band of a raster file, converts it to float32, and replaces
    nodata values with NaN for consistent numerical handling.

    Args:
        path: Path to the raster file (GeoTIFF or any GDAL-supported format).

    Returns:
        RasterData object containing the array, profile, nodata mask, and
        geographic metadata (transform, CRS).

    Raises:
        rasterio.errors.RasterioIOError: If the file cannot be opened.

    Example:
        >>> dem_data = load_raster(Path("dtm_vigo.tif"))
        >>> dem_data.array.shape
        (500, 600)
        >>> np.nanmin(dem_data.array)
        12.5
    """
    with rasterio.open(path) as src:
        arr_raw = src.read(1)
        arr = arr_raw.astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    if nodata is not None:
        nodata_mask = (arr_raw == nodata) | np.isclose(arr_raw, nodata, rtol=1e-5)
        arr[nodata_mask] = np.nan
    else:
        nodata_mask = np.zeros(shape, dtype=bool)

    return RasterData(
        array=arr,
        profile=profile,
        nodata=nodata,
        nodata_mask=nodata_mask,
        shape=shape,
        transform=transform,
        crs=crs
    )


def save_raster(
    arr: ArrayF32,
    out_path: Path,
    reference_path: Path,
    nodata: float = -9999,
    compress: str = "LZW"
) -> None:
    """Save a numpy array as a GeoTIFF file.

    Writes the array to a GeoTIFF, copying the georeference (CRS, transform,
    dimensions) from a reference raster. NaN values are replaced with the
    specified nodata value.

    Args:
        arr: 2D numpy array to save.
        out_path: Destination path for the output GeoTIFF.
        reference_path: Path to a reference raster for copying georeference.
        nodata: Value to use for nodata pixels in the output. Defaults to -9999.
        compress: Compression algorithm ("LZW", "DEFLATE", "ZSTD"). Defaults to "LZW".

    Raises:
        rasterio.errors.RasterioIOError: If the reference file cannot be opened
            or the output cannot be written.

    Example:
        >>> twi = compute_twi(sca, slope)
        >>> save_raster(twi, Path("twi_output.tif"), Path("dtm_reference.tif"))
    """
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()

    profile.update(dtype=rasterio.float32, compress=compress, nodata=nodata)

    # Replace NaN with nodata
    out_arr = np.where(np.isnan(arr), nodata, arr)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr.astype(np.float32), 1)


def _save_temp_raster(arr: ArrayF32, path: Path, profile: dict, nodata: float) -> None:
    """Save temporary raster using profile directly (no reference file needed)."""
    prof = profile.copy()
    prof.update(dtype=rasterio.float32, nodata=nodata)

    out_arr = np.where(np.isnan(arr), nodata, arr)

    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out_arr.astype(np.float32), 1)


def resample_to_grid(
    src_arr: ArrayF32,
    src_transform,
    src_crs,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    method: Resampling = Resampling.bilinear
) -> ArrayF32:
    """Reproject/resample an array to the destination grid."""
    dst_arr = np.empty(dst_shape, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=method
    )
    return dst_arr


# =====================================================================
# HYDROLOGICAL CALCULATIONS
# =====================================================================

def compute_flow_accumulation(dem: ArrayF32, profile: dict, nodata: float = -9999.0) -> ArrayF32:
    """Compute D8 flow accumulation from a Digital Elevation Model.

    Uses WhiteboxTools to calculate flow accumulation, which represents
    the number of upstream cells draining through each cell. The algorithm:
    1. Fills depressions (sinks) in the DEM to ensure continuous flow
    2. Computes D8 flow direction (steepest descent to one of 8 neighbors)
    3. Accumulates flow counts downstream

    Args:
        dem: 2D array of elevation values (meters). NaN values are treated
            as nodata and excluded from flow routing.
        profile: Rasterio profile dict with CRS, transform, and dimensions.
        nodata: NoData value for temporary files. Defaults to -9999.0.

    Returns:
        2D array of flow accumulation values (number of upstream cells).
        Invalid cells (nodata, edges) are set to NaN.

    Note:
        The returned values represent cell counts, not area. Multiply by
        cell area (pixel_size²) to get contributing area in m².

    Example:
        >>> dem_data = load_raster(Path("dtm.tif"))
        >>> flow = compute_flow_accumulation(dem_data.array, dem_data.profile)
        >>> flow.max()  # Maximum upstream cells
        15234.0
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        dem_tmp = tmp / "dem.tif"
        filled_tmp = tmp / "filled.tif"
        fac_tmp = tmp / "fac.tif"

        # Save DEM to temporary file
        _save_temp_raster(dem, dem_tmp, profile, nodata)

        # 1) Fill depressions
        _wbt.fill_depressions(str(dem_tmp), str(filled_tmp), fix_flats=True)

        # 2) D8 flow accumulation
        _wbt.d8_flow_accumulation(str(filled_tmp), str(fac_tmp), out_type="cells")

        # Read result
        fac_arr = load_raster(fac_tmp).array.copy()

    # Clean invalid values
    invalid = ~np.isfinite(fac_arr) | (fac_arr < 0)
    fac_arr[invalid] = np.nan

    return fac_arr


def ensure_radians(slope: ArrayF32, threshold: float = 3.2) -> ArrayF32:
    """Convert slope to radians if it appears to be in degrees."""
    max_val = np.nanmax(slope)
    if max_val > threshold:
        _log(f"  Slope in degrees (max={max_val:.2f}), converting to radians.")
        return np.deg2rad(slope)
    return slope


def compute_twi(
    sca: ArrayF32,
    slope_rad: ArrayF32,
    eps_tan: float = 1e-12,
    eps_sca: float = 1e-6,
    clip_range: tuple[float, float] | None = None
) -> ArrayF32:
    """Compute the Topographic Wetness Index (TWI).

    TWI is calculated as: TWI = ln(SCA / tan(β))

    Where:
    - SCA = Specific Catchment Area (contributing area per unit contour length)
    - β = Local slope angle

    Higher TWI values indicate flatter areas with large upslope contributing
    areas, which tend to accumulate water and remain wetter.

    Args:
        sca: Specific Catchment Area in meters. Typically computed as
            flow_accumulation * pixel_size.
        slope_rad: Slope values in radians. If values exceed ~3.2, they are
            assumed to be in degrees and automatically converted.
        eps_tan: Minimum threshold for tan(slope) to prevent division by zero
            in flat areas. Defaults to 1e-12.
        eps_sca: Minimum threshold for SCA to prevent log(0). Defaults to 1e-6.
        clip_range: Optional (min, max) tuple to clip extreme TWI values.
            Useful for visualization or to remove outliers.

    Returns:
        2D array of TWI values (dimensionless). Typical values range from
        2-3 (steep ridges) to 15-20+ (flat valley bottoms).

    Example:
        >>> sca = flow_accumulation * 100  # 100m pixel size
        >>> slope = load_raster(Path("slope.tif")).array
        >>> twi = compute_twi(sca, slope, clip_range=(0, 25))
        >>> twi.mean()
        8.5
    """
    slope_rad = ensure_radians(slope_rad)

    # Calculate tan(beta) with minimum
    tan_beta = np.tan(slope_rad)
    np.maximum(tan_beta, eps_tan, out=tan_beta)

    # SCA with minimum
    sca_safe = np.maximum(sca, eps_sca)

    # TWI = ln(SCA / tan(beta))
    twi = np.log(sca_safe / tan_beta, dtype=np.float32)

    if clip_range is not None:
        np.clip(twi, clip_range[0], clip_range[1], out=twi)

    return twi


def apply_nodata_mask(arr: ArrayF32, mask: NDArray[np.bool_]) -> ArrayF32:
    """Apply nodata mask (True = nodata) to array."""
    arr[mask] = np.nan
    return arr


# =====================================================================
# STATISTICS
# =====================================================================

def compute_stats(arr: ArrayF32, nodata: float = -9999.0) -> tuple[float, float, float]:
    """Compute (min, max, mean) ignoring nodata and NaN."""
    valid = arr[(arr != nodata) & np.isfinite(arr)]
    if valid.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(valid.min()), float(valid.max()), float(valid.mean()))


# =====================================================================
# REGION PROCESSING
# =====================================================================

@dataclass
class RegionResult:
    """Processing results for a single region.

    Contains metadata and statistics for a successfully processed region,
    returned by process_region() and collected by run_twi_processing().

    Attributes:
        region: Name/identifier of the region (extracted from filename).
        dtm_path: Path to the input DTM file.
        slope_path: Path to the input slope file.
        flow_stats: Tuple of (min, max, mean) for flow accumulation.
        sca_stats: Tuple of (min, max, mean) for Specific Catchment Area.
        twi_stats: Tuple of (min, max, mean) for TWI values.

    Example:
        >>> result = process_region(Path("dtm_vigo.tif"), config)
        >>> result.region
        'vigo'
        >>> result.twi_stats
        (2.1, 18.7, 9.3)
    """
    region: str
    dtm_path: Path
    slope_path: Path
    flow_stats: tuple[float, float, float]
    sca_stats: tuple[float, float, float]
    twi_stats: tuple[float, float, float]


def find_slope_file(dtm_path: Path, prefix: str = "slope_") -> Path | None:
    """Find the slope file corresponding to the DTM."""
    region = dtm_path.stem.replace("dtm_", "")
    slope_path = dtm_path.parent / f"{prefix}{region}.tif"
    return slope_path if slope_path.exists() else None


def process_region(dtm_path: Path, config: ConfigTWI) -> RegionResult | None:
    """Process a single region: compute flow accumulation, SCA, and TWI.

    This function performs the complete TWI workflow for one region:
    1. Loads the DTM and corresponding slope raster
    2. Resamples slope to match DTM resolution if needed
    3. Computes flow accumulation using D8 algorithm
    4. Calculates Specific Catchment Area (SCA = flow * pixel_size)
    5. Computes TWI = ln(SCA / tan(slope))
    6. Saves all outputs as GeoTIFFs

    Args:
        dtm_path: Path to the DTM file. Expected naming: "dtm_<region>.tif".
            The corresponding slope file is found using config.slope_prefix.
        config: Configuration object with processing parameters.

    Returns:
        RegionResult with statistics if processing succeeded.
        None if the region was skipped (missing slope file or outputs exist
        and overwrite=False).

    Note:
        Output files are saved in the same directory as the DTM:
        - flow_<region>.tif: Flow accumulation (cells)
        - sca_<region>.tif: Specific Catchment Area (m)
        - twi_<region>.tif: Topographic Wetness Index

    Example:
        >>> config = ConfigTWI(verbose=False)
        >>> result = process_region(Path("data/dtm_vigo.tif"), config)
        >>> if result:
        ...     print(f"TWI range: {result.twi_stats[0]:.1f} - {result.twi_stats[1]:.1f}")
        TWI range: 2.1 - 18.7
    """
    region = dtm_path.stem.replace("dtm_", "")
    folder = dtm_path.parent

    _log(f"\n{'='*50}")
    _log(f"Processing: {region}")
    _log(f"{'='*50}")

    # Find slope file
    slope_path = find_slope_file(dtm_path, config.slope_prefix)
    if slope_path is None:
        _log(f"  SKIPPING: {config.slope_prefix}{region}.tif not found")
        return None

    # Output paths
    flow_out = folder / f"flow_{region}.tif"
    sca_out = folder / f"sca_{region}.tif"
    twi_out = folder / f"twi_{region}.tif"

    # Check if outputs already exist
    if not config.overwrite and all(p.exists() for p in [flow_out, sca_out, twi_out]):
        _log(f"  SKIPPING: Outputs already exist.")
        return None

    # Load DTM (single read)
    _log("  Loading DTM...")
    dtm_data = load_raster(dtm_path)
    pixel_size = float(dtm_data.transform.a)
    nodata_mask = dtm_data.nodata_mask

    # Load and resample slope
    _log("  Loading and resampling slope...")
    slope_data = load_raster(slope_path)

    slope_100m = resample_to_grid(
        slope_data.array, slope_data.transform, slope_data.crs,
        dtm_data.shape, dtm_data.transform, dtm_data.crs,
        method=config.resample_method
    )

    # Propagate nodata masks to slope
    if slope_data.nodata is not None:
        slope_100m[np.isclose(slope_100m, slope_data.nodata)] = np.nan
    slope_100m[nodata_mask] = np.nan

    # 1) Flow accumulation
    _log("  Computing flow accumulation (D8)...")
    flow_arr = compute_flow_accumulation(dtm_data.array, dtm_data.profile, config.nodata_out)
    apply_nodata_mask(flow_arr, nodata_mask)
    save_raster(flow_arr, flow_out, dtm_path, config.nodata_out)
    _log(f"  -> {flow_out.name}")

    # 2) Specific Catchment Area
    _log("  Computing SCA...")
    sca_arr = flow_arr * pixel_size
    apply_nodata_mask(sca_arr, nodata_mask)
    save_raster(sca_arr, sca_out, dtm_path, config.nodata_out)
    _log(f"  -> {sca_out.name}")

    # 3) TWI
    _log("  Computing TWI...")
    twi_arr = compute_twi(
        sca_arr, slope_100m,
        eps_tan=config.eps_tan,
        eps_sca=config.eps_sca,
        clip_range=config.twi_clip
    )
    apply_nodata_mask(twi_arr, nodata_mask)
    save_raster(twi_arr, twi_out, dtm_path, config.nodata_out)
    _log(f"  -> {twi_out.name}")

    # Statistics
    flow_stats = compute_stats(flow_arr, config.nodata_out)
    sca_stats = compute_stats(sca_arr, config.nodata_out)
    twi_stats = compute_stats(twi_arr, config.nodata_out)

    _log(f"  FLOW [min, max, mean]: {flow_stats}")
    _log(f"  SCA  [min, max, mean]: {sca_stats}")
    _log(f"  TWI  [min, max, mean]: {twi_stats}")

    return RegionResult(
        region=region,
        dtm_path=dtm_path,
        slope_path=slope_path,
        flow_stats=flow_stats,
        sca_stats=sca_stats,
        twi_stats=twi_stats
    )


# =====================================================================
# MAIN FUNCTIONS
# =====================================================================

def compute_twi_from_files(
    dtm_path: Path | str,
    slope_path: Path | str,
    twi_output_path: Path | str,
    *,
    flow_output_path: Path | str | None = None,
    sca_output_path: Path | str | None = None,
    nodata: float = -9999.0,
    eps_tan: float = 1e-12,
    eps_sca: float = 1e-6,
    twi_clip: tuple[float, float] | None = None,
    verbose: bool = True,
) -> ArrayF32:
    """Compute TWI from DTM and slope files and save the result.

    This is a convenience function for processing a single pair of files
    without needing to create a ConfigTWI object or follow naming conventions.

    The function:
    1. Loads DTM and slope rasters
    2. Resamples slope to match DTM grid if dimensions differ
    3. Computes flow accumulation (D8)
    4. Calculates Specific Catchment Area (SCA = flow * pixel_size)
    5. Computes TWI = ln(SCA / tan(slope))
    6. Saves output raster(s)

    Args:
        dtm_path: Path to the Digital Terrain Model raster.
        slope_path: Path to the slope raster (radians or degrees).
        twi_output_path: Path where the TWI raster will be saved.
        flow_output_path: Optional path to save flow accumulation raster.
        sca_output_path: Optional path to save SCA raster.
        nodata: NoData value for output rasters. Defaults to -9999.0.
        eps_tan: Minimum for tan(slope) to avoid division by zero.
            Defaults to 1e-12.
        eps_sca: Minimum for SCA to avoid log(0). Defaults to 1e-6.
        twi_clip: Optional (min, max) tuple to clip TWI values.
        verbose: If True, print progress messages. Defaults to True.

    Returns:
        2D numpy array (float32) with TWI values. NoData pixels are set to NaN.

    Raises:
        FileNotFoundError: If dtm_path or slope_path don't exist.
        rasterio.errors.RasterioIOError: If files cannot be read/written.

    Example:
        Basic usage:

        >>> twi_array = compute_twi_from_files(
        ...     "dem.tif",
        ...     "slope.tif",
        ...     "twi_output.tif"
        ... )
        >>> print(f"TWI range: {np.nanmin(twi_array):.1f} - {np.nanmax(twi_array):.1f}")
        TWI range: 2.3 - 19.1

        With all intermediate outputs:

        >>> twi_array = compute_twi_from_files(
        ...     dtm_path="dem.tif",
        ...     slope_path="slope.tif",
        ...     twi_output_path="twi.tif",
        ...     flow_output_path="flow.tif",
        ...     sca_output_path="sca.tif",
        ...     twi_clip=(0, 25),
        ...     verbose=False
        ... )
        >>> twi_array.shape
        (500, 600)
    """
    # Convert to Path if strings
    dtm_path = Path(dtm_path)
    slope_path = Path(slope_path)
    twi_output_path = Path(twi_output_path)

    if flow_output_path is not None:
        flow_output_path = Path(flow_output_path)
    if sca_output_path is not None:
        sca_output_path = Path(sca_output_path)

    # Set verbosity
    _set_verbose(verbose)

    # Validate inputs
    if not dtm_path.exists():
        raise FileNotFoundError(f"DTM file not found: {dtm_path}")
    if not slope_path.exists():
        raise FileNotFoundError(f"Slope file not found: {slope_path}")

    # Load DTM
    _log(f"Loading DTM: {dtm_path.name}")
    dtm_data = load_raster(dtm_path)
    pixel_size = float(dtm_data.transform.a)
    nodata_mask = dtm_data.nodata_mask

    # Load slope
    _log(f"Loading slope: {slope_path.name}")
    slope_data = load_raster(slope_path)

    # Resample slope if needed
    if slope_data.shape != dtm_data.shape:
        _log(f"Resampling slope from {slope_data.shape} to {dtm_data.shape}")
        slope_arr = resample_to_grid(
            slope_data.array, slope_data.transform, slope_data.crs,
            dtm_data.shape, dtm_data.transform, dtm_data.crs
        )
    else:
        slope_arr = slope_data.array.copy()

    # Apply nodata masks to slope
    if slope_data.nodata is not None:
        slope_arr[np.isclose(slope_arr, slope_data.nodata)] = np.nan
    slope_arr[nodata_mask] = np.nan

    # 1) Flow accumulation
    _log("Computing flow accumulation (D8)...")
    flow_arr = compute_flow_accumulation(dtm_data.array, dtm_data.profile, nodata)
    apply_nodata_mask(flow_arr, nodata_mask)

    if flow_output_path is not None:
        save_raster(flow_arr, flow_output_path, dtm_path, nodata)
        _log(f"  -> Saved: {flow_output_path.name}")

    # 2) Specific Catchment Area
    _log("Computing SCA...")
    sca_arr = flow_arr * pixel_size
    apply_nodata_mask(sca_arr, nodata_mask)

    if sca_output_path is not None:
        save_raster(sca_arr, sca_output_path, dtm_path, nodata)
        _log(f"  -> Saved: {sca_output_path.name}")

    # 3) TWI
    _log("Computing TWI...")
    twi_arr = compute_twi(
        sca_arr, slope_arr,
        eps_tan=eps_tan,
        eps_sca=eps_sca,
        clip_range=twi_clip
    )
    apply_nodata_mask(twi_arr, nodata_mask)

    save_raster(twi_arr, twi_output_path, dtm_path, nodata)
    _log(f"  -> Saved: {twi_output_path.name}")

    # Log statistics
    twi_stats = compute_stats(twi_arr, nodata)
    _log(f"TWI stats [min, max, mean]: {twi_stats}")

    return twi_arr


def find_dtm_files(config: ConfigTWI) -> list[Path]:
    """Find all DTM files matching the configuration pattern.

    Searches through all directories specified in config.directories for files
    matching config.dtm_pattern. Optionally filters to a single region.

    Args:
        config: Configuration object specifying search directories, file pattern,
            and optional single_region filter.

    Returns:
        Sorted list of Path objects for matching DTM files.
        Empty list if no files found or directories don't exist.

    Example:
        >>> config = ConfigTWI(base_path=Path("/data"), dtm_pattern="dtm_*.tif")
        >>> dtm_files = find_dtm_files(config)
        >>> [f.name for f in dtm_files]
        ['dtm_lugo.tif', 'dtm_ourense.tif', 'dtm_vigo.tif']

        >>> config = ConfigTWI(base_path=Path("/data"), single_region="vigo")
        >>> find_dtm_files(config)
        [Path('/data/region1/dtm_vigo.tif')]
    """
    dtm_files = []
    for directory in config.directories:
        if directory.exists():
            dtm_files.extend(directory.glob(config.dtm_pattern))

    # Filter by specific region if specified
    if config.single_region:
        target = f"dtm_{config.single_region.lower()}.tif"
        dtm_files = [p for p in dtm_files if p.name.lower() == target]

    return sorted(dtm_files)

def write_log_csv(results: Sequence[RegionResult], path: Path) -> None:
    """Write CSV log with results."""
    header = [
        "region", "path_dtm", "path_slope",
        "flow_min", "flow_max", "sca_min", "sca_max",
        "twi_min", "twi_max", "twi_mean"
    ]

    rows = [header]
    for r in results:
        rows.append([
            r.region,
            str(r.dtm_path),
            str(r.slope_path),
            r.flow_stats[0], r.flow_stats[1],
            r.sca_stats[0], r.sca_stats[1],
            r.twi_stats[0], r.twi_stats[1], r.twi_stats[2]
        ]) # type: ignore

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def run_twi_processing(
    config: ConfigTWI | None = None,
    dtm_files: Sequence[Path] | None = None,
    save_log: bool = True,
) -> list[RegionResult]:
    """Process multiple regions and compute TWI in parallel.

    This is the main entry point for using the module programmatically.
    It orchestrates the parallel processing of multiple regions and
    collects the results.

    Args:
        config: TWI configuration object. If None, uses ConfigTWI defaults.
        dtm_files: Explicit list of DTM files to process. If None, files are
            discovered automatically using find_dtm_files(config).
        save_log: If True, writes a CSV summary to config.log_csv_path.
            Defaults to True.

    Returns:
        List of RegionResult objects for successfully processed regions.
        Skipped regions (missing files, already processed) are not included.

    Raises:
        FileNotFoundError: If config.base_path doesn't exist and subdirs=None.

    Example:
        Basic usage with defaults:

        >>> results = run_twi_processing()

        Custom configuration:

        >>> config = ConfigTWI(
        ...     base_path=Path("/data/galicia"),
        ...     max_workers=8,
        ...     verbose=False,
        ...     overwrite=True
        ... )
        >>> results = run_twi_processing(config)
        >>> len(results)
        4

        Process specific files:

        >>> files = [Path("dtm_vigo.tif"), Path("dtm_lugo.tif")]
        >>> results = run_twi_processing(config, dtm_files=files, save_log=False)
    """
    if config is None:
        config = ConfigTWI()

    # Set global verbosity
    _set_verbose(config.verbose)

    # Find DTMs if not provided
    if dtm_files is None:
        dtm_files = find_dtm_files(config)

    if not dtm_files:
        _log("No regional DTM files found.")
        return []

    _log(f"Found {len(dtm_files)} regional DTM files.")
    _log(f"Using {config.max_workers} parallel workers.")

    # Process regions in parallel
    process_func = partial(process_region, config=config)

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        all_results = list(executor.map(process_func, list(dtm_files)))

    # Filter valid results (not None)
    results: list[RegionResult] = [r for r in all_results if r is not None]

    # Save log
    if save_log and results:
        try:
            write_log_csv(results, config.log_csv_path)
            _log(f"\nCSV log saved: {config.log_csv_path}")
        except OSError as e:
            _log(f"\nError saving CSV log: {e}")

    _log(f"\nProcessing completed: {len(results)} regions.")

    return results

def reclass_twi(twi_arr: NDArray) -> NDArray[np.int16]:
    """Reclassify TWI values into 5 risk categories.

    Converts continuous TWI values into discrete categories based on
    fire risk thresholds. Higher TWI values (wetter areas) correspond
    to lower fire risk codes.

    Classification scheme:
        - Very high risk (code 5): TWI 2–6 (dry ridges/slopes)
        - High risk (code 4): TWI 6–8
        - Moderate risk (code 3): TWI 8–10
        - Low risk (code 2): TWI 10–14
        - Very low risk (code 1): TWI 14–25 (wet valley bottoms)

    Args:
        twi_arr: 2D array of TWI values. NaN values and values outside
            range are assigned nodata (-9999).

    Returns:
        2D array of int16 classification codes (1–5), with -9999 for
        nodata/out-of-range pixels.
    """
    conditions = [
        (twi_arr >= 2.0) & (twi_arr < 6.0),    # Muy alto
        (twi_arr >= 6.0) & (twi_arr < 8.0),    # Alto
        (twi_arr >= 8.0) & (twi_arr < 10.0),   # Moderado
        (twi_arr >= 10.0) & (twi_arr < 14.0),  # Bajo
        (twi_arr >= 14.0) & (twi_arr <= 25.0), # Muy bajo
    ]
    choices = [5, 4, 3, 2, 1]

    return np.select(conditions, choices, default=-9999).astype(np.int16)

def main() -> None:
    """Entry point for direct script execution."""
    run_twi_processing()


if __name__ == "__main__":
    main()
