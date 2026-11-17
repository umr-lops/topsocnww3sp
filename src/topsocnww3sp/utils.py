#!/usr/bin/env python3
"""Utility functions for the topsocnww3sp package."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def get_config(path_config: str | Path | None = None) -> dict:
    """Load configuration from a YAML file.

    Args:
        path_config (str or Path, optional): Path to the YAML configuration file.
            If None, defaults to 'config.yml' in the same directory as this script.

    Returns:
        dict: Configuration parameters loaded from the YAML file.
    """
    if path_config is None:
        config_path = Path(__file__).parent / "config.yml"
    else:
        config_path = Path(path_config)

    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate the great circle distance between two points on the earth
    (specified in decimal degrees).

    Args:
        lon1: Longitude of first point in decimal degrees
        lat1: Latitude of first point in decimal degrees
        lon2: Longitude of second point in decimal degrees
        lat2: Latitude of second point in decimal degrees

    Returns:
        Distance between the two points in kilometers
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371


def spherical_centroid(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    """Centroid (lon, lat in degrees) of points, robust to the date line and poles.

    Args:
        lon: Longitudes in degrees
        lat: Latitudes in degrees

    Returns:
        (lon0, lat0) of the normalised mean of the 3D unit vectors
    """
    lam = np.radians(np.asarray(lon, dtype="f8"))
    phi = np.radians(np.asarray(lat, dtype="f8"))
    x = np.mean(np.cos(phi) * np.cos(lam))
    y = np.mean(np.cos(phi) * np.sin(lam))
    z = np.mean(np.sin(phi))
    return float(np.degrees(np.arctan2(y, x))), float(
        np.degrees(np.arctan2(z, np.hypot(x, y)))
    )


def gnomonic_projection(
    lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic projection of points on the plane tangent at (lon0, lat0).

    Great circles map to straight lines, so the planar convex hull of
    projected points is their spherical convex hull. This keeps the lasso
    footprint compact across the +/-180 meridian and near the poles, where a
    hull built in raw lon/lat spans the whole globe. Coordinates are in units
    of the Earth radius (~radians close to the tangent point); points 90
    degrees or more away from the tangent point get NaN.

    Args:
        lon: Longitudes in degrees
        lat: Latitudes in degrees
        lon0: Tangent point longitude in degrees
        lat0: Tangent point latitude in degrees

    Returns:
        (x, y) projected coordinates
    """
    lam = np.radians(np.asarray(lon, dtype="f8"))
    phi = np.radians(np.asarray(lat, dtype="f8"))
    lam0 = np.radians(lon0)
    phi0 = np.radians(lat0)
    dlam = lam - lam0
    cos_c = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(dlam)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(cos_c > 0, 1.0 / cos_c, np.nan)
    x = np.cos(phi) * np.sin(dlam) * inv
    y = (np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(dlam)) * inv
    return x, y


def tidy_ww3_track_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Remove the WW3 ``track_name`` character array when it carries no data.

    The WW3 track files store ``track_name`` as a ``char(time, string32)``
    array with a ``string32`` index coordinate; in the IRI runs it is always
    empty. Writing it back produced an undescriptive ``string32`` dimension
    (and a spurious ``string1``) in the L2C products (issue #24). The
    ``string32`` coordinate is always dropped; ``track_name`` is dropped when
    empty, otherwise it is kept as a plain string variable (no extra dim).

    Args:
        ds: WW3 track dataset (track dimension first for ``track_name``)

    Returns:
        Dataset without ``string32``, and without an empty ``track_name``
    """
    ds = ds.drop_vars(["string32"], errors="ignore")
    if "track_name" not in ds.variables:
        return ds
    var = ds["track_name"]
    track_dim = var.dims[0]
    raw = np.asarray(var.to_numpy())
    if raw.dtype.kind in "SU" and raw.ndim > 1:
        # char array (n, nchar[, 1]) -> one string per track point
        flat = raw.reshape(raw.shape[0], -1).astype("S1")
        names = np.array([b"".join(row).decode("utf-8", "replace") for row in flat])
    else:
        names = np.array(
            [
                v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                for v in raw.ravel()
            ]
        )
    names = np.char.strip(names.astype(str))
    ds = ds.drop_vars("track_name")
    if (names != "").any():
        ds["track_name"] = xr.DataArray(
            names, dims=track_dim, attrs={"long_name": "track name"}
        )
    else:
        logger.debug("Dropping empty WW3 track_name variable")
    return ds


# Parameters integrated from each WW3 spectrum: name -> CF attributes
SPECTRAL_PARAMETERS: dict[str, dict[str, str]] = {
    "hs_spec": {
        "long_name": "significant wave height integrated from the WW3 spectrum "
        "efth: 4 * sqrt(m0)",
        "standard_name": "sea_surface_wave_significant_height",
        "units": "m",
    },
    "tm02_spec": {
        "long_name": "mean wave period Tm02 integrated from the WW3 spectrum "
        "efth: sqrt(m0 / m2)",
        "standard_name": "sea_surface_wave_mean_period_from_variance_spectral_"
        "density_second_frequency_moment",
        "units": "s",
    },
    "tm0m1_spec": {
        "long_name": "mean wave period Tm0,-1 (energy period) integrated from "
        "the WW3 spectrum efth: m-1 / m0",
        "standard_name": "sea_surface_wave_mean_period_from_variance_spectral_"
        "density_inverse_frequency_moment",
        "units": "s",
    },
}


def spectral_integrated_parameters(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Integrated wave parameters of each WW3 directional spectrum.

    With the frequency moments ``mn = sum_f sum_theta f**n E(f, theta)
    df dtheta`` of ``efth`` (m2 s rad-1), returns ``hs_spec = 4 sqrt(m0)``,
    ``tm02_spec = sqrt(m0 / m2)`` and ``tm0m1_spec = m-1 / m0`` (DLR-3, issue
    #20: the IRI WW3 FIELD files only provide ``hs``). The frequency bin
    widths are ``frequency2 - frequency1`` when available, otherwise derived
    from the centre frequencies.

    Args:
        ds: WW3 track dataset with ``efth(<track dim>, frequency, direction)``,
            ``frequency`` and ``direction``

    Returns:
        Mapping of parameter name to a DataArray on the track dimension
        (empty if ``efth`` is missing); NaN where the spectrum is empty
    """
    if "efth" not in ds.variables:
        return {}
    efth = ds["efth"]
    track_dim = efth.dims[0]
    freq = ds["frequency"].astype("f8")
    if "frequency1" in ds.variables and "frequency2" in ds.variables:
        dfreq = (ds["frequency2"] - ds["frequency1"]).astype("f8")
    else:
        # bin edges half way between centre frequencies
        fc = freq.to_numpy()
        edges = np.concatenate(
            (
                [fc[0] - (fc[1] - fc[0]) / 2],
                (fc[1:] + fc[:-1]) / 2,
                [fc[-1] + (fc[-1] - fc[-2]) / 2],
            )
        )
        dfreq = xr.DataArray(np.diff(edges), dims="frequency")
    n_dir = ds.sizes["direction"]
    dtheta = 2.0 * np.pi / n_dir
    # spectrum integrated over directions, times the bin width: m2 per bin
    e_f = efth.astype("f8").fillna(0.0).sum("direction") * dtheta * dfreq
    m_minus1 = (e_f / freq).sum("frequency")
    m0 = e_f.sum("frequency")
    m2 = (e_f * freq**2).sum("frequency")
    valid = m0 > 0
    values = {
        "hs_spec": 4.0 * np.sqrt(m0),
        "tm02_spec": np.sqrt(m0 / m2),
        "tm0m1_spec": m_minus1 / m0,
    }
    out: dict[str, xr.DataArray] = {}
    for name, val in values.items():
        da = val.where(valid).astype("f4").transpose(track_dim)
        da.attrs = dict(SPECTRAL_PARAMETERS[name])
        out[name] = da
    return out


def format_logs(logger_obj: logging.Logger, level: str) -> logging.Logger:
    """
    Configure the logger with a specific format and level.

    Args:
        logger_obj: The logger instance to configure.
        level: The logging level as a string (e.g., "debug", "info").
    returns:
        The configured logger instance.
    """
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    logger_obj.setLevel(logging.DEBUG if level == "debug" else logging.INFO)
    formatter = logging.Formatter(fmt, datefmt="%d/%m/%Y %H:%S")
    for handler in logger_obj.handlers:
        handler.setFormatter(formatter)
    return logger_obj


def load_ww3_multi_grid(
    ww3_dir: Path, sar_start: datetime, config: dict[str, Any]
) -> xr.Dataset:
    """
    Load and merge WW3 data from multiple grids (IRI configuration).

    Args:
        ww3_dir: Directory containing WW3 grid subdirectories
        sar_start: SAR acquisition time for temporal filtering
        config: Configuration dictionary with ww3_grids patterns

    Returns:
        Merged xarray.Dataset with all available WW3 grids
    """

    if "ww3_grids" not in config:
        error_msg = "Missing 'ww3_grids' configuration for multi-grid mode"
        raise ValueError(error_msg)

    t_thresh = timedelta(minutes=config["TIME_THRESHOLD_MINUTES"])
    t_sar = pd.Timestamp(sar_start).tz_localize(None)
    datasets = []
    grid_files_ds = []
    grid_names_used = []

    # Build flag meaning string once
    flag_meaning_str = ", ".join(config["ww3_grids"])
    # Create flag values string: "0 1 2"
    flag_values_str = " ".join([str(i) for i in range(len(config["ww3_grids"]))])

    for grid_name, grid_config in config["ww3_grids"].items():
        # Build pattern with year
        year_str = sar_start.strftime("%Y")
        pattern = grid_config["pattern"].replace("YYYY", year_str)
        logger.debug("Looking for WW3 files in %s with pattern %s", ww3_dir, pattern)
        files = list(ww3_dir.glob(pattern))

        if not files:
            logger.debug(
                "No files found for grid %s with pattern %s", grid_name, pattern
            )
            continue

        # Load and concatenate files for this grid - FIXED: added data_vars='all'
        ds_grid = xr.open_mfdataset(
            files, combine="nested", concat_dim="time", data_vars="all"
        )
        grid_files_ds.append(ds_grid)
        ww3_times = pd.to_datetime(ds_grid.time.to_numpy())

        # Temporal filter
        t_mask = (ww3_times >= t_sar - t_thresh) & (ww3_times <= t_sar + t_thresh)
        if not np.any(t_mask):
            logger.debug("No temporal match for grid %s", grid_name)
            continue

        ds_filtered = ds_grid.isel(time=t_mask)

        # Add provenance variable
        grid_code = list(config["ww3_grids"].keys()).index(grid_name)
        provenance = xr.DataArray(
            np.full(ds_filtered.sizes["time"], grid_code, dtype=np.int64),
            dims="time",
            attrs={
                "long_name": "WW3 grid provenance",
                "flag_meanings": flag_meaning_str,
                "flag_values": flag_values_str,
            },
        )
        ds_filtered["ww3_grid_provenance"] = provenance
        datasets.append(ds_filtered)
        grid_names_used.append(grid_name)

    if not datasets:
        error_msg = f"No WW3 data found in {ww3_dir} for time window around {sar_start}"
        raise FileNotFoundError(error_msg)

    # Merge all grids along time dimension. The time-filtered subset is small
    # (a +/- TIME_THRESHOLD_MINUTES window) and is used once per subswath, so
    # it is loaded in memory once instead of being re-read from disk each time.
    ds_merged = xr.concat(datasets, dim="time", join="inner").load()
    for ds_grid in grid_files_ds:
        ds_grid.close()
    logger.info(
        "Merged %d WW3 grids, total %d time steps",
        len(datasets),
        ds_merged.sizes["time"],
    )

    # Add metadata about grids used
    ds_merged.attrs["ww3_grids_merged"] = ", ".join(grid_names_used)

    return ds_merged


def _find_field_files(ww3_dir: Path, pattern: str) -> list[Path]:
    """Find FIELD files matching a pattern, looking upwards from ww3_dir.

    The run root holding the grid directories sits a few levels above the
    track files, so the search walks up the tree when a track file (or its
    containing directory) is provided instead of the run root.

    Args:
        ww3_dir: Run root directory, or a track file / directory below it
        pattern: Relative glob pattern for the FIELD files

    Returns:
        Sorted list of matching FIELD file paths (possibly empty)
    """
    root = ww3_dir
    if root.is_file():
        root = root.parent
    for _ in range(6):
        files = sorted(root.glob(pattern))
        if files:
            return files
        if root.parent == root:
            break
        root = root.parent
    return []


def _nearest_grid_index(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    pt_lon: np.ndarray,
    pt_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest grid index (flattened) and distance in degrees for each point.

    Points are also queried shifted by +/-360 degrees in longitude so that
    grids crossing the date line (e.g. regular lat/lon grids) resolve the
    correct neighbour across the seam.

    Args:
        grid_lon: Flattened grid longitudes
        grid_lat: Flattened grid latitudes
        pt_lon: Point longitudes
        pt_lat: Point latitudes

    Returns:
        Tuple of (flattened grid index per point, distance in degrees)
    """
    tree = cKDTree(np.column_stack((grid_lon, grid_lat)))
    best_dist = np.full(pt_lon.size, np.inf, dtype="f8")
    best_idx = np.full(pt_lon.size, -1, dtype="i8")
    for shift in (-360.0, 0.0, 360.0):
        dist, idx = tree.query(np.column_stack((pt_lon + shift, pt_lat)), k=1)
        take = dist < best_dist
        best_dist[take] = dist[take]
        best_idx[take] = idx[take]
    return best_dist, best_idx


def _nearest_time_index(
    field_times: np.ndarray,
    track_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest field-time index and absolute time delta for each track time.

    Args:
        field_times: Sorted datetime64 array of FIELD file times
        track_times: datetime64 array of track point times

    Returns:
        Tuple of (field time index per track point, absolute delta in ns)
    """
    pos = np.searchsorted(field_times, track_times)
    if field_times.size == 1:
        idx = np.zeros(track_times.size, dtype="i8")
        return idx, np.abs(field_times[0] - track_times)
    pos = np.clip(pos, 1, field_times.size - 1)
    d_prev = np.abs(field_times[pos - 1] - track_times)
    d_next = np.abs(field_times[pos] - track_times)
    use_next = d_next < d_prev
    idx = np.where(use_next, pos, pos - 1)
    delta = np.where(use_next, d_next, d_prev)
    return idx, delta


def load_ww3_field(
    ww3_dir: Path,
    ds_ww3_tracks: xr.Dataset,
    config: dict[str, Any],
) -> dict[str, xr.DataArray]:
    """Sample WW3 FIELD (integrated parameter) variables at WW3 track points.

    Each track point is sampled only from the FIELD file of its own
    provenance grid (from ``ww3_grid_provenance``, or inferred from the track
    file name in single-file mode), so that each track point gets exactly one
    value per variable, on the track dataset ``time`` dimension.

    Args:
        ww3_dir: Run root directory, or the track file path in single-file mode
        ds_ww3_tracks: Merged WW3 track dataset with ``time``, ``longitude``,
            ``latitude`` and (in multi-grid mode) ``ww3_grid_provenance``
        config: Configuration dictionary with ``ww3_grids``,
            ``ww3_field_grids``, ``ww3_field_variables`` and
            ``TIME_THRESHOLD_MINUTES``; optional ``ww3_field_max_distance_deg``
            (default 0.25) bounds the spatial search

    Returns:
        Mapping of FIELD variable name (preserved, e.g. ``hs``) to a
        DataArray on the track ``time`` dimension with CF attributes (NaN
        where no FIELD file or no matching grid point / time is available).
        Variables that end up entirely NaN are omitted.
    """
    if "ww3_field_grids" not in config or "ww3_field_variables" not in config:
        return {}

    grid_keys = list(config["ww3_grids"].keys())
    if not grid_keys:
        return {}

    t_thresh = timedelta(minutes=config["TIME_THRESHOLD_MINUTES"])
    max_dist_deg = float(config.get("ww3_field_max_distance_deg", 0.25))
    t_thresh_delta = np.timedelta64(round(t_thresh.total_seconds()), "s")

    n = ds_ww3_tracks.sizes["time"]
    track_lons = ds_ww3_tracks["longitude"].to_numpy().astype("f8")
    track_lats = ds_ww3_tracks["latitude"].to_numpy().astype("f8")
    track_times = pd.to_datetime(ds_ww3_tracks["time"].to_numpy())
    track_time_np = track_times.to_numpy()
    t_ref = track_times.to_series().median()

    # Determine, for each track point, the grid it belongs to
    grid_of_point: np.ndarray | None
    if "ww3_grid_provenance" in ds_ww3_tracks.variables:
        provenance = ds_ww3_tracks["ww3_grid_provenance"].to_numpy()
        grid_of_point = np.array(grid_keys, dtype=object)[provenance]
    else:
        # Single-file mode: infer the grid from the track file name
        grid_of_point = None
        fname = ww3_dir.name
        for grid_name, grid_cfg in config["ww3_field_grids"].items():
            prefix = grid_cfg["pattern"].split("WW3-", 1)[1].split("*", 1)[0]
            if fname.startswith(f"WW3-{prefix}"):
                grid_of_point = np.full(n, grid_name, dtype=object)
                break
    if grid_of_point is None:
        logger.warning(
            "Cannot determine the WW3 grid of the track points; "
            "skipping WW3 FIELD variable sampling"
        )
        return {}

    # The FIELD variable name is preserved as the output variable name, so a
    # track point always carries its integrated parameters under the same name
    # used in the WW3 FIELD file (e.g. ``hs`` stays ``hs``).
    field_vars = config["ww3_field_variables"]
    results: dict[str, np.ndarray] = {
        var: np.full(n, np.nan, dtype="f4") for var in field_vars
    }

    for grid_name, grid_cfg in config["ww3_field_grids"].items():
        if grid_name not in grid_keys:
            logger.warning("WW3 FIELD grid %s not in ww3_grids; skipping", grid_name)
            continue
        pt_idx = np.where(grid_of_point == grid_name)[0]
        if pt_idx.size == 0:
            continue

        pattern = grid_cfg["pattern"].replace("YYYY", t_ref.strftime("%Y"))
        files = _find_field_files(ww3_dir, pattern)
        if not files:
            logger.warning(
                "No WW3 FIELD file for grid %s (pattern %s); "
                "its %d track points keep NaN field values",
                grid_name,
                pattern,
                pt_idx.size,
            )
            continue

        ds_field = xr.open_mfdataset(
            files, combine="nested", concat_dim="time", data_vars="all"
        )
        try:
            present = [var for var in field_vars if var in ds_field.data_vars]
            if not present:
                logger.warning(
                    "None of the configured WW3 FIELD variables %s present in %s",
                    sorted(field_vars),
                    ", ".join(f.name for f in files),
                )
                continue

            lon_arr = ds_field["longitude"].to_numpy().astype("f8")
            lat_arr = ds_field["latitude"].to_numpy().astype("f8")
            if lon_arr.ndim == 1:
                lon_arr = lon_arr[None, :]
            if lat_arr.ndim == 1:
                lat_arr = lat_arr[:, None]
            grid_lon, grid_lat = np.broadcast_arrays(lon_arr, lat_arr)
            grid_lon = grid_lon.ravel()
            grid_lat = grid_lat.ravel()
            dist_deg, flat_idx = _nearest_grid_index(
                grid_lon,
                grid_lat,
                track_lons[pt_idx],
                track_lats[pt_idx],
            )
            ok_space = dist_deg <= max_dist_deg
            if not np.all(ok_space):
                logger.warning(
                    "Grid %s: %d track points more than %.2f deg from the "
                    "FIELD grid; kept as NaN",
                    grid_name,
                    int(np.sum(~ok_space)),
                    max_dist_deg,
                )

            field_time_np = pd.to_datetime(ds_field.time.to_numpy()).to_numpy()
            t_idx, t_delta = _nearest_time_index(field_time_np, track_time_np[pt_idx])
            ok_time = t_delta <= t_thresh_delta
            if not np.all(ok_time):
                logger.warning(
                    "Grid %s: %d track points with no FIELD time within "
                    "%d minutes; kept as NaN",
                    grid_name,
                    int(np.sum(~ok_time)),
                    config["TIME_THRESHOLD_MINUTES"],
                )

            ok = ok_space & ok_time
            if not np.any(ok):
                continue
            sel = pt_idx[ok]
            n_lon = ds_field.sizes["longitude"]
            flat_ok = flat_idx[ok]
            i_lat = flat_ok // n_lon
            i_lon = flat_ok % n_lon
            for var in present:
                vals = (
                    ds_field[var]
                    .transpose("time", "latitude", "longitude")
                    .to_numpy()[t_idx[ok], i_lat, i_lon]
                    .astype("f4")
                )
                results[var][sel] = vals
        finally:
            ds_field.close()

    out: dict[str, xr.DataArray] = {}
    for var, arr in results.items():
        if not np.isfinite(arr).any():
            continue
        var_cfg = field_vars[var]
        out[var] = xr.DataArray(
            arr,
            dims="time",
            coords={"time": ds_ww3_tracks["time"].to_numpy()},
            attrs=dict(var_cfg.get("attrs", {})),
        )
    return out
