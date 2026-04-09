import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import Counter
import logging
import argparse
from topsocnww3sp.utils import get_config

# --- Configuration ---
# DISTANCE_THRESHOLD_KM = 20.0
# TIME_THRESHOLD_MINUTES = 30.0

def haversine(lon1, lat1, lon2, lat2):
    """Calculate great circle distance in km."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    return c * 6371

def parse_track_file(filepath):
    """Parses the text track file into a list of dictionaries."""
    data = []
    logging.info(f"Reading track file: {filepath}")
    try:
        with open(filepath, 'r') as f:
            header = f.readline()
            for i, line in enumerate(f):
                parts = line.split()
                if len(parts) < 4: continue
                dt_str = f"{parts[0]} {parts[1]}"
                try:
                    dt = datetime.strptime(dt_str, "%Y%m%d %H%M%S")
                    data.append({
                        'line_idx': i + 2,
                        'datetime': dt,
                        'longitude': float(parts[2]),
                        'latitude': float(parts[3])
                    })
                except ValueError: continue
    except FileNotFoundError:
        logging.error(f"Trackfile not found: {filepath}")
    return data

def core_count_coverage(track_points, ww3_nc_lons, ww3_nc_lats, ww3_nc_times,pathconfig=None):
    """
    
    Arguments:
        track_points: List of dicts with keys 'line_idx', 'datetime', 'longitude', 'latitude'
        ww3_nc_lons: 1D array of longitudes from WW3 NetCDF
        ww3_nc_lats: 1D array of latitudes from WW3 NetCDF
        ww3_nc_times: 1D array of datetimes from WW3 NetCDF
        pathconfig: Optional path to config.yml for thresholds

    Returns:
        summary_lines: List of strings summarizing the distribution
        results: Dict mapping track line index to count of associated spectra
    
    """
    config = get_config(path_config=pathconfig)
    results = {}
    time_delta = timedelta(minutes=config['TIME_THRESHOLD_MINUTES'])

    # 3. Matching Loop
    logging.info("Starting matching process...")
    for pt in tqdm(track_points, desc="Matching"):
        t_start, t_end = pt['datetime'] - time_delta, pt['datetime'] + time_delta
        
        # Temporal mask
        time_mask = (ww3_nc_times >= t_start) & (ww3_nc_times <= t_end)
        
        if not np.any(time_mask):
            results[pt['line_idx']] = 0
            continue

        # Spatial distance for points in time window
        distances = haversine(pt['longitude'], pt['latitude'], 
                              ww3_nc_lons[time_mask], ww3_nc_lats[time_mask])
        
        results[pt['line_idx']] = int(np.sum(distances <= config['DISTANCE_THRESHOLD_KM']))

    # 4. Distribution and Statistics Calculation
    dist_counts = Counter(results.values())
    total_tiles = len(track_points)
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append(f"{'DISTRIBUTION OF WW3 SPECTRA ASSOCIATED PER OCN TILE within %i minutes and %i km':^80}"%( config['TIME_THRESHOLD_MINUTES'], config['DISTANCE_THRESHOLD_KM']))
    summary_lines.append("="*80)
    summary_lines.append(f"{'Spectra Count':<20} | {'Trackfile Points':<15} | {'Percentage':<15}")
    summary_lines.append("-" * 80)
    
    # Sort by number of spectra (0, 1, 2...)
    for count in sorted(dist_counts.keys()):
        freq = dist_counts[count]
        pct = (freq / total_tiles) * 100
        summary_lines.append(f"{count:<20} | {freq:<15} | {pct:>6.2f}%")
    
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'TOTAL':<20} | {total_tiles:<15} | 100.00%")
    summary_lines.append("="*80)

    # Print summary to terminal
    for line in summary_lines:
        print(line)
    return summary_lines, results

   


def main():
    parser = argparse.ArgumentParser(description="Match WW3 track output with a trackfile list.")
    parser.add_argument("-t", "--trackfile", required=True)
    parser.add_argument("-n", "--ncfiles", nargs='+', required=True)
    parser.add_argument("-o", "--output", default="counts_report.txt")
    parser.add_argument("--config", default=None, help="Path to config.yml (optional)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
     # 1. Load Trackfile
    track_points = parse_track_file(args.trackfile)
    if not track_points:
        return
     # 2. Open NetCDF datasets
    logging.info(f"Opening {len(args.ncfiles)} NetCDF file(s)...")
    try:
        ds = xr.open_mfdataset(args.ncfiles, combine='nested', concat_dim='time', decode_times=True)
        ww3_nc_lons = ds.longitude.values
        ww3_nc_lats = ds.latitude.values
        ww3_nc_times = pd.to_datetime(ds.time.values)
    except Exception as e:
        logging.error(f"Failed to process NetCDF: {e}")
        return
    
    summary_lines, results = core_count_coverage(track_points,
                                                  ww3_nc_lons,
                                                 ww3_nc_lats,
                                                   ww3_nc_times,
                                                    pathconfig=args.config
                                                   )
     # 5. Write to Output File
    logging.info(f"Writing detailed report to {args.output}")
    with open(args.output, 'w') as out:
        out.write("\n".join(summary_lines) + "\n\n")
        out.write("DETAILED DATA:\n")
        out.write("Line_Number | DateTime | Longitude | Latitude | Spectra_Match_Count\n")
        out.write("-" * 80 + "\n")
        for pt in track_points:
            idx = pt['line_idx']
            out.write(f"{idx:<11} | {pt['datetime']} | {pt['longitude']:>9.3f} | {pt['latitude']:>8.3f} | {results[idx]}\n")

if __name__ == "__main__":
    main()