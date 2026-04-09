#!/usr/bin/python
# encoding *-utf-8-*
"""
purpose: version of WW3 trackfile generator adapted to S1 ESA TOPS OCN Level-2 osw cross spectra nc files
This version generates an agnostic trackfile with raw positions from OCN files ordered by dates
"""
import traceback
import sys
import os
from  tqdm import tqdm
import datetime
import xarray as xr
import pandas as pd
import glob
import logging
import argparse
import time
import numpy as np
import shapely
from shapely.wkt import loads
import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

def get_polygon_subswath(dsosw)->shapely.geometry.polygon.Polygon:
    """
    method to get polygon of subswath from osw intraburst group

    :param dsosw: xarray dataset of osw intraburst group
    :return: shapely polygon of the subswath
    """
    # use convex hull method from shapely to create polygon with each grid points
    lons = dsosw['oswLon'].squeeze().values.ravel()
    lats = dsosw['oswLat'].squeeze().values.ravel()
    # Créer une liste de points (lon, lat)
    points = list(zip(lons, lats))

    # Créer un MultiPoint et calculer l'enveloppe convexe
    multi_point = MultiPoint(points)
    convex_hull_polygon = multi_point.convex_hull
    return convex_hull_polygon

def collect_each_matching_locations(safedir, dirout):
    """
    Collect raw positions from OCN files and create an agnostic trackfile
    
    :param safedir : str path of S1 OCN SAFE where to find measurement (.nc) files
    :param dirout: str path where to store the .txt files
    :return:
    """
    # Pattern to find OCN measurement files
    pattern_osw = os.path.join(safedir, 'measurement', '*osw*.nc')
    lst_osw = glob.glob(pattern_osw)
    base_safe = os.path.basename(safedir)
    logging.info('SAFE to process: %s', base_safe)
    lst_osw = sorted(lst_osw)
    logging.info('nb L2 OCN osw files to read : %s', len(lst_osw))
    
    # Store all positions with their timestamps
    all_positions = []
    
    pbar = tqdm(range(len(lst_osw)), disable=True if len(lst_osw) < 10 else False)
    
    for ii in pbar:
        try:
            # Open the OCN file
            dstmp = xr.open_dataset(lst_osw[ii], group='intraburst', engine='h5netcdf')
            
            # Extract the timestamp from filename
            filename = os.path.basename(lst_osw[ii])
            date_part = filename.split('-')[5]  # Extract date part from filename like 20211122T191629
            date = datetime.datetime.strptime(date_part, '%Y%m%dt%H%M%S')
            
            # Get all positions from the subswath
            lons = dstmp['oswLon'].squeeze().values.ravel()
            lats = dstmp['oswLat'].squeeze().values.ravel()
            
            # Create list of (lon, lat, date) tuples for each position
            for lon, lat in zip(lons, lats):
                all_positions.append({
                    'lon': lon,
                    'lat': lat,
                    'date': date
                })
                
        except KeyboardInterrupt:
            raise Exception('stop')
        except Exception as e:
            logging.error('Error processing %s: %s', lst_osw[ii], traceback.format_exc())
            continue
    
    # Sort all positions by date
    all_positions.sort(key=lambda x: x['date'])
    
    # Create DataFrame with sorted positions
    if len(all_positions) > 0:
        df = pd.DataFrame(all_positions)
        logging.info('Total positions collected: %d', len(df))
        logging.info('Date range: %s to %s', 
                    df['date'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    df['date'].max().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Format dates
        df['YYYYMMDD'] = df['date'].dt.strftime('%Y%m%d')
        df['HHMMSS'] = df['date'].dt.strftime('%H%M%S')
        
        # Format coordinates to 2 decimal places
        df['lon_str'] = df['lon'].map(lambda x: '%.2f' % x)
        df['lat_str'] = df['lat'].map(lambda x: '%.2f' % x)
        
        # Reorder columns for trackfile format
        df = df[['YYYYMMDD', 'HHMMSS', 'lon_str', 'lat_str']]
        
        # Create output directory
        os.makedirs(dirout, exist_ok=True)
        
        # Save to file
        fout = os.path.join(dirout, 'trackfile-ww3spectra-agnostic-%s.txt' % (base_safe.replace('.SAFE', '')))
        df.to_csv(fout, header=False, index=False, sep=' ')
        logging.info('Agnostic trackfile saved: %s', fout)
    else:
        logging.info('No positions found')

def entry_point_one_safe():
    """Process a single SAFE directory"""
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    
    parser = argparse.ArgumentParser(description='Agnostic trackfile generator for S1 OCN files')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--outputdir', required=True, help='directory where to store output')
    parser.add_argument('--OCNSAFE', required=True, help='directory SAFE where to find OCN files')
    args = parser.parse_args()
    
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    
    t0 = time.time()
    collect_each_matching_locations(safedir=args.OCNSAFE, dirout=args.outputdir)
    elapsed = time.time()
    logging.info('Time to process SAFE: %1.1f seconds', elapsed - t0)

def entry_point_one_listing_of_safe():
    """Process multiple SAFE directories from a listing file"""
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    
    parser = argparse.ArgumentParser(description='Agnostic trackfile generator for S1 OCN files')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--outputdir', required=True, help='directory where to store output')
    parser.add_argument('--listing-safe', required=True, help='path of a listing of OCN SAFE files')
    args = parser.parse_args()
    
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    
    t0 = time.time()
    safes = pd.read_csv(args.listing_safe, header=None)[0].tolist()
    
    for ss in tqdm(range(len(safes))):
        logging.info('Processing SAFE: %s', safes[ss])
        collect_each_matching_locations(safedir=safes[ss], dirout=args.outputdir)
    
    elapsed = time.time()
    logging.info('Check out the txt files generated in %s', args.outputdir)
    logging.info('Total time: %1.1f seconds', elapsed - t0)

def entry_point_ocn_between_dates():
    """Process OCN files between specific dates"""
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    
    parser = argparse.ArgumentParser(description='Agnostic trackfile generator for S1 OCN files')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--outputdir', required=True, help='directory where to store output')
    parser.add_argument('--inputdir', required=True, help='directory where OCN are stored')
    parser.add_argument('--start', required=True, help='start date YYYYMMDD inclusive')
    parser.add_argument('--end', required=True, help='end date YYYYMMDD inclusive')
    args = parser.parse_args()
    
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    
    t0 = time.time()
    safes = []
    
    for day in pd.date_range(start=args.start, end=args.end):
        pattern_safe = os.path.join(args.inputdir, day.strftime('%Y'), day.strftime('%m'), day.strftime('%d'), 'S1*_OCN__2S*.SAFE')
        lst_safe = glob.glob(pattern_safe)
        safes.extend(lst_safe)
    
    logging.info('Number of SAFE to process: %s', len(safes))
    
    for ss in tqdm(range(len(safes))):
        logging.info('Processing SAFE: %s', safes[ss])
        collect_each_matching_locations(safedir=safes[ss], dirout=args.outputdir)
    
    elapsed = time.time()
    logging.info('Total time: %1.1f seconds', elapsed - t0)

if __name__ == '__main__':
    # Use one of these entry points depending on your needs:
    # entry_point_one_safe()  # Process single SAFE directory
    # entry_point_one_listing_of_safe()  # Process multiple SAFE directories from listing file
    # entry_point_ocn_between_dates()  # Process SAFE directories between dates
    
    # Current usage - process listing file
    entry_point_one_listing_of_safe()