#!/usr/bin/env python3
import os
import glob
import pandas as pd
import argparse
from tqdm import tqdm

def concatenate_trackfiles(input_dir, output_file):
    pattern = os.path.join(input_dir, 'trackfile-ww3spectra-agnostic-*.txt')
    trackfiles = glob.glob(pattern)
    
    if not trackfiles:
        print("No trackfiles found!")
        return
    
    all_data = []
    
    for trackfile in tqdm(trackfiles, desc="Reading trackfiles"):
        try:
            # Read columns. Keep them as strings or ints. 
            # We use dtype to ensure HHMMSS doesn't lose leading zeros if handled as strings, 
            # but sorting integers is actually safer.
            df = pd.read_csv(trackfile, sep='\s+', header=None, 
                           names=['YYYYMMDD', 'HHMMSS', 'lon_str', 'lat_str'],
                           dtype={'YYYYMMDD': int, 'HHMMSS': int})
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {trackfile}: {e}")
    
    if not all_data:
        return

    # 1. Concatenate
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 2. Sort by Date AND Time columns globally
    print("Sorting data chronologically...")
    combined_df = combined_df.sort_values(by=['YYYYMMDD', 'HHMMSS']).reset_index(drop=True)
    
    # 3. Save to output
    print(f"Saving to {output_file}")
    with open(output_file, 'w') as f:
        f.write("WAVEWATCH III TRACK LOCATIONS DATA \n")
        
        # Optimization: Create formatted strings for the whole dataframe
        # This is much faster than opening the file in 'a' mode inside a loop
        for _, row in combined_df.iterrows():
            line = (
                f"{int(row['YYYYMMDD']):8d} "
                f"{int(row['HHMMSS']):06d} " # 06d ensures leading zeros (e.g., 062501)
                f"{float(row['lon_str']):10.5f} "
                f"{float(row['lat_str']):10.5f}\n"
            )
            f.write(line)
    
    print(f"Done. Total records: {len(combined_df)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()
    concatenate_trackfiles(args.input_dir, args.output_file)

if __name__ == '__main__':
    main()