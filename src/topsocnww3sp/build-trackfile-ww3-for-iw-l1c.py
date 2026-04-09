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
from shapely.wkt import loads
import numpy as np
from shapely.geometry import MultiPoint,Point

lons_ww3 = np.arange(-180,180,0.5) #based on /home/ref-ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/2022/FIELD_NC
lats_ww3 = np.arange(-78,83.5,0.5)
print(lons_ww3.shape,lats_ww3.shape)
XX3,YY3 = np.meshgrid(lons_ww3,lats_ww3)
geogridpts = np.stack([XX3.flatten(),YY3.flatten()]).T
points = MultiPoint(geogridpts)

def collect_each_matching_locations(day_to_treat,product_id,dirout):
    """

    :param day_to_treat: datetime.datetime
    :param product_id: str B07
    :param dirout: str path where to store the .txt files
    :return:
    """
    dir_l1c = '/home/datawork-cersat-public/project/sarwave/data/products/tests2/slc/iw/l1c/' # update Jan 2025 to get the whole training dataset A13->B07 (ifremer+creodias part of training dataset)
    # patl1b = os.path.join(dir_l1b,'*SAFE','*vv*.nc')
    patl1c = os.path.join(dir_l1c,day_to_treat.strftime('%Y'),day_to_treat.strftime('%j'),'*'+product_id+'.SAFE','*1sdv*.nc')
    lst_l1c = glob.glob(patl1c)
    # lst_l1b = glob.glob(patl1b)
    # patl1b = os.path.join(dir_l1b,'*SAFE','*hh*.nc')
    # lst_l1b += glob.glob(patl1b)
    # lst_l1b = sorted(lst_l1b)
    lst_l1c = sorted(lst_l1c)
    logging.info('nb L1C files to read : %s',len(lst_l1c))
    pbar = tqdm(range(len(lst_l1c)))
    df = pd.DataFrame()
    dates = []
    lons_match = []
    lats_match = []
    cpt_undreadable = 0
    # unreadable_l1b = []
    unreadable_l1c = []
    for ii in pbar:
        try:
            dstmp = xr.open_dataset(lst_l1c[ii],group='intraburst',engine='h5netcdf')
            date = datetime.datetime.strptime(os.path.basename(lst_l1c[ii]).split('-')[5],'%Y%m%dt%H%M%S')
            fp_dilated = loads(dstmp.attrs['footprint']).buffer(0.25)
            valid_points = points.intersection(fp_dilated)
            for individualpt in valid_points.geoms:
                lons_match.append(individualpt.x)
                lats_match.append(individualpt.y)
                dates.append(date)
        except KeyboardInterrupt:
            raise Exception('stoop')
        except:
            logging.error('%s',traceback.format_exc())
            cpt_undreadable += 1
            unreadable_l1c.append(lst_l1c[ii])
            dmkljslfkjl
        if ii==4:
            pass
    logging.info('unreadable : %s',cpt_undreadable)
    df['lon'] = lons_match
    df['lat'] = lats_match
    df['date'] = dates
    os.makedirs(dirout,exist_ok=True)
    if len(dates)>0:
        fout = os.path.join(dirout,'trackfile-ww3spectra-IW-%s-%s.txt'%(day_to_treat.strftime('%Y%m%d'),product_id))
        df.to_csv(fout,header=False,index=False)
        logging.info('fout : %s',fout)
    else:
        logging.info('no data')

def main():
    root = logging.getLogger()
    if root.handlers:
       for handler in root.handlers:
           root.removeHandler(handler)
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description='trackfileiwWW3')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--day', required=True, help='YYYYMMDD')
    parser.add_argument('--outputdir', required=True, help='directory where to store output')
    parser.add_argument('--productid',
                        help='proudct ID to read (Level-1C IW Ifremer)',
                        required=False, default='B07')
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S',force=True)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S',force=True)
    t0 = time.time()
    day_to_treat = datetime.datetime.strptime(args.day,'%Y%m%d')
    collect_each_matching_locations(day_to_treat=day_to_treat,product_id=args.productid,dirout=args.outputdir)
    elapsed = time.time()
    logging.info('time to do a day: %1.1f seconds',elapsed)

if __name__ == '__main__':
    main()