# uses conda environment gpflow6_0

from memory_profiler import profile

# generic
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
import time

# plotting

from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D 
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import gpflow as gpf
from gpflow.ci_utils import ci_niter, ci_range
from gpflow.utilities import print_summary

from eu_functions import *

# tensorflow
import tensorflow as tf
import argparse

@profile

def gpr_it():
    
    
    parser = argparse.ArgumentParser(description='import vars via c-line')
    parser.add_argument("--mod", default='d6g_h6g_')
    parser.add_argument("--lith", default='l71C')
    parser.add_argument("--tmax", default=15000)
    parser.add_argument("--tmin", default=50)
    parser.add_argument("--place", default="northsea_uk_tight")
    parser.add_argument("--nout", default=60)
    parser.add_argument("--iters", default=3000)
    parser.add_argument("--kernels", default=[2500, 10000, 100, 6000])

    args = parser.parse_args()
    
    ice_model = args.mod
    lith = args.lith
    tmax = int(args.tmax)
    tmin = int(args.tmin)
    place = args.place
    nout = int(args.nout)
    iterations = int(args.iters)
    k1 = int(args.kernels[0])
    k2 = int(args.kernels[1])
    k3 = int(args.kernels[2])
    k4 = int(args.kernels[3])
    
    agemax = round(tmax, -3) + 100
    agemin = round(tmin, -3) - 100
    ages = np.arange(agemin, agemax, 100)[::-1]
    

    locs = {
            'northsea_uk': [-10, 10, 45, 59],
            'northsea_uk_tight': [-5, 10, 50, 55],
           }
    extent = locs[place]

    #import khan dataset
    path = '../data/HOLSEA_2019_uknorthsea.csv'
    df_place = import_rsls(path, tmin, tmax, extent)

    # add zeros at present-day.  
    nout = 50
    df_place = add_presday_0s(df_place, nout)

    #####################  Make xarray template  #######################

    filename = '../data/xarray_template.mat'
    ds_template = xarray_template(filename, ages, extent)

    #####################    Load GIA datasets   #######################

    ds = make_mod(ice_model, lith, ages, extent)

    likelist = []
    namelist = []
    wrsslist = []
    rmselist = []
    wrmselist = []

    ds_load = ds.load().chunk((-1,-1,-1)).interp(lon=ds_template.lon, lat=ds_template.lat).to_dataset()

    #####################    Run Iterative GP Regression   ##################

    def ds_ageselect(ds, row):
        return ds.rsl.interp(age=[row.age]).age.values[0]

    for i, ds_single in ds_load.groupby('modelrun'):

        #interpolate/select priors from GIA model
        df_place['rsl_giaprior'] = df_place.apply(lambda row: ds_select(ds_single, row), axis=1)
        df_place['age_giaprior'] = df_place.apply(lambda row: ds_ageselect(ds_single, row), axis=1)

        #calculate residuals
        df_place['rsl_realresid'] = df_place.rsl - df_place.rsl_giaprior
        df_place['age_realresid'] = df_place.age - df_place.age_giaprior

        # Calculate weighted root mean squared error and weighted residual sum of squares
        df_place['wrss'] = (df_place.age_realresid/df_place.age_er)**2 + (df_place.rsl_realresid/df_place.rsl_er)**2

        wrss = df_place.wrss.sum()

        weights = df_place.rsl_er/df_place.rsl_er.sum()
        rmse = np.sqrt((df_place.rsl_realresid ** 2).sum()/len(df_place))
        wrmse = np.sqrt((df_place.rsl_realresid ** 2/weights).sum()/len(df_place))


        print('number of datapoints = ', df_place.shape)

        ##################	  RUN GP REGRESSION 	#######################
        ##################  --------------------	 ######################


        ds_giapriorinterp, da_zp, ds_priorplusgpr, ds_varp, loglike, m = run_gpr(nout, iterations, ds_single, ages, k1, k2, k3, k4, df_place)
        likelist.append(loglike)
        namelist.append(ds_single.modelrun.values.tolist()[0])
        wrsslist.append(wrss)
        rmselist.append(rmse)
        wrmselist.append(wrmse)

    df_out = pd.DataFrame({'modelrun': namelist,
                 'log_marginal_likelihood': likelist,
                          'weighted residual sum of squares': wrsslist,
                          'root mean squared error': rmselist,
                          'weighted root mean squared error': wrmselist})
    
    df_out.to_csv(f'output/metrics_{place}_{ice_model}{tmax}_{tmin}', index=False)

if __name__ == '__main__':
    gpr_it()