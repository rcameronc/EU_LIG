# uses conda environment gpflow6_0

from memory_profiler import profile

# generic
import numpy as np
import xarray as xr
import time

# plotting

from matplotlib import pyplot as plt

import gpflow as gpf
from gpflow.ci_utils import ci_niter, ci_range
from gpflow.utilities import print_summary

from synth_eu_functions import *

# tensorflow
import tensorflow as tf
import argparse

@profile

def gpr_it():


    parser = argparse.ArgumentParser(description='import vars via c-line')
    parser.add_argument("--mod", default='d6g_h6g_')
    parser.add_argument("--lith", default='l71C')
    parser.add_argument("--um", default='p3')
    parser.add_argument("--lm", default='3')
    parser.add_argument("--tmax", default=15000)
    parser.add_argument("--tmin", default=50)
    parser.add_argument("--place", default="arctic")
    parser.add_argument("--nout", default=80)
    parser.add_argument("--kernels", default=[500, 10000, 5000, 10000])

    args = parser.parse_args()

    ice_model = args.mod
    lith = args.lith
    um = args.um        # must be string not int
    lm = args.lm        # must be string not int
    tmax = int(args.tmax)
    tmin = int(args.tmin)
    place = args.place
    nout = int(args.nout)
    k1 = int(args.kernels[0])
    k2 = int(args.kernels[1])
    k3 = int(args.kernels[2])
    k4 = int(args.kernels[3])

    modelrun = ice_model + lith + '_um' + um + '_lm' + lm


    agemax = round(tmax, -3) + 100
    agemin = round(tmin, -3)
    ages = np.arange(agemin, agemax, 100)[::-1]


    locs = {
            'northsea_uk': [-10, 10, 45, 59],
            'northsea_uk_tight': [-5, 10, 50, 55],
            'fennoscandia': [-15, 50, 45, 75],
            'norway': [0, 50, 50, 75],
            'europe_arctic': [-15, 88, 45, 85],
            'arctic': [15, 88, 40, 85],
            'english_channel': [-5, 2, 48, 52],
            'denmark_netherlands': [3, 12, 52, 56],
            'north_england':[-5, 2.5, 53, 56],
    }
    extent = locs[place]

    ##Get Norway data sheet from Google Drive
    sheet = 'Norway_isolation'
    path = '../../data/holocene_fennoscandian_data_05132020.csv'
    df_nor = load_nordata_fromsheet(sheet, path, fromsheet=False)
    
    # Get Barnett data
    path = f'../../data/SciAdv_Barnett_SupplementaryDataset_S1_long.csv'
    df_barnett = import_barnett2020(path)

    #import khan dataset
    path = '../../data/GSL_LGM_120519_.csv'
    df_place = import_rsls(path, df_nor, df_barnett, tmin, tmax, extent)

    # add zeros at present-day.
    nout = 50
    df_place = add_presday_0s(df_place, nout)

    #####################  Make xarray template  #######################

    filename = '../../data/xarray_template.mat'
    ds_template = xarray_template(filename, ages, extent)

    #####################    Load GIA datasets   #######################

    path = f'../../data/{ice_model}/output_{ice_model}{lith}'

    ds = make_mod(path, ice_model, lith, ages, extent)
    ds = ds.load().chunk((-1,-1,-1))
    ds = ds.interp(lon=ds_template.lon, lat=ds_template.lat).to_dataset()

    # choose arbitrary true model
    ds_true = ds.sel(modelrun=['glac1d_l96C_ump4_lm10'])

    # choose prior model
    ds = ds.sel(modelrun=modelrun)

    likelist = []
    namelist = []
    wrsslist = []
    rmselist = []
    wrmselist = []


    #####################   Interpolate at datapoints   ##################

    #interpolate/select priors from GIA model
    df_place['rsl_giatrue'] = df_place.apply(lambda row: ds_select(ds_true, row), axis=1)
    df_place['rsl_giaprior'] = df_place.apply(lambda row: ds_select(ds, row), axis=1)
    df_place['age_giaprior'] = df_place.apply(lambda row: ds_ageselect(ds, row), axis=1)

    #calculate residuals
    # df_place['rsl_realresid'] = df_place.rsl - df_place.rsl_giaprior
    df_place['rsl_realresid'] = df_place.rsl_giatrue - df_place.rsl_giaprior
    df_place['age_realresid'] = df_place.age - df_place.age_giaprior

    # Calculate weighted root mean squared error and weighted residual sum of squares
    df_place['wrss'] = (df_place.age_realresid/df_place.age_er)**2 + (df_place.rsl_realresid/df_place.rsl_er)**2

    wrss = df_place.wrss.sum()

    weights = df_place.rsl_er/df_place.rsl_er.sum()
    rmse = np.sqrt((df_place.rsl_realresid ** 2).sum()/len(df_place))
    wrmse = np.sqrt((df_place.rsl_realresid ** 2/weights).sum()/len(df_place))

    print('number of datapoints = ', df_place.shape)

        ##################	  RUN GP REGRESSION 	#######################


    ds_giapriorinterp, da_zp, ds_priorplusgpr, ds_varp, loglike, m, df_place = run_gpr(nout, ds, ages, k1, k2, k3, k4, df_place)

    name = ds.modelrun.values.tolist()

    path = f'output/{place}_{name}_{ages[0]}_{ages[-1]}'
    da_zp.to_netcdf(path + '_dazp')
    da_zp['model'] = name
    da_zp['likelihood'] = loglike

    ds_giapriorinterp.to_netcdf(path + '_giaprior')
    ds_giapriorinterp['model'] = name
    ds_giapriorinterp['likelihood'] = loglike

    ds_priorplusgpr.to_netcdf(path + '_posterior')
    ds_priorplusgpr['model'] = name
    ds_priorplusgpr['likelihood'] = loglike

    ds_varp.to_netcdf(path + '_gpvariance')
    ds_varp['model'] = name
    ds_varp['likelihood'] = loglike

    # write hyperparameters to csv

    k1k2 = [[k.lengthscales.numpy(), k.variance.numpy()] for _, k in enumerate(m.kernel.kernels[0].kernels)]
    k3k4 = [[k.lengthscales.numpy(), k.variance.numpy()] for _, k in enumerate(m.kernel.kernels[1].kernels)]
    k5 = [[np.nan,m.kernel.kernels[2].variance.numpy()]]

    cols = ['lengthscale', 'variance']
    idx = ['k1', 'k2', 'k3', 'k4', 'k5']

    df_params = pd.DataFrame(np.concatenate([k1k2, k3k4, k5]), columns=cols, index=idx)

    df_params['model'] = name
    df_params['likelihood'] = loglike
    df_params['rmse'] = rmse
    df_params.to_csv(path + '_hyperparams.csv', index=True)


    df_out = pd.DataFrame({'modelrun': name,
                 'log_marginal_likelihood': [loglike],
                          'weighted residual sum of squares': [wrss],
                          'root mean squared error': [rmse],
                          'weighted root mean squared error': [wrmse]})

    df_out.to_csv(path + '_metrics.csv', index=False)

if __name__ == '__main__':
    gpr_it()
