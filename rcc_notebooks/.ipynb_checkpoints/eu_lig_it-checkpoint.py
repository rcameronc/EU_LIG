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

from eu_functions import *

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
    parser.add_argument("--kernels", default=[100, 10000, 5000, 10000])

    args = parser.parse_args()

    ice_model = args.mod
    lith = args.lith
    um = args.um        # must be string not int
    lm = args.lm        # must be string not int
    tmax = int(args.tmax)
    tmin = int(args.tmin)
    place = args.place
    nout = int(args.nout)
    k1len = int(args.kernels[0])
    k2len = int(args.kernels[1])
    k3len = int(args.kernels[2])
    k4len = int(args.kernels[3])

    modelrun = ice_model + lith + '_um' + um + '_lm' + lm


    agemax = round(tmax, -3) + 100
    agemin = round(tmin, -3)
    ages = np.arange(agemin, agemax, 100)[::-1]


    locs = {
            'northsea_uk': [-10, 10, 45, 59],
            'northsea_uk_tight': [-5, 10, 50, 55],
            'fennoscandia': [-15, 50, 45, 75],
            'norway': [0, 50, 50, 78],
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
    ds = ds.sel(modelrun=modelrun)
    ds = ds.load().chunk((-1,-1,-1))
    ds = ds.interp(lon=ds_template.lon, lat=ds_template.lat).to_dataset()

    likelist = []
    namelist = []
    wrsslist = []
    rmselist = []
    wrmselist = []


    #####################   Interpolate at datapoints   ##################

    #interpolate/select priors from GIA model
    df_place['rsl_giaprior'] = df_place.apply(lambda row: ds_select(ds, row), axis=1)
    df_place['age_giaprior'] = df_place.apply(lambda row: ds_ageselect(ds, row), axis=1)

    #calculate residuals
    df_place['rsl_realresid'] = df_place.rsl - df_place.rsl_giaprior
    df_place['age_realresid'] = df_place.age - df_place.age_giaprior

    # Calculate weighted root mean squared error and weighted residual sum of squares
    df_place['wrss'] = (df_place.age_realresid/df_place.age_er)**2 + (df_place.rsl_realresid/df_place.rsl_er)**2

    wrss = df_place.wrss.sum()

    weights = df_place.rsl_er/df_place.rsl_er.sum()
    rmse = np.sqrt((np.square(df_place.rsl_realresid)/len(df_place)).sum())
    
    
    wrmse = np.sqrt((df_place.rsl_realresid ** 2/weights).sum()/len(df_place))

    print('number of datapoints = ', df_place.shape)

    ##################	  RUN GP REGRESSION 	#######################

    # Input space, rsl normalized to zero mean, unit variance
    X = np.stack((df_place.lon, df_place.lat, df_place.age), 1)
    
#     RSL = df_place.rsl_realresid.values.reshape(-1,1) 

    RSL = normalize(df_place.rsl_realresid)

    #define kernels  with bounds
    k1 = HaversineKernel_Matern32(active_dims=[0, 1], lengthscales=1000)
    k1.lengthscales = bounded_parameter(10, 100000, k1len) 
#     k1.variance = bounded_parameter(0.01, 10000, 2)

    k2 = gpf.kernels.Matern32(active_dims=[2], lengthscales=1) 
    k2.lengthscales = bounded_parameter(0.1, 100000, k2len)
#     k2.variance = bounded_parameter(0.01, 10000, 1)

    k3 = HaversineKernel_Matern32(active_dims=[0, 1], lengthscales=1)
    k3.lengthscales = bounded_parameter(1000, 20000, k3len)  
#     k3.variance = bounded_parameter(0.01, 10000, 2)

    k4 = gpf.kernels.Matern52(active_dims=[2], lengthscales=1) 
    k4.lengthscales = bounded_parameter(1, 100000, k4len)
    k4.variance = bounded_parameter(0.01, 10000, 1)

    k5 = gpf.kernels.White(active_dims=[0, 1, 2])
#     k5.variance = bounded_parameter(0.01, 10000, 1)
    
    k6 = gpf.kernels.Constant(0.00001, active_dims=[2])

    kernel = (k1 * k2) + (k3 * k4) + k6 #  + (k3 * k4)# 

    ##################	  BUILD AND TRAIN MODELS 	#######################
    noise_variance = (df_place.rsl_er.ravel())**2 + 1e-6

    m = GPR_new((X, RSL), kernel=kernel, noise_variance=noise_variance) 
    
    #Sandwich age of each lat/lon to enable gradient calculation
    lonlat = df_place[['lon', 'lat']]
    agetile = np.stack([df_place.age - 10, df_place.age, df_place.age + 10], axis=-1).flatten()
    xyt_it = np.column_stack([lonlat.loc[lonlat.index.repeat(3)], agetile])

    #hardcode indices for speed (softcoded alternative commented out)
    indices = np.arange(1, len(df_place)*3, 3)
    # indices = np.where(np.in1d(df_place.age, agetile))[0]
    
    # First optimize without age errs to get slope
    min_kwargs = ({'method':'L-BFGS-B', 'jac':True})
    
    tf.print('___First optimization___')
    opt = Optimize()
    closure = m.training_loss 
    variables = m.trainable_variables
    
    opt_logs = opt.basinhopping(closure=m.training_loss, variables=m.trainable_variables, minimizer_kwargs=min_kwargs)
    
    # Calculate posterior at training points + adjacent age points
    mean, _ = m.predict_f(xyt_it)

    # make diagonal matrix of age slope at training points
    Xgrad = np.diag(np.gradient(mean.numpy(), axis=0)[indices][:,0])

    # multipy age errors by gradient 
    Xnigp = np.diag(Xgrad @ np.diag((df_place.age_er/2)**2) @ Xgrad.T)    
    
    m = GPR_new((X, RSL), kernel=kernel, noise_variance=noise_variance + Xnigp)

    #reoptimize
    tf.print('___Second optimization___')
#     opt = tf.optimizers.Adam(learning_rate)

    opt = Optimize()
    closure = m.training_loss 
    variables = m.trainable_variables

    opt_logs = opt.basinhopping(closure=closure, variables=variables, minimizer_kwargs=min_kwargs)


    ##################	  INTERPOLATE MODELS 	#######################
    ##################  --------------------	 ######################
    # output space
    da_zp, da_varp = predict_post_f(nout, ages, ds, df_place, m)

    #interpolate all models onto GPR grid
    ds_giapriorinterp  = interp_likegpr(ds, da_zp)

    # add total prior RSL back into GPR
    ds_priorplusgpr = da_zp + ds_giapriorinterp
    ds_varp = da_varp.to_dataset(name='rsl')
    ds_zp = da_zp.to_dataset(name='rsl')

        
    #Calculate data-model misfits & GPR vals at RSL data locations
    df_place['gpr_posterior'] = df_place.apply(lambda row: ds_select(ds_priorplusgpr, row), axis=1)
    df_place['gprpost_std'] = df_place.apply(lambda row: ds_select(ds_varp, row), axis=1)
    df_place['gpr_diff'] = df_place.apply(lambda row: row.rsl - row.gpr_posterior, axis=1)
    df_place['diffdiv'] = df_place.gpr_diff / df_place.rsl_er

    likelihood = m.log_marginal_likelihood().numpy()
    
        
    ################ END REGRESSION ########################
        
#     ds_giapriorinterp, da_zp, ds_priorplusgpr, ds_varp, loglike, m, df_place = run_gpr(nout, ds, ages, k1, k2, k3, k4, df_place)

    name = ds.modelrun.values.tolist()

    path = f'output/{place}_{name}_{ages[0]}_{ages[-1]}'
    ds_zp.to_netcdf(path + '_dazp')
    ds_zp['model'] = name
    ds_zp['likelihood'] = likelihood

    ds_giapriorinterp.to_netcdf(path + '_giaprior')
    ds_giapriorinterp['model'] = name
    ds_giapriorinterp['likelihood'] = likelihood

    ds_priorplusgpr.to_netcdf(path + '_posterior')
    ds_priorplusgpr['model'] = name
    ds_priorplusgpr['likelihood'] = likelihood

    ds_varp.to_netcdf(path + '_gpvariance')
    ds_varp['model'] = name
    ds_varp['likelihood'] = likelihood

    # write hyperparameters to csv

    k1k2 = [[k.lengthscales.numpy(), k.variance.numpy()] for _, k in enumerate(m.kernel.kernels[0].kernels)]
    k3k4 = [[k.lengthscales.numpy(), k.variance.numpy()] for _, k in enumerate(m.kernel.kernels[1].kernels)]
#     k4 = [[m.kernel.kernels[1].lengthscales.numpy(), m.kernel.kernels[1].variance.numpy()]]

#     k5 = [[np.nan,m.kernel.kernels[2].variance.numpy()]

    k6 = [[np.nan,m.kernel.kernels[2].variance.numpy()]]


    cols = ['lengthscale', 'variance']
#     idx = ['k1', 'k2', 'k3', 'k4', 'k5']
#     idx = ['k1', 'k2', 'k4', 'k6']
    ks = ['k1', 'k2','k3', 'k4', 'k6']


    df_params = pd.DataFrame(np.concatenate([k1k2, k3k4, k6]), columns=cols)
#     df_params = pd.DataFrame(np.concatenate([k1k2, k4, k6]), columns=cols, index=idx)


    df_params['model'] = name
    df_params['likelihood'] = likelihood
    df_params['rmse'] = rmse
    df_params['k'] = ks
    df_params.to_csv(path + '_hyperparams.csv', index=False)


    df_out = pd.DataFrame({'modelrun': name,
                 'log_marginal_likelihood': [likelihood],
                          'weighted residual sum of squares': [wrss],
                          'root mean squared error': [rmse],
                          'weighted root mean squared error': [wrmse]})

    df_out.to_csv(path + '_metrics.csv', index=False)

if __name__ == '__main__':
    gpr_it()
