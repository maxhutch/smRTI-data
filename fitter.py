#/usr/bin/env python3 

import pickle
from slict import CachedSlict

import numpy as np
from scipy.optimize import minimize
import cma
from os.path import exists
from scipy.interpolate import UnivariateSpline

from scipy.optimize import basinhopping

with open("data_table.p", 'rb') as f:
    data_table_d = pickle.load(f)

data_table = CachedSlict(data_table_d)

from model import error, filter_trajectory
from model import guess, fix_thin, guess_thin, bounds_thin, bounds_thin_cma
from model import merge_coef
from model import mix_error, mix_bounds_cma, mix_error
from model import get_scaling, thin_scaling

bounds_cma = [(1.0, 0.01, 0.01, 0.01, 1./(2*np.pi), 0.01, 1.0),
              (1.0,   10,  400,  100, 1./(2*np.pi), 100., 1.0) ]

if exists("fit_results.p"):
    with open("fit_results.p", "rb") as f:
        results = pickle.load(f)
else:
    results = {}

for v, c in data_table[:,:,'time'].keys():

    this = data_table[v, c, :]

    times, heights, mix = filter_trajectory(this['time'], this['height2'], this['mixed_mass'], this['extent_mesh'][2])
    mix = 2*(64 - mix)

    if len(times) < len(this['time']):
        print("TRUNC: {} {} stopped at {}".format(v, c, times[-1]))

    if len(times) < 16:
        continue

    print("#========================")
    print("# MINIMIZING nu = {}, D = {}".format(v, c))
    print("#========================")

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0.]

    h_spline = UnivariateSpline(times, heights, k=3, s = 0.00000001)

    def func0(x):
        return mix_error([x[0], 1.0], L, c, this['delta'], times, h_spline, mix)
 
    def func1(x):
        if len(x) == 5:
            return error(merge_coef(x, fix_thin), Atwood, v, L, c, this['delta'], y0, times, heights, mix)
        return error(x, Atwood, v, L, c, this['delta'], y0, times, heights, mix)
  
    if (v,c) in results:
        #if results[v,c][1] < 0.1:
        #    continue
        start = results[v,c][0]
        sigma = 1
    else:
        start = guess_thin
        sigma = 1
   
    #opts = {'maxiter': 1000, 'disp': True}
    #res_l = minimize(func, start, bounds=bounds_thin, method='SLSQP', options=opts, tol=1.e-12)
    #rmse_l = error(merge_coef(res_l.x, fix_thin), Atwood, v, L, c, y0, times, heights, mix)

    mix_opts = {
        'bounds'    : mix_bounds_cma,
#        'tolfun'    : 1.0e-8,
        'scaling_of_variables' : get_scaling(mix_bounds_cma),
        'popsize'   : 256,
#        'maxfevals' : 10000
    }

    #mix_res = cma.fmin(func0, [4.0,], 1.0, mix_opts)
    def accept_test(f_new, x_new):
        return x_new[0] > 0.1

    mix_res = basinhopping(func0, [start[3],], disp=False, minimizer_kwargs={'bounds':[(0.1,5.0),]})

    #start[3] = mix_res[0][0]
    start[3] = mix_res.x[0]
    bounds_thin_cma[0][3] = start[3] *  .9
    bounds_thin_cma[1][3] = start[3] * 1.1

    for i in range(len(start)):
        start[i] = max(start[i], bounds_thin_cma[0][i])
        start[i] = min(start[i], bounds_thin_cma[1][i])

    cma_opts = {
        'bounds'  : bounds_thin_cma,
        'tolfun'  : 1.0e-8,
#        'fixed_variables' : {3:mix_res.x[0]},
        'scaling_of_variables' : thin_scaling, #get_scaling(bounds_thin_cma), 
        'popsize' : 256,
#        'verbose' : 0,
#        'tolfacupx' : 1.0e12,
#        'maxfevals' : 256
    }
    res_cma = cma.fmin(func1, start, sigma, cma_opts)

    results[v, c] = (res_cma[0], res_cma[1], res_cma[6], [mix_res.x[0], 1.0], mix_res.fun)

    with open("fit_results.p", "wb") as f:
        pickle.dump(results, f)

for v, c in data_table[:,:,'time'].keys():
    res = results[v,c]
    print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res[1]) + str(res[0]))

