#/usr/bin/env python3 

import pickle
from slict import CachedSlict

import numpy as np
from scipy.optimize import minimize
import cma
from os.path import exists
from scipy.interpolate import UnivariateSpline

with open("data_table.p", 'rb') as f:
    data_table_d = pickle.load(f)

data_table = CachedSlict(data_table_d)

from model import error, filter_trajectory
from model import guess, fix_thin, guess_thin, bounds_thin, bounds_thin_cma
from model import merge_coef, scaling_of_vars
from model import mix_error, mix_bounds_cma, mix_error_direct
from model import scaling_of_mix

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
    y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]

    h_spline = UnivariateSpline(times, heights, k=3, s = 0.00000001)

    def func0(x):
        return mix_error_direct(x, L, c, y0, times, h_spline, mix)
 
    def func1(x):
        if len(x) == 6:
            return error(merge_coef(x, fix_thin), Atwood, v, L, c, y0, times, heights, mix)
        return error(x, Atwood, v, L, c, y0, times, heights, mix)
  
    if (v,c) in results:
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
        'scaling_of_variables' : scaling_of_mix,
        'popsize'   : 64,
#        'maxfevals' : 10000
    }

    mix_res = cma.fmin(func0, np.ones(scaling_of_mix.size), 1.0, mix_opts)

    start[3] = mix_res[0][0]
    start[4] = mix_res[0][1]

    cma_opts = {
        'bounds'  : bounds_thin_cma,
        'tolfun'  : 1.0e-8,
        'scaling_of_variables' : scaling_of_vars, 
        'popsize' : 64,
        'maxfevals' : 100
    }
    res_cma = cma.fmin(func1, start, sigma, cma_opts)

    results[v, c] = (res_cma[0], res_cma[1], res_cma[6], mix_res[0], mix_res[1])

"""
def func(x):
    if len(x) == 6:
        x_full = merge_coef(x, fix_thin)
    else:
        x_full = x

    err = 0.
    count = 0
    for v, c in data_table[:,:,'time'].keys():
        this = data_table[v, c, :]
        times, heights, mix = filter_trajectory(this['time'], this['height2'], this['mixed_mass'], this['extent_mesh'][2])
        mix = 2*(64. - mix)

        L = np.sqrt(2) / this["kmin"]
        Atwood = this['atwood'] * this['g']
        y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]

        err += error(x_full, Atwood, v, L, c, y0, times, heights, mix)
        count += 1
    return err / count  

if (-1,-1) in results:
    start = results[-1,-1][0]
    sigma = 1
else:
    start = guess_thin
    sigma = 5 

#res_cma = cma.fmin(func, start, sigma, cma_opts)
#print("Overall error is {}".format(res_cma[1]))
#print(res_cma[0])
#results[-1, -1] = (res_cma[0], res_cma[1], res_cma[6])
"""

with open("fit_results.p", "wb") as f:
    pickle.dump(results, f)

for v, c in data_table[:,:,'time'].keys():
    res = results[v,c]
    print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res[1]) + str(res[0]))

