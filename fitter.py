#/usr/bin/env python3 

import pickle
from slict import CachedSlict

import numpy as np
from scipy.optimize import minimize
import cma

with open("data_table.p", 'rb') as f:
    data_table_d = pickle.load(f)

data_table = CachedSlict(data_table_d)

from model import model, error, filter_trajectory

guess = [1.0, 1.0, 113.0, 1.0, 1./(2*np.pi), 4.0, 1.0]
bounds = ((1, 1), (0, 10), (0,400), (0.01, 100), (1./(2*np.pi), 1./(2*np.pi)), (0.01, 100), (1,1))

guess_thin = [1.0, 113.0, 1.0, 4.0]
bounds_thin     = ((0.01, 10), (0.01,400), (0.01, 100), (0.01, 100))
bounds_thin_cma = [(0.01, 0.01, 0.01, 0.01),
                   (10, 400, 100, 100) ]

fix_thin = [1.0, 1./(2.*np.pi), 1.0]

def merge_coef(var, fix):
  return [fix[0], var[0], var[1], var[2], fix[1], var[3], fix[2]]

results = {}

for v, c in data_table[:,:,'time'].keys():

    this = data_table[v, c, :]

    times, heights = filter_trajectory(this['time'], this['height2'], this['extent_mesh'][2])


    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]


    T, H, V, At = model(guess, Atwood, v, L, c, y0, times)
    rmse_i = error(guess, Atwood, v, L, c, y0, times, heights)
  
    def func(x):
        if len(x) == 4:
            return error(merge_coef(x, fix_thin), Atwood, v, L, c, y0, times, heights)
        return error(x, Atwood, v, L, c, y0, times, heights)
   
    opts = {'maxiter': 1000, 'disp': True}
    res_l = minimize(func, guess_thin, bounds=bounds_thin, method='SLSQP', options=opts, tol=1.e-12)
    rmse_l = error(merge_coef(res_l.x, fix_thin), Atwood, v, L, c, y0, times, heights)

    cma_opts = {
        'bounds' : bounds_thin_cma
    }
    res_cma = cma.fmin(func, guess_thin, 5, cma_opts)

    results[v, c] = (res_cma[0], res_cma[1])

def func(x):
    if len(x) == 4:
        x_full = merge_coef(x, fix_thin)
    else:
        x_full = x

    err = 0.
    count = 0
    for v, c in data_table[:,:,'time'].keys():
        this = data_table[v, c, :]
        times, heights = filter_trajectory(this['time'], this['height2'], this['extent_mesh'][2])

        L = np.sqrt(2) / this["kmin"]
        Atwood = this['atwood'] * this['g']
        y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]

        err += error(x_full, Atwood, v, L, c, y0, times, heights)
        count += 1
    return err / count  

res_cma = cma.fmin(func, guess_thin, 5, cma_opts)
print("Overall error is {}".format(res_cma[1]))
print(res_cma[0])

with open("fit_results.p", "wb") as f:
    pickle.dump(results, f)

for v, c in data_table[:,:,'time'].keys():
    res = results[v,c]
    print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res[1]) + str(res[0]))

