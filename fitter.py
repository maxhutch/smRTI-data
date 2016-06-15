#/usr/bin/env python3 

import pickle
from slict import CachedSlict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.interpolate import UnivariateSpline



with open("data_table.p", 'rb') as f:
    data_table_d = pickle.load(f)

data_table = CachedSlict(data_table_d)

from model import model, error, filter_trajectory

guess = [1.0, 1.0, 113.0, 1.0, 1./(2*np.pi), 4.0, 1.0]
bounds = ((1, 1), (0, 10), (0,400), (0.01, 100), (1./(2*np.pi), 1./(2*np.pi)), (0.01, 100), (1,1))

guess_thin = [1.0, 113.0, 1.0, 4.0]
bounds_thin = ((0, 10), (0,400), (0.01, 100), (0.01, 100))
fix_thin = [1.0, 1./(2.*np.pi), 1.0]

def merge_coef(var, fix):
  return [fix[0], var[0], var[1], var[2], fix[1], var[3], fix[2]]

def accept_test(f_new, x_new, f_old, x_old):
    for i in range(len(bounds)):
        if x_new[i] < bounds[i][0] or x_new[i] > bounds[i][1]:
            return False
    return True

results = {}

for v, c in data_table[:,:,'time'].keys():
    fig, axs = plt.subplots(1, 2, figsize=(8,8))

    this = data_table[v, c, :]

    times, heights = filter_trajectory(this['time'], this['height'], this['extent_mesh'][2])

    axs[0].plot(times, heights)

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]


    T, H, V, At = model(guess, Atwood, v, L, c, y0, times)
    rmse_i = error(guess, Atwood, v, L, c, y0, times, heights)
    axs[0].plot(times, H)
  
    def func(x):
        if len(x) == 4:
            return error(merge_coef(x, fix_thin), Atwood, v, L, c, y0, times, heights)
        return error(x, Atwood, v, L, c, y0, times, heights)
   
    opts = {'maxiter': 1000, 'disp': True}
    res_l = minimize(func, guess_thin, bounds=bounds_thin, method='SLSQP', options=opts, tol=1.e-12)
    rmse_l = error(merge_coef(res_l.x, fix_thin), Atwood, v, L, c, y0, times, heights)

    margs = {
        "method": 'SLSQP',
        "bounds": bounds,
        "options": opts,
    }
    """
    res_n = basinhopping(func, res_l.x, 
                minimizer_kwargs=margs, disp=True, 
                accept_test = accept_test,
                T=0.001, 
                niter_success=50, 
                niter=1000)
    """
    res_n = differential_evolution(func, bounds_thin, disp=True, polish=True, tol=0.0001)
    rmse_n = error(merge_coef(res_n.x, fix_thin), Atwood, v, L, c, y0, times, heights)

    print(v, c, rmse_i, rmse_l, rmse_n)
    print(guess_thin)
    print(res_l.x)
    print(res_n.x)
    results[v, c] = (res_n.x, res_n.fun)

    T, H, V, At = model(merge_coef(res_n.x, fix_thin), Atwood, v, L, c, y0, times)
    spl = UnivariateSpline(times, heights, k=3, s = 0.00000001).derivative()

    axs[0].plot(times, H)
    axs[1].plot(times, V / np.sqrt(Atwood * L))
    axs[1].plot(times, spl(times)/ np.sqrt(Atwood * L))
    axs[1].axhline(1./np.sqrt(np.pi))
    plt.savefig('H-{}-{}.png'.format(v,c))


for v, c in list(results.keys()):
    res = results[v,c]
    print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res[1]) + str(res[0]))

