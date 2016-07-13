#/usr/bin/env python3 

import pickle
from slict import CachedSlict

import numpy as np
from scipy.optimize import minimize
import cma
from os.path import exists
from scipy.interpolate import UnivariateSpline

from scipy.optimize import basinhopping, minimize

with open("data_table.p", 'rb') as f:
    data_table_d = pickle.load(f)

data_table = CachedSlict(data_table_d)

from model import filter_trajectory, error
from model import dyn_error, guess_dyn, bounds_dyn, exp_dyn, scaling_dyn, bounds_dyn_t
from model import mix_error, guess_mix, bounds_mix, exp_mix

if exists("fit_results.p"):
    with open("fit_results.p", "rb") as f:
        results = pickle.load(f)
else:
    exit()

override = {}
override[0.0016, 0.0016] = [ 0.000,   61.079,  1.326,  1.026]

for v, c in override:

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
    m_spline = UnivariateSpline(times, mix, k=3, s = 0.00000001)

    if (v,c) in results:
        start_mix = results[v,c]["C_mix"]
        start_dyn = override[v,c]
#        if results[v,c]["E_dyn"] < .1:
#            continue
#        else:
#            start_dyn = guess_dyn 
    else:
        results[v,c] = {}
        start_mix = guess_mix
        start_dyn = guess_dyn

    def func0(x):
        return mix_error(exp_mix(x), L, c, this['delta'], times, h_spline, mix)

    def accept_test_mix(f_new, x_new, f_old, x_old):
        res = x_new[0] > bounds_mix[0][0] and x_new[0] < bounds_mix[0][1]
        return bool(res)

    mix_res = basinhopping(
                func0, 
                start_mix, 
                disp=False, 
                minimizer_kwargs={'method' : 'SLSQP', 'bounds': bounds_mix},
                accept_test=accept_test_mix,
              )

    results[v,c]["C_mix"] = mix_res.x
    results[v,c]["E_mix"] = mix_res.fun
 
    def func1(x):
        #return dyn_error(exp_dyn(x), Atwood, v, L, y0, times, heights, m_spline)
        return error(exp_dyn(x), exp_mix(results[v,c]["C_mix"]), Atwood, v, L, c, this['delta'], y0, times, heights, mix)
   
    for i in range(len(start_dyn)):
        start_dyn[i] = max(start_dyn[i], bounds_dyn[0][i])
        start_dyn[i] = min(start_dyn[i], bounds_dyn[1][i])

    cma_opts = {
        'bounds'  : bounds_dyn,
        'tolfun'  : 1.0e-9,
#        'fixed_variables' : {3:mix_res.x[0]},
        'scaling_of_variables' : scaling_dyn, #get_scaling(bounds_thin_cma), 
        'popsize' : 256,
#        'verbose' : 0,
#        'tolfacupx' : 1.0e12,
#        'maxfevals' : 256
    }
    res_cma = cma.fmin(func1, start_dyn, 1.0, cma_opts)

    res_slsqp = minimize(func1, res_cma[0], method='SLSQP', bounds=bounds_dyn_t, tol=1.0e-12, options={'ftol': 1.0e-12})

    results[v,c]["C_dyn"] = res_slsqp.x
    results[v,c]["E_dyn"] = res_slsqp.fun

    with open("fit_results.p", "wb") as f:
        pickle.dump(results, f)

for v, c in data_table[:,:,'time'].keys():
    res = results[v,c]
    print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res[1]) + str(res[0]))

