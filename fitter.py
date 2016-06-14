#/usr/bin/env python3 

import pickle
from slict import CachedSlict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution


with open("data_table.p", 'rb') as f:
    data_table_d = pickle.load(f)

data_table = CachedSlict(data_table_d)

from model import model, error

guess = [1.0, 1.0, 113.0, 1.0, 1./(2*np.pi), 4.0, 1.0]
bounds = ((1, 1), (0, 10), (0,400), (0.01, 100), (1./(2*np.pi), 1./(2*np.pi)), (0.01, 100), (1,1))

def accept_test(f_new, x_new, f_old, x_old):
    for i in range(len(bounds)):
        if x_new[i] < bounds[i][0] or x_new[i] > bounds[i][1]:
            return False
    return True

results = {}

for v, c in data_table[:,:,'time'].keys():
    plt.figure()
    this = data_table[v, c, :]
    plt.plot(this['time'], this['height'])

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]


    T, H, V, At = model(guess, Atwood, v, L, c, y0, this['time'])
    rmse_i = error(guess, Atwood, v, L, c, y0, this['time'], this['height'])
    plt.plot(this['time'], H)
  
    func = lambda x: error(x, Atwood, v, L, c, y0, this['time'], this['height'])
   
    opts = {'maxiter': 1000, 'disp': True}
    res_l = minimize(func, guess, bounds=bounds, method='SLSQP', options=opts, tol=1.e-12)
    rmse_l = error(res_l.x, Atwood, v, L, c, y0, this['time'], this['height'])

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
    res_n = differential_evolution(func, bounds, popsize=64, disp=True)
    rmse_n = error(res_n.x, Atwood, v, L, c, y0, this['time'], this['height'])

    print(v, c, rmse_i, rmse_l, rmse_n)
    print(guess)
    print(res_l.x)
    print(res_n.x)
    results[v, c] = res_n.x

    T, H, V, At = model(res_n.x, Atwood, v, L, c, y0, this['time'])

    plt.plot(this['time'], H)
    plt.savefig('H-{}-{}.png'.format(v,c))


for v, c in list(results.keys()):
    print("V={},D={},C=".format(v, c) + str(results[v,c]))

