#/usr/bin/env python3 

import pickle
from slict import CachedSlict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


with open("data_table.p", 'rb') as f:
    data_table_d = pickle.load(f)

data_table = CachedSlict(data_table_d)

from model import model, error


for v, c in data_table[:,:,'time'].keys():
    this = data_table[v, c, :]
    plt.plot(this['time'], this['height'])

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]


    coeffs = [1.0, 113.0, 1.0, 1.0, 1./(2*np.pi), 4.0, 1.0]
    T, H, V, At = model(coeffs, Atwood, v, L, c, y0, this['time'][0], this['time'][-1], this['time'][1] - this['time'][0])
    rmse_i = error(coeffs, Atwood, v, L, c, y0, this['time'], this['height'])
    plt.plot(this['time'], H)
  
    
    bnds = ((1, 1), (0,400), (0, 10), (0, 10), (1./(2*np.pi), 1./(2*np.pi)), (0, 10), (0,10))
    opts = {'maxiter': 1000, 'disp': True}
    res = minimize(lambda x: error(x, Atwood, v, L, c, y0, this['time'], this['height']),
                   coeffs, bounds=bnds, method='SLSQP', options=opts, tol=1.e-12)
    rmse_f = error(res.x, Atwood, v, L, c, y0, this['time'], this['height'])

    print(v, c, rmse_i, rmse_f)
    print(coeffs)
    print(res.x)

    T, H, V, At = model(res.x, Atwood, v, L, c, y0, this['time'][0], this['time'][-1], this['time'][1] - this['time'][0])

    plt.plot(this['time'], H)

plt.savefig('foo.png')

