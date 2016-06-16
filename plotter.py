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

with open("fit_results.p", "rb") as f:
    results = pickle.load(f)

for v, c in data_table[:,:,'time'].keys():
    this = data_table[v, c, :]

    times, heights = filter_trajectory(this['time'], this['height2'], this['extent_mesh'][2])

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]
    
    fig, axs = plt.subplots(1, 2, figsize=(8,8))
    axs[0].plot(times, heights)

    T, H, V, At = model(guess, Atwood, v, L, c, y0, times)
    axs[0].plot(times, H)

    T, H, V, At = model(merge_coef(results[v,c][0], fix_thin), Atwood, v, L, c, y0, times)
    spl = UnivariateSpline(times, heights, k=3, s = 0.00000001).derivative()

    axs[0].plot(times, H)
    axs[1].plot(times, V / np.sqrt(Atwood * L))
    axs[1].plot(times, spl(times)/ np.sqrt(Atwood * L))
    axs[1].axhline(1./np.sqrt(np.pi))
    plt.savefig('H-{}-{}.png'.format(v,c))

    res = results[v,c]
    print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res[1]) + str(res[0]))

Grashof = []
Schmidt = []
Err = []
C1 = []
C2 = []
C3 = []
C5 = []
for v, c in data_table[:,:,'time'].keys():
    this = data_table[v,c, :]
    Grashof.append(this['atwood']*this['g']/(v**2 * this['kmin']**3))
    Schmidt.append(v/c)
    Err.append(results[v,c][1])
    C1.append(results[v,c][0][0])
    C2.append(results[v,c][0][1])
    C3.append(results[v,c][0][2])
    C5.append(results[v,c][0][3])


def plot_scatter(x, y, c, name):
  plt.figure()
  plt.scatter(x, y, c=c, s=400)
  ax = plt.gca()
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel('Grashof')
  ax.set_ylabel('Schmidt');
  plt.colorbar()
  plt.savefig("{}-vs-Gr-Sc.png".format(name))

plot_scatter(Grashof, Schmidt, Err, "Err")
plot_scatter(Grashof, Schmidt, C1, "C1")
plot_scatter(Grashof, Schmidt, C2, "C2")
plot_scatter(Grashof, Schmidt, C3, "C3")
plot_scatter(Grashof, Schmidt, C5, "C5")

