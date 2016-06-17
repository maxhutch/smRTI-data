#/usr/bin/env python3 

import pickle
from slict import CachedSlict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.interpolate import UnivariateSpline

img_format = 'png'


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
    results_in = pickle.load(f)

results_d = {}
for v, c in results_in.keys():
    for i in range(4):
        results_d[v, c, 'coef', i] = results_in[v,c][0][i]
        results_d[v, c, 'std', i] = results_in[v,c][2][i]
    results_d[v, c, 'err', 0] = results_in[v, c][1]

results = CachedSlict(results_d)

for v, c in data_table[:,:,'time'].keys():
    this = data_table[v, c, :]

    times, heights = filter_trajectory(this['time'], this['height2'], this['extent_mesh'][2])
    #times = this['time']; heights = this['height2']

    if heights.size < 4:
        continue

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]
    
    fig, axs = plt.subplots(1, 2, figsize=(8,8))
    axs[0].plot(times, heights)

    T, H, V, At = model(guess, Atwood, v, L, c, y0, times)
    axs[0].plot(times, H)

    T, H, V, At = model(merge_coef(results[v,c,'coef',:].values(), fix_thin), Atwood, v, L, c, y0, times)
    spl = UnivariateSpline(times, heights, k=3, s = 0.00000001).derivative()

    axs[0].plot(times, H)
    axs[1].plot(times, V / np.sqrt(Atwood * L))
    axs[1].plot(times, spl(times)/ np.sqrt(Atwood * L))
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Height")
    axs[1].axhline(1./np.sqrt(np.pi))
    axs[1].axhline(np.sqrt(Atwood * L) / (results[v,c,'coef',1] * v))
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")
    plt.savefig('H-{}-{}.{}'.format(v,c,img_format))

    res = results[v,c,:,:]
    print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res['err',0]) + str(res['coef',:].values()))
    print(" >> S=" + str(res['std',:].values()))

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
    rerror = results[v,c,'err',0] / min(24.0, np.max(this['height2']))
    Err.append(rerror)
    if this['height2'].size < 16 or rerror > 0.01:
        C1.append(None)
        C2.append(None)
        C3.append(None)
        C5.append(None)
        continue

    C1.append(results[v,c,'coef',0])
    C2.append(results[v,c,'coef',1])
    C3.append(results[v,c,'coef',2])
    C5.append(results[v,c,'coef',3])


def plot_scatter(x, y, c, name):
  plt.figure()
  plt.scatter(x, y, c=c, s=400)
  ax = plt.gca()
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel('Grashof')
  ax.set_ylabel('Schmidt');
  plt.colorbar()
  plt.savefig("{}-vs-Gr-Sc.{}".format(name, img_format))

def plot_dep(res, key, name):
    plt.figure()
    for c in results[0.0016,:,'coef',key].keys():
        plt.errorbar(res[:,c,'coef',key].keys(), res[:,c,'coef',key].values(), res[:,c,'std',key].values(), label="C={}".format(c))
    ax = plt.gca()
    ax.set_xscale("log")
    plt.savefig("{}-dep.{}".format(name, img_format))

plot_scatter(Grashof, Schmidt, Err, "Err")
plot_scatter(Grashof, Schmidt, C1, "C1")
plot_scatter(Grashof, Schmidt, C2, "C2")
plot_scatter(Grashof, Schmidt, C3, "C3")
plot_scatter(Grashof, Schmidt, C5, "C5")

plot_dep(results, 0, 'C1')
plot_dep(results, 1, 'C2')
plot_dep(results, 2, 'C3')
plot_dep(results, 3, 'C5')

