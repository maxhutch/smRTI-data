#/usr/bin/env python3 

import pickle
from slict import CachedSlict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.interpolate import UnivariateSpline
from scipy.special import erf

img_format = 'png'


with open("data_table.p", 'rb') as f:
    data_table_d = pickle.load(f)

data_table = CachedSlict(data_table_d)

from model import full_model, error, filter_trajectory, guess
from model import merge_coef
from model import fix_thin
from model import mix_model_direct

with open("fit_results.p", "rb") as f:
    results_in = pickle.load(f)

results_d = {}
for v, c in results_in.keys():
    if len(results_in[v,c]) < 4:
        continue

    for i in range(len(results_in[v,c][0])):
        results_d[v, c, 'coef', i] = results_in[v,c][0][i]
        results_d[v, c, 'std', i] = results_in[v,c][2][i]
    results_d[v, c, 'err', 0] = results_in[v, c][1]
    for i in range(len(results_in[v,c][3])):
        results_d[v, c, 'mix_coef', i] = results_in[v,c][3][i]
    results_d[v, c, 'mix_err', 0] = results_in[v, c][4]

results = CachedSlict(results_d)


for v, c in data_table[:,:,'time'].keys():

    """
    if v !=  0.0008 or c != 0.0008:
        continue

    results_d[v,c,'coef',0] = 4.
    results_d[v,c,'coef',1] = 400.
    results_d[v,c,'coef',2] = 2.
    results_d[v,c,'coef',3] = 4.
    results_d[v,c,'coef',4] = 4.
    results_d[v,c,'coef',5] = 0.
    """

    this = data_table[v, c, :]

    times, heights, mix = filter_trajectory(
        this['time'], this['height2'], this['mixed_mass'], this['extent_mesh'][2]
        )
    #times = this['time']; heights = this['height2']

    if len(times) < len(this['time']):
                print("TRUNC: {} {} stopped at {}".format(v, c, times[-1]))

    if heights.size < 4:
        continue

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0., 2*this['delta']*L*L/np.sqrt(np.pi)]

    offset = this['delta']**2. / c    

    h_spline = UnivariateSpline(times, heights, k=3, s = 0.00000001)
    v_spline = h_spline.derivative()
    #T, H, V, MM = model(guess, Atwood, v, L, c, y0, times)
    #axs[0].plot(times, H)
    #print(results[v,c,'coef',:].values())
    T, MO = mix_model_direct(results[v,c,'mix_coef',:].values(), L, c, y0, times, h_spline)

    T, H, V, MM = full_model(merge_coef(results[v,c,'coef',:].values(), fix_thin), Atwood, v, L, c, y0, times)

    # Mixing plot
    fig, axs = plt.subplots(1,2, figsize=(8,8))

    axs[0].plot(times, heights, label="Simulation")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Height")
    #axs[0].legend()

    axs[1].plot(times, 2*(64 - mix)/(L*L), label="Simulation")
    axs[1].plot(times, MO/(L*L), label="Model")
    delta = np.sqrt((times+offset) * c)
    #axs[1].plot(times, 2*delta/np.sqrt(np.pi) * (1 + 2.*heights), label="Model")
    diam = L/4.
    DM = (2*delta/np.sqrt(np.pi)*(1-np.exp(-(diam**2./np.square(delta)))) + 2*diam*(1-erf(diam / delta))) * (1 + 2.*heights)
    #axs[1].plot(times, DM, label="Model")
    #axs[1].legend()

    plt.savefig('M-{}-{}.{}'.format(v,c,img_format))
    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12,8))
    axs[0].plot(times, heights, label="Simulation")
    axs[0].plot(times, H, label="Model")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Height")
    axs[0].legend()

    axs[1].plot(times, v_spline(times)/ np.sqrt(Atwood * L), label="Simulation")
    axs[1].plot(times, V / np.sqrt(Atwood * L), label="Model")
    axs[1].legend()
    axs[1].axhline(1./np.sqrt(np.pi))
    axs[1].axhline(L * np.sqrt(Atwood * L) / (results[v,c,'coef',1] * v))
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(times, 2*(64 - mix)/(L*L), label="Simulation")
    axs[2].plot(times, MM/(L*L), label="Model")
    axs[2].legend()

    plt.savefig('H-{}-{}.{}'.format(v,c,img_format))
    plt.close()

    plt.figure()
    plt.plot(times, H - heights)
    plt.savefig('diff-{}-{}.{}'.format(v,c,img_format))
    plt.close()

    plt.figure()
    plt.plot(this['time'], this['height2'], label="Inflection")
    plt.plot(this['time'], this['height'], label="Max")
    plt.plot(this['time'], this['height3'], label="Mean")
    plt.legend()
    plt.savefig('comp-height-{}-{}.{}'.format(v,c,img_format))
    plt.close()


    res = results[v,c,:,:]
    print("V={}, D={}, C=[{:8.3f}, {:8.3f}], Err={err:8.3f}".format(v, c, *(res['mix_coef',:].values()), err=res['mix_err',0]))
    #print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res['err',0]) + str(res['coef',:].values()))
    #print(" >> S=" + str(res['std',:].values()))

Grashof = []
Schmidt = []
Rayleigh = []
Err = []
C1 = []
C2 = []
C3 = []
C5 = []
C7 = []
for v, c in data_table[:,:,'time'].keys():
    this = data_table[v,c, :]
    times, heights, mix = filter_trajectory(this['time'], this['height2'], this['mixed_mass'], this['extent_mesh'][2])
    Grashof.append(this['atwood']*this['g']/(v**2 * this['kmin']**3))
    Rayleigh.append(this['atwood']*this['g']/(v*c * this['kmin']**3))
    Schmidt.append(v/c)
    rerror = results[v,c,'err',0] / min(24.0, np.max(this['height2']))
    Err.append(rerror)
    #if this['height2'].size < 64 or rerror > 0.01:
    if heights.size < 64:
        C1.append(None)
        C2.append(None)
        C3.append(None)
        C5.append(None)
        C7.append(None)
        print("Skipping {} {}".format(v,c))
        continue

    C1.append(results[v,c,'coef',0])
    C2.append(results[v,c,'coef',1])
    C3.append(results[v,c,'coef',2])
    C5.append(results[v,c,'coef',3])
    C7.append(results[v,c,'coef',4])


def plot_scatter(x, y, c, name, xlabel='Grashof', ylabel='Schmidt'):
    plt.figure()
    plt.scatter(x, y, c=c, s=400)
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel);
    ax.set_title("{} vs {} and {} numbers".format(name, xlabel, ylabel))
    plt.colorbar()
    plt.savefig("{}-vs-{}-{}.{}".format(name, xlabel, ylabel, img_format))
    plt.close()

def plot_dep(res, key, name):
    plt.figure()
    for c in results[0.0016,:,'coef',key].keys():
        plt.errorbar(res[:,c,'coef',key].keys(), res[:,c,'coef',key].values(), res[:,c,'std',key].values(), label="C={}".format(c))
    ax = plt.gca()
    ax.set_xscale("log")
    plt.savefig("{}-dep.{}".format(name, img_format))
    plt.close()

plot_scatter(Grashof, Schmidt, Err, "Error")
plot_scatter(Rayleigh, Schmidt, Err, "Error", xlabel='Rayleigh')
plot_scatter(Grashof, Schmidt, C1, "C1")
plot_scatter(Rayleigh, Schmidt, C1, "C1", xlabel='Rayleigh')
plot_scatter(Grashof, Schmidt, C2, "C2")
plot_scatter(Rayleigh, Schmidt, C2, "C2", xlabel='Rayleigh')
plot_scatter(Grashof, Schmidt, C3, "C3")
plot_scatter(Rayleigh, Schmidt, C3, "C3", xlabel='Rayleigh')
plot_scatter(Grashof, Schmidt, C5, "C5")
plot_scatter(Rayleigh, Schmidt, C5, "C5", xlabel="Rayleigh")
plot_scatter(Grashof, Schmidt, C7, "C7")
plot_scatter(Rayleigh, Schmidt, C7, "C7", xlabel="Rayleigh")

plot_dep(results, 0, 'C1')
plot_dep(results, 1, 'C2')
plot_dep(results, 2, 'C3')
plot_dep(results, 3, 'C5')
plot_dep(results, 4, 'C7')

