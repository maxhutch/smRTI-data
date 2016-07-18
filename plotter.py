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
from sys import argv
from argparse import ArgumentParser
from json import loads

img_format = 'png'
title = False

parser = ArgumentParser(description="Plotter for smRTI data")
parser.add_argument("--traj", action="store_true", default=False)
parser.add_argument("--only", type=str, default=None)
parser.add_argument("params", type=str, default="fit_results.p")

args = parser.parse_args()

with open("data_table.p", 'rb') as f:
    data_table_d = pickle.load(f)

data_table = CachedSlict(data_table_d)

from model import exp_mix, mix_model
from model import exp_dyn, dyn_model
from model import both_error, full_model, filter_trajectory

with open(args.params, "rb") as f:
    results_in = pickle.load(f)


if args.only is not None:
    todo = loads(args.only)
else:
    todo = data_table[:,:,'time'].keys()

if not args.traj:
    todo = []

#results = CachedSlict(results_d)
results = results_in


def plot_model(this, C_mix, C_dyn, cascade=False):
    v = this['viscosity']
    c = this['conductivity']

    times, heights, mix = filter_trajectory(
        this['time'], this['height2'], this['mixed_mass'], this['extent_mesh'][2]
        )
    mix = 2*(64 - mix)
    #times = this['time']; heights = this['height2']

    if len(times) < len(this['time']):
        print("TRUNC: {} {} stopped at {}".format(v, c, times[-1]))

    if heights.size < 4:
        return

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0.]

    offset = this['delta']**2. / c    

    h_spline = UnivariateSpline(times, heights, k=3, s = 0.00000001)
    v_spline = h_spline.derivative()
    m_spline = UnivariateSpline(times,     mix, k=3, s = 0.00000001)

    #T, H, V = dyn_model(exp_dyn(C_dyn), Atwood, v, L, m_spline, y0, times)

    T, HB, VB, MB = full_model(exp_dyn(C_dyn), exp_mix(C_mix), Atwood, v, L, c, this['delta'], y0, times)

    dyn_error, mix_error = both_error(exp_dyn(C_dyn), exp_mix(C_mix), Atwood, v, L, c, this['delta'], y0, times, heights, mix)

    fig, axs = plt.subplots(1, 3, figsize=(12,8))
    axs[0].plot(times, heights, label="Simulation")
    axs[0].plot(times, HB, label="Model")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Height")
    axs[0].set_ylim(0.0, np.max(heights))
    axs[0].legend(loc=2)

    axs[1].plot(times, v_spline(times)/ np.sqrt(Atwood * L), label="Simulation")
    axs[1].plot(times, VB / np.sqrt(Atwood * L), label="Model")
    #axs[1].legend()
    axs[1].axhline(1./np.sqrt(np.pi), color='black', linestyle='dashed')
    #axs[1].axhline(L * np.sqrt(Atwood * L) / (results[v,c]['C_dyn'][1] * v))
    axs[1].set_ylim(0.0, np.max(v_spline(times)))
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(times, mix/(2*L*L), label="Simulation")
    axs[2].plot(times, MB/(2*L*L), label="Model")
    axs[2].set_ylim(0.0, np.max(mix/(2*L*L)))
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Mixing height")
    axs[2].yaxis.set_label_position("right")
    #axs[2].legend()

    axs[1].set_title(r"Fit of $\nu=${} and D={}".format(v, c))
    plt.savefig('H-{:d}-{:d}.{}'.format(int(v*10000),int(c*10000),img_format))


    res = results[v,c]
    #print("V={}, D={}, C=[{:8.3f}], Err={err:8.3f}".format(v, c, *(C_mix), err=res['E_mix']))
    #print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res['E_',0]) + str(res['coef',:].values()))
    if np.max(heights) < np.sqrt(2):
        return 
    print("LW: V={}, D={}, C_dyn=[{:6.3f}, {:8.3f}, {:6.3f}, {:6.3f}], C_mix=[{C5:6.3f}]: [{derr:6.3f}, {merr:6.3f}]".format(
                  v, c, *(C_dyn), C5=C_mix[0], derr=dyn_error, merr=mix_error))
    #print(" >> S=" + str(res['std',:].values()))

    if not cascade:
        return

    fig, axs = plt.subplots(1, 1)

    axs.plot(heights/L, v_spline(times)/ np.sqrt(Atwood * L), label="Simulation", color='black')
    axs.set_ylim(0.0, 2*np.max(v_spline(times))/ np.sqrt(Atwood * L))
    #axs.set_xlim(0.0, np.max(heights)/L)
    axs.set_xlim(0.0, 2)
    axs.set_xlabel("Bubble Height ($h/\lambda$)")
    axs.set_ylabel("Froude number")

    C_mix_tmp = [0., 0.]
    C_dyn_tmp = [0., 0., 0., 0.]
    T, HB, VB, MB = full_model(exp_dyn(C_dyn_tmp), exp_mix(C_mix_tmp), Atwood, v, L, c, this['delta'], y0, times)
    axs.plot(HB/L, VB / np.sqrt(Atwood * L), label='0')

    C_mix_tmp = [0., 0.]
    C_dyn_tmp = [0., 0., C_dyn[2], 0.]
    T, HB, VB, MB = full_model(exp_dyn(C_dyn_tmp), exp_mix(C_mix_tmp), Atwood, v, L, c, this['delta'], y0, times)
    axs.plot(HB/L, VB / np.sqrt(Atwood * L), label='C3')

    C_mix_tmp = [0., 0.]
    C_dyn_tmp = [C_dyn[0], 0., C_dyn[2], 0.]
    T, HB, VB, MB = full_model(exp_dyn(C_dyn_tmp), exp_mix(C_mix_tmp), Atwood, v, L, c, this['delta'], y0, times)
    axs.plot(HB/L, VB / np.sqrt(Atwood * L), label='C1')

    C_mix_tmp = [0., 0.]
    C_dyn_tmp = [C_dyn[0], C_dyn[1], C_dyn[2], 0.]
    T, HB, VB, MB = full_model(exp_dyn(C_dyn_tmp), exp_mix(C_mix_tmp), Atwood, v, L, c, this['delta'], y0, times)
    axs.plot(HB/L, VB / np.sqrt(Atwood * L), label='C2')

    C_mix_tmp = [C_mix[0], 0.]
    C_dyn_tmp = [C_dyn[0], C_dyn[1], C_dyn[2], C_dyn[3]]
    T, HB, VB, MB = full_model(exp_dyn(C_dyn_tmp), exp_mix(C_mix_tmp), Atwood, v, L, c, this['delta'], y0, times)
    axs.plot(HB/L, VB / np.sqrt(Atwood * L), label='C5,C7')

    axs.legend()
    axs.grid()
    plt.savefig('Cascade-{:d}-{:d}.{}'.format(int(v*10000),int(c*10000),img_format))

    return


for v, c in todo:

    if v < 0.0002:
        continue
    if (v, c) not in results:
        continue

    this = data_table[v, c, :]
    plot_model(this, results[v,c]['C_mix'], results[v,c]['C_dyn'])

vfoo = 0.0004; cfoo = 0.0001
this = data_table[vfoo, cfoo, :]
C_mix_full = results[vfoo,cfoo]['C_mix']
C_dyn_full = results[vfoo, cfoo]['C_dyn']

plot_model(this, C_mix_full, C_dyn_full, cascade=True)


Grashof = []
Schmidt = []
Rayleigh = []
RDE = []
RME = []
C1 = []
C2 = []
C3 = []
C5 = []
C7 = []
depth = []
status = []
for v, c in results.keys():
    if v < 0.0002:
        continue

    this = data_table[v,c, :]
    times, heights, mix = filter_trajectory(this['time'], this['height2'], this['mixed_mass'], this['extent_mesh'][2])
    mix = 2*(64 - mix)

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0.]


    if heights.size < 64 or np.max(heights) < np.sqrt(2): # or not (trunc or done):
        print("Skipping {} {}".format(v,c))
        continue

    Grashof.append(this['atwood']*this['g']*np.sqrt(2)**3/(v**2 * this['kmin']**3))
    Schmidt.append(v/c)
    Rayleigh.append(Grashof[-1] * Schmidt[-1])


    dyn_error, mix_error = both_error(exp_dyn(results[v,c]['C_dyn']), exp_mix(results[v,c]['C_mix']), Atwood, v, L, c, this['delta'], y0, times, heights, mix)


    rel_dyn_error = dyn_error / np.max(heights)
    rel_mix_error = mix_error / np.max(mix)

    trunc = False
    done = False
    color = 'yellow'
    if np.max(heights) > 23.5:
        color = 'red'
        trunc = True
        print(v, c, "got trunc")
    elif len(times) < len(this['time']) or np.max(times) > 1000.0:
        color = 'green'
        done = True
        print(v, c, "got done")

    status.append(color)
    depth.append(np.max(heights))
    res = results[v,c]
    RDE.append(rel_dyn_error)
    RME.append(rel_mix_error)
    C1.append(res['C_dyn'][0])
    C2.append(res['C_dyn'][1])
    C3.append(res['C_dyn'][2])
    C5.append(res['C_mix'][0])
    C7.append(res['C_dyn'][3])

def plot_scatter(x, y, c, name, xlabel='Grashof', ylabel='Schmidt'):
    plt.figure()
    plt.scatter(x, y, c=c, s=1500)
    plt.set_cmap('plasma')
    ax = plt.gca()
    ax.grid()
    ax.set_xscale("log")
    ax.set_yscale("log")
    xdiff = (np.max(x)/np.min(x))**(1.0/8.0)
    ax.set_xlim(np.min(x)/xdiff, np.max(x)*xdiff)
    ax.set_ylim(1.0/np.sqrt(2.0), np.max(y)*2.0)

    plt.axvline(2.0**15.9024, color='black', linestyle='dashed')
    plt.annotate(s="Complete", xy=(10**4, 64))
    plt.annotate(s="Incomplete", xy=(10**5.5, 64))
        
    ax.set_xlabel("Rayleigh Number")
    ax.set_ylabel("Schmidt Number");
    if title:
        ax.set_title("Trends in {}".format(name))
    plt.colorbar()
    fname = name.replace(" ","")
    fname = fname.replace("_","")
    fname = fname.replace("$","")
    fxlabel = xlabel.replace(" ","")
    fylabel = ylabel.replace(" ","")
    plt.savefig("{}-vs-{}-{}.{}".format(fname, fxlabel, fylabel, img_format))
    plt.close()

"""
def plot_dep(res, key, name):
    plt.figure()
    for c in results[0.0016,:,'coef',key].keys():
        plt.errorbar(res[:,c,'coef',key].keys(), res[:,c,'coef',key].values(), res[:,c,'std',key].values(), label="C={}".format(c))
    ax = plt.gca()
    ax.set_xscale("log")
    plt.savefig("{}-dep.{}".format(name, img_format))
    plt.close()
"""

"""
plot_scatter(Grashof, Schmidt, RDE, "DynError")
plot_scatter(Grashof, Schmidt, RME, "MixError")
plot_scatter(Grashof, Schmidt, C1, "C1")
plot_scatter(Grashof, Schmidt, C2, "C2")
plot_scatter(Grashof, Schmidt, C3, "C3")
plot_scatter(Grashof, Schmidt, C5, "C5")
plot_scatter(Grashof, Schmidt, C7, "C7")
plot_scatter(Grashof, Schmidt, depth, "Penetration Depth")
"""

plot_scatter(Rayleigh, Schmidt, RDE, "Dynamics Error", xlabel='Rayleigh')
plot_scatter(Rayleigh, Schmidt, RME, "Mixing Error", xlabel='Rayleigh')
plot_scatter(Rayleigh, Schmidt, C1, "$C_1$", xlabel='Rayleigh')
plot_scatter(Rayleigh, Schmidt, C2, "$C_2$", xlabel='Rayleigh')
plot_scatter(Rayleigh, Schmidt, C3, "$C_3$", xlabel='Rayleigh')
plot_scatter(Rayleigh, Schmidt, C5, "$C_5$", xlabel="Rayleigh")
plot_scatter(Rayleigh, Schmidt, C7, "$C_7$", xlabel="Rayleigh")

plot_scatter(Rayleigh, Schmidt, depth, "Penetration Depth", xlabel='Rayleigh')

"""
plot_dep(results, 0, 'C1')
plot_dep(results, 1, 'C2')
plot_dep(results, 2, 'C3')
plot_dep(results, 3, 'C5')
plot_dep(results, 4, 'C7')
"""
