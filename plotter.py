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

from model import exp_mix, mix_model
from model import exp_dyn, dyn_model
from model import both_error, full_model, filter_trajectory

with open("fit_results.p", "rb") as f:
    results_in = pickle.load(f)

"""
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
"""

#results = CachedSlict(results_d)
results = results_in

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
    mix = 2*(64 - mix)
    #times = this['time']; heights = this['height2']

    if len(times) < len(this['time']):
                print("TRUNC: {} {} stopped at {}".format(v, c, times[-1]))

    if heights.size < 4:
        continue

    L = np.sqrt(2) / this["kmin"]
    Atwood = this['atwood'] * this['g']
    y0 = [this["amp0"]/this["kmin"], 0.]

    offset = this['delta']**2. / c    

    h_spline = UnivariateSpline(times, heights, k=3, s = 0.00000001)
    v_spline = h_spline.derivative()
    m_spline = UnivariateSpline(times,     mix, k=3, s = 0.00000001)

    T, MO = mix_model(exp_mix(results[v,c]['C_mix']), L, c, this['delta'], times, h_spline)

    T, H, V = dyn_model(exp_dyn(results[v,c]['C_dyn']), Atwood, v, L, m_spline, y0, times)

    T, HB, VB, MB = full_model(exp_dyn(results[v,c]['C_dyn']), exp_mix(results[v,c]['C_mix']), Atwood, v, L, c, this['delta'], y0, times)

    dyn_error, mix_error = both_error(exp_dyn(results[v,c]['C_dyn']), exp_mix(results[v,c]['C_mix']), Atwood, v, L, c, this['delta'], y0, times, heights, mix)

    # Mixing plot
    fig, axs = plt.subplots(1,2, figsize=(8,8))

    axs[0].plot(times, heights, label="Simulation")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Height")
    #axs[0].legend()

    axs[1].plot(times, mix/(L*L), label="Simulation")
    axs[1].plot(times, MO/(L*L), label="Model")
    delta = np.sqrt((times+offset) * c)
    #axs[1].plot(times, 2*delta/np.sqrt(np.pi) * (1 + 2.*heights), label="Model")
    diam = L/4.
    DM = (2*delta/np.sqrt(np.pi)*(1-np.exp(-(diam**2./np.square(delta)))) + 2*diam*(1-erf(diam / delta))) * (1 + 2.*heights)
    #axs[1].plot(times, DM, label="Model")
    #axs[1].legend()

    plt.savefig('M-{:d}-{:d}.{}'.format(int(v*10000),int(c*10000),img_format))
    plt.close()

    """
    fig, axs = plt.subplots(1, 2, figsize=(8,8))
    axs[0].plot(times, heights, label="Simulation")
    axs[0].plot(times, H, label="Model")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Height")
    axs[0].legend(loc=2)

    axs[1].plot(times, v_spline(times)/ np.sqrt(Atwood * L), label="Simulation")
    axs[1].plot(times, V / np.sqrt(Atwood * L), label="Model")
    #axs[1].legend()
    axs[1].axhline(1./np.sqrt(np.pi))
    axs[1].axhline(L * np.sqrt(Atwood * L) / (results[v,c]['C_dyn'][1] * v))
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")

    plt.savefig('H-{:d}-{:d}.{}'.format(int(v*10000),int(c*10000),img_format))
    plt.close()
    """

    fig, axs = plt.subplots(1, 3, figsize=(12,8))
    axs[0].plot(times, heights, label="Simulation")
    axs[0].plot(times, HB, label="Model")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Height")
    axs[0].legend(loc=2)

    axs[1].plot(times, v_spline(times)/ np.sqrt(Atwood * L), label="Simulation")
    axs[1].plot(times, VB / np.sqrt(Atwood * L), label="Model")
    #axs[1].legend()
    axs[1].axhline(1./np.sqrt(np.pi), color='black', linestyle='dashed')
    #axs[1].axhline(L * np.sqrt(Atwood * L) / (results[v,c]['C_dyn'][1] * v))
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(times, mix/(2*L*L), label="Simulation")
    axs[2].plot(times, MB/(2*L*L), label="Model")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Mixing height")
    axs[2].yaxis.set_label_position("right")
    #axs[2].legend()

    axs[1].set_title(r"Fit of $\nu=${} and D={}".format(v, c))
    plt.savefig('H-{:d}-{:d}.{}'.format(int(v*10000),int(c*10000),img_format))
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


    res = results[v,c]
    print("V={}, D={}, C=[{:8.3f}], Err={err:8.3f}".format(v, c, *(res['C_mix']), err=res['E_mix']))
#    print("V={}, D={}, T={}, Err={}\n >> C=".format(v, c, data_table[v,c,'time'][-1], res['E_',0]) + str(res['coef',:].values()))
    if np.max(heights) < np.sqrt(2):
        continue
    print("LW: V={}, D={}, C_dyn=[{:6.3f}, {:8.3f}, {:6.3f}, {:6.3f}], C_mix=[{C5:6.3f}]: [{derr0:6.3f}, {merr0:6.3f}, {derr:6.3f}, {merr:6.3f}]".format(
                  v, c, *(res['C_dyn']), C5=res['C_mix'][0], derr0=res['E_dyn'], merr0=res['E_mix'], derr=dyn_error, merr=mix_error))
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
status = []
for v, c in results.keys():
    this = data_table[v,c, :]
    times, heights, mix = filter_trajectory(this['time'], this['height2'], this['mixed_mass'], this['extent_mesh'][2])
    Grashof.append(this['atwood']*this['g']/(v**2 * this['kmin']**3))
    Rayleigh.append(this['atwood']*this['g']/(v*c * this['kmin']**3))
    Schmidt.append(v/c)
    rerror = results[v,c]['E_dyn'] / min(24.0, np.max(this['height2']))
    Err.append(rerror)
    #if this['height2'].size < 64 or rerror > 0.01:

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


    if heights.size < 64 or np.max(heights) < np.sqrt(2): # or not (trunc or done):
        C1.append(None)
        C2.append(None)
        C3.append(None)
        C5.append(None)
        C7.append(None)
        print("Skipping {} {}".format(v,c))
        continue

    status.append(color)
    res = results[v,c]
    C1.append(res['C_dyn'][0])
    C2.append(res['C_dyn'][1])
    C3.append(res['C_dyn'][2])
    C5.append(res['C_mix'][0])
    C7.append(res['C_dyn'][3])


print("LENS: ", len(status), len(C1))

def plot_scatter(x, y, c, name, xlabel='Grashof', ylabel='Schmidt'):
    plt.figure()
    if name is not "Error":
        plt.scatter(x, y, c=c, s=400, edgecolors=status, linewidths=2)
    else:
        plt.scatter(x, y, c=c, s=400)
    plt.set_cmap('plasma')
    ax = plt.gca()
    ax.grid()
    ax.set_xscale("log", basex=2.0)
    ax.set_yscale("log", basey=2.0)
    #if xlabel == 'Grashof':
    #    ax.set_xlim(128, 1048576)
    #else:
    #    ax.set_xlim(2048, 1048576)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel);
    ax.set_title("{} vs {} and {} numbers".format(name, xlabel, ylabel))
    plt.colorbar()
    plt.savefig("{}-vs-{}-{}.{}".format(name, xlabel, ylabel, img_format))
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

"""
plot_dep(results, 0, 'C1')
plot_dep(results, 1, 'C2')
plot_dep(results, 2, 'C3')
plot_dep(results, 3, 'C5')
plot_dep(results, 4, 'C7')
"""
