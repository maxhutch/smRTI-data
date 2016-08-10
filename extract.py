#!/home/maxhutch/anaconda3/bin/python3

from chest import Chest
from slict import CachedSlict
import numpy as np

import json
from os import getcwd
from os.path import exists

from nekpy.dask.subgraph import series
from nekpy.dask.tasks import configure
from nekpy.dask.utils import outer_product, work_name
from nekpy.dask import run_all

from my_utils import find_root, find_steep

from sys import argv

def extract_values(c, p, table):
    sc = CachedSlict(c)
    times = sc[:,'z_z'].keys()[:max_index]

    ts = []
    h1 = []
    h2 = []
    h3 = []
    rA = []
    mm = []
    for time in times:
        zs = sc[time, "z_z"]
        tz = np.array(sc[time,"t_max_z"])
        t_proj = np.array(sc[time,"t_proj_z"])
        h_exp, i = find_root(zs, tz, y0 = 0.0)
#        if tz[i-1] < .5 and abs(tz[i] - tz[i-1]) < 0.01:
#            print("Truncating at {}, h = {}, t = {}".format(i, h_exp, time))
#            break
        
        mixed_mass = 4. * sc[time, 'Xi'] #(1.-4 * np.mean(sc[time, 't_abs_proj_z'])) * p['extent_mesh'][2]
        relative_A = 1. - ((1.-4 * np.mean(sc[time, 't_abs_proj_z'])) 
                          * p['extent_mesh'][2] / max(h_exp, p['amp0'] + p['delta']))

        h_exp2, j = find_steep(zs, tz)
        h_exp3, j = find_root(zs, 4.0*t_proj, y0=-0.99)

#        if len(h2) > 3 and h_exp2 < h2[-1]:
#            break

#        print("{} vs {}, {} vs {}".format(i, j, h_exp, h_exp2))
        ts.append(time)
        h1.append(h_exp)
        h2.append(h_exp2)
        h3.append(h_exp3)
        rA.append(relative_A)
        mm.append(mixed_mass)

    table[p['viscosity'], p['conductivity'], 'time'] = np.array(ts)
    print("Got to time {}".format(ts[-1]))
    table[p['viscosity'], p['conductivity'], 'height'] = np.array(h1)
    table[p['viscosity'], p['conductivity'], 'height2'] = np.array(h2)
    table[p['viscosity'], p['conductivity'], 'height3'] = np.array(h3)
    table[p['viscosity'], p['conductivity'], 'relative_atwood'] = np.array(rA)
    table[p['viscosity'], p['conductivity'], 'mixed_mass'] = np.array(mm)

    for k, v in p.items():
        table[p['viscosity'], p['conductivity'], k] = v

    return

with open(argv[1], "r") as f:
    base = json.load(f)

with open(argv[2], "r") as f:
    sweeps = json.load(f)

base["prefix"] = sweeps["prefix"]
del sweeps["prefix"]

# Take simple outer product of contents of sweep file
candidates = list(outer_product(sweeps))

# Filter out the cases we don't want
overrides = []
for c in candidates:
    overrides.append(c)

# Tune the remaining cases
aspect = 4
for ov in overrides:
    ov["name"] = work_name(base["prefix"], ov)
    ov['viscosity'] = ov['nu'] * 1.0e-4
    ov['conductivity'] = ov['viscosity'] / ov['Sc']

from os.path import join
workdirs = [join(getcwd(), x["name"]) for x in overrides]
configs = [configure(base, override, workdir) for override, workdir in zip(overrides, workdirs)]

data_table = {}

max_index = -1
for p, wd in zip(configs, workdirs):
    path = join(wd, "{}-results".format(p['name']))
    print(path)
    if exists(path):
        c = Chest(path=path)
        extract_values(c, p, data_table)

import pickle
with open("data_table.p", "wb") as f:
    pickle.dump(data_table, f)

