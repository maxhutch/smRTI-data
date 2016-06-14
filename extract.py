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

from sys import argv
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
height = 'H_exp'
for p, wd in zip(configs, workdirs):
    path = join(wd, "{}-results".format(p['name']))
    print(path)
    if exists(path):
        c = Chest(path=path)
        sc = CachedSlict(c)
        times = sc[:,height].keys()[:max_index]
        data_table[p['viscosity'], p['conductivity'], 'time'] = np.array(times) 
        data_table[p['viscosity'], p['conductivity'], 'height'] = np.array(
            [sc[t, height] for t in times]) 
        data_table[p['viscosity'], p['conductivity'], 'atwood'] = np.array(
            [4 * np.mean(sc[t, 't_abs_proj_z']) for t in times]) 
        for k, v in p.items():
            data_table[p['viscosity'], p['conductivity'], k] = v

import pickle
with open("data_table.p", "wb") as f:
    pickle.dump(data_table, f)
print(data_table)

