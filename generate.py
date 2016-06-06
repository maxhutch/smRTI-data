#!/home/maxhutch/anaconda3/bin/python3

import json
from os import getcwd

from nekpy.dask.subgraph import series
from nekpy.dask.tasks import configure
from nekpy.dask.utils import outer_product, work_name
from nekpy.dask import run_all

from sys import argv
with open(argv[1], "r") as f:
    base = json.load(f)

with open(argv[2], "r") as f:
    sweeps = json.load(f)

with open(argv[3], "r") as f:
    tusr = f.read()

base["prefix"] = sweeps["prefix"]
del sweeps["prefix"]

# Take simple outer product of contents of sweep file
candidates = list(outer_product(sweeps))

# Filter out the cases we don't want
overrides = []
for c in candidates:
    overrides.append(c)

# Tune the remaining cases
aspect = 16
for ov in overrides:
    ov["name"] = work_name(base["prefix"], ov)
    ov["shape_mesh"] = [ov["elms"], ov["elms"], aspect*ov["elms"]]
    nodes = 1 #max(1, int(4 * (ov["order"]*ov["elms"])**3 / 8388608))
    ov["procs"] = 4*nodes
    ov["io_files"] = -nodes
    ov["dt"] = (2/(ov["elms"]*(ov["order"]-1)**2))/0.0558519

from os.path import join
workdirs = [join(getcwd(), x["name"]) for x in overrides]
configs = [configure(base, override, workdir) for override, workdir in zip(overrides, workdirs)]
res = [series(config, tusr, job_time = 16.0) for config in configs]
final = run_all(res, base)
