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


end_times = {}
end_times[2, 2] = 176.0
end_times[2, 1] = 176.0
end_times[4, 4] = 192.0
end_times[4, 2] = 200.0
end_times[4, 1] = 232.0
end_times[8, 8] = 280.0
end_times[8, 4] = 312.0
end_times[8, 2] = 456.0
end_times[8, 1] = 504.0
end_times[16, 16] = 504.0
end_times[16, 8] = 700.0
end_times[16, 4] = 700.0
end_times[16, 2] = 368.0
end_times[16, 1] = 224.0
end_times[32, 32] = 952.0
end_times[32, 16] = 1208.0
end_times[32, 8] = 544.0
end_times[32, 4] = 280.0
end_times[32, 2] = 200.0
end_times[32, 1] = 40.0
end_times[64, 32] = 984.0
end_times[64, 16] = 464.0
end_times[64, 8] = 288.0
end_times[64, 4] = 40.0
end_times[64, 2] = 8.0
end_times[64, 1] = 8.0

# Filter out the cases we don't want
overrides = []
for c in candidates:
    #if c['nu'] / c['Sc'] < 2 or c['nu'] < 4:
    if c['nu'] / c['Sc'] < 1 or c['nu'] < 2:
        continue
    overrides.append(c)

# Tune the remaining cases
for ov in overrides:
    ov["name"] = work_name(base["prefix"], ov)

    ov['viscosity'] = ov['nu'] * 1.0e-4
    ov['conductivity'] = ov['viscosity'] / ov['Sc']

    if ov['nu'] <= 2 or ov['nu'] / ov['Sc'] <= 1:
        elm = 4
    else:
        elm = 2
    ov['shape_mesh'] = [elm, elm, elm*128] 
    nodes = max(1, int(elm * elm * elm * 128 / 64))
    ov["procs"] = 32*nodes
    ov["io_files"] = -nodes
    ov["dt"] = (2/(elm*(base["order"]-1)**2))/0.0558519
    if (ov['nu'], ov['Sc']) in end_times:
      ov["end_time"] = end_times[ov["nu"], ov["Sc"]]

from os.path import join
workdirs = [join(getcwd(), x["name"]) for x in overrides]
configs = [configure(base, override, workdir) for override, workdir in zip(overrides, workdirs)]
res = [series(config, tusr, job_time = 8.0) for config in configs]
final = run_all(res, base, num_workers=16)
