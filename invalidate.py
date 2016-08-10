from sys import argv
import json

cache_file = argv[1]

with open(cache_file, "r") as f:
    cache = json.load(f)

"""
for k in list(cache.keys()):
    if k[0:7] == "analyze":
        del cache[k]
"""
for k in list(cache.keys()):
    if "swp_Sc_8.0_nu_16.0" in k and k.count('-') == 2:
        print(k)
        ind = int(k.rpartition("-")[2])
        if ind > 31:
            del cache[k]
            print("Deleting {}".format(k))
    if "swp_Sc_16.0_nu_32.0" in k and k.count('-') == 2:
        print(k)
        ind = int(k.rpartition("-")[2])
        if ind > 31:
            del cache[k]
            print("Deleting {}".format(k))
    if "swp_Sc_32.0_nu_64.0" in k and k.count('-') == 2:
        print(k)
        ind = int(k.rpartition("-")[2])
        if ind > 31:
            del cache[k]
            print("Deleting {}".format(k))

with open(cache_file, "w") as f:
    json.dump(cache, f)
