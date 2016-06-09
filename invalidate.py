from sys import argv
import json

cache_file = argv[1]

with open(cache_file, "r") as f:
    cache = json.load(f)

for k in list(cache.keys()):
    if k[0:7] == "analyze":
        del cache[k]

with open(cache_file, "w") as f:
    json.dump(cache, f)
