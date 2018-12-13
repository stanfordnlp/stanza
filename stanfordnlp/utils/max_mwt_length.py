import sys

import json

with open(sys.argv[1]) as f:
    d = json.load(f)
    l = max([0] + [len(" ".join(x[0][1])) for x in d])

with open(sys.argv[2]) as f:
    d = json.load(f)
    l = max([l] + [len(" ".join(x[0][1])) for x in d])

print(l)
