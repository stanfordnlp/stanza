import sys

import json

def max_mwt_length(filenames):
    max_len = 0
    for filename in filenames:
        with open(filename) as f:
            d = json.load(f)
            max_len = max([max_len] + [len(" ".join(x[0][1])) for x in d])
    return max_len

if __name__ == '__main__':
    print(max_max_jlength(sys.argv[1:]))
