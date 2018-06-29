from collections import OrderedDict
from functools import reduce
import json
import numpy as np
from operator import mul
import os
import pickle
from pprint import pprint
import random
import sys
import subprocess

config_file, sweep_progress, command = sys.argv[1], sys.argv[2], sys.argv[3:]

with open(config_file, 'r') as f:
    loaded = ''.join([x.strip() for x in f.readlines()])
    config = json.loads(loaded, object_pairs_hook=OrderedDict)

SAVED_PROGRESS = sweep_progress

PRIOR_STRENGTH = .01
BINARY_PRIOR_STRENGTH = 1

unitary = {k: [[0.0, PRIOR_STRENGTH] for _ in range(len(config[k])-1)] for k in config}

binary_keys = [k for k in config.keys() if len(config[k]) > 2]
binary = {"{}<>{}".format(k1, k2):[[0.0, BINARY_PRIOR_STRENGTH] for _ in range((len(config[k1]) - 2) * (len(config[k2]) - 2))] for i, k1 in enumerate(binary_keys[:-1]) for k2 in binary_keys[i+1:]}

overall = [0, PRIOR_STRENGTH]

def estimate_params(progress, unitary=unitary, overall=overall, config=config, binary=binary, binary_keys=binary_keys):
    print("Estimating hyperparameter optimizer parameters...")
    print(" > Generating features...")
    D = sum([len(unitary[k]) for k in unitary])
    D2 = sum([len(binary[k]) for k in binary])

    # build prior
    SQRT_PRIOR = np.sqrt(PRIOR_STRENGTH)
    SQRT_BPRIOR = np.sqrt(BINARY_PRIOR_STRENGTH)
    A = [] # features are organized as follows [overall bias, unitary features, binary interaction features]
    b = []

    for i in range(D+D2):
        A += [[0] + [(SQRT_PRIOR if i < D else SQRT_BPRIOR) if j == i else 0 for j in range(D+D2)]]
        b += [0]

    #for i in range(D):
    #    A += [[SQRT_PRIOR] + [SQRT_PRIOR if j == i else 0 for j in range(D)] + [0] * D2]
    #    b += [0]

    #for i, k1 in enumerate(binary_keys[:-1]):
    #    for k2 in binary_keys[i+1:]:
    #        for x in range(2, len(config[k1])):
    #            for y in range(2, len(config[k2])):
    #                cur = [SQRT_PRIOR] + [SQRT_PRIOR if (k == k1 and j == x) or (k == k2 and j == y) else 0 for k in config.keys() for j in range(1, len(config[k]))]
    #                cur += [SQRT_PRIOR if (k1_ == k1 and x_ == x and k2_ == k2 and y_ == y) else 0 for i_, k1_ in enumerate(binary_keys[:-1]) for k2_ in binary_keys[i_+1:] for x_ in range(2, len(config[k1_])) for y_ in range(2, len(config[k2_]))]

    #                A += [cur]
    #                b += [0]

    # convert actual data
    for proposal, res in progress:
        cur = [1]
        try:
            for k in config:
                idx = config[k].index(proposal.get(k, config[k][0])) - 1
                cur += [1 if idx == j else 0 for j in range(len(config[k]) - 1)]
        except ValueError:
            continue

        for i, k1 in enumerate(binary_keys[:-1]):
            idx1 = config[k1].index(proposal.get(k1, config[k1][0]))
            for k2 in binary_keys[i+1:]:
                idx2 = config[k2].index(proposal.get(k2, config[k2][0]))
                cur += [1 if a == idx1 and b == idx2 else 0 for a in range(2, len(config[k1])) for b in range(2, len(config[k2]))]

        A += [cur]
        b += [res - 100]

    A = np.array(A)
    b = np.array(b)

    print(" > Solving for parameters...")
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    print(" > Unpacking parameters...")

    overall[0] = params[0]
    overall[1] = A.shape[0] - (D+D2) + PRIOR_STRENGTH

    counts = A[(D+D2):].sum(0)
    idx = 1
    for k in config:
        for j in range(len(unitary[k])):
            unitary[k][j] = params[idx], counts[idx] + PRIOR_STRENGTH
            idx += 1

    for i, k1 in enumerate(binary_keys[:-1]):
        for k2 in binary_keys[i+1:]:
            k = "{}<>{}".format(k1, k2)
            for j in range(len(binary[k])):
                binary[k][j] = params[idx], counts[idx] + BINARY_PRIOR_STRENGTH
                idx += 1

    assert idx == len(params)

    print(overall)
    pprint(unitary)
    #pprint(binary)

def get_proposal(invtemp=1, unitary=unitary, config=config, binary=binary, binary_keys=binary_keys):
    res = OrderedDict()
    for k in config:
        if np.random.random() < .05:
            # epsilon-greedy
            res[k] = config[k][np.random.randint(len(unitary[k])+1)]
            continue
        p = np.array([0] + [x[0] + np.random.randn() / np.sqrt(x[1]) / invtemp for x in unitary[k]], dtype=np.float64)

        if k in binary_keys:
            for k1 in binary_keys:
                if k1 == k: break

                idx1 = config[k1].index(res[k1])

                if idx1 < 2:
                    continue

                key = "{}<>{}".format(k1, k)

                for j in range(2, len(config[k])):
                    cand = binary[key][(idx1 - 2) * (len(config[k]) - 2) + j - 2]
                    p[j] += cand[0] + np.random.randn() / np.sqrt(cand[1]) / invtemp

        p += np.random.randn(*p.shape) / invtemp / np.sqrt(overall[1])
#        p = p - np.max(p)
#        p = np.exp(p * invtemp)
#        p /= np.sum(p)
#        res[k] = config[k][np.random.choice(np.arange(len(config[k])), p=p)]

        res[k] = config[k][np.argmax(p, axis=0)]

    return res

def evaluate_proposal(proposal, command=command, config=config):
    cmd = ['bash'] + command
    is_conv = False
    conv_str = ''
    for k in config:
        if not k.startswith('conv_filters'):
            if proposal[k] != False or not isinstance(proposal[k], bool):
                cmd += ["--{}".format(k)]
                if proposal[k] != True or not isinstance(proposal[k], bool):
                    cmd += [str(proposal[k])]
        else:
            if not is_conv:
                cmd += ['--conv_filters']
                conv_str += proposal[k]
                is_conv = True
            elif proposal[k] != False:
                conv_str += ',,' + proposal[k]
            else:
                break

    cmd += [conv_str]

    res = subprocess.run(cmd, stderr=subprocess.PIPE)
    try:
        return float(res.stderr)
    except Exception as e:
        print(res.stderr.decode('utf-8'))
        raise e

def save_load_progress(progress, update=[], filename=SAVED_PROGRESS):
    print('Saving sweep progress to "{}", please be patient...'.format(filename))
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            progress = pickle.load(f)
    progress += update
    with open(filename, 'wb') as f:
        pickle.dump(progress, f)
    print('Done!')
    return progress

progress = []
if os.path.exists(SAVED_PROGRESS):
    with open(SAVED_PROGRESS, 'rb') as f:
        progress = pickle.load(f)

    estimate_params(progress)

try:
    while True:
        #invtemp = min(1, .001 * (1+len(progress)))
        invtemp = 1
        print('Inv Temp = {}'.format(invtemp))
        print('Grid size = {}'.format(reduce(mul, [len(config[k]) for k in config], 1)))
        proposal = get_proposal(invtemp=invtemp)
        res = evaluate_proposal(proposal)
        progress = save_load_progress(progress, [[proposal, res]])
        estimate_params(progress)
except:
    import traceback
    traceback.print_last()
    save_load_progress(progress)
