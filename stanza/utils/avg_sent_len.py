import sys
import json

toklabels = sys.argv[1]

if toklabels.endswith('.json'):
    with open(toklabels, 'r') as f:
        l = json.load(f)

    l = [''.join([str(x[1]) for x in para]) for para in l]
else:
    with open(toklabels, 'r') as f:
        l = ''.join(f.readlines())

    l = l.split('\n\n')

sentlen = [len(x) + 1 for para in l for x in para.split('2')]
print(sum(sentlen) / len(sentlen))
