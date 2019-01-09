import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

# fix up multi-word token

lines = []

for line in sys.stdin:
    line = line.strip()
    lines += [line]

input_lines = []
with open(input_file) as f:
    for line in f:
        line = line.strip()

        input_lines += [line]

with open(output_file, 'w') as outf:
    i = 0
    for line in input_lines:
        if len(line) == 0:
            print(lines[i], file=outf)
            i += 1
            continue

        if line[0] == '#':
            continue

        line = line.split('\t')
        if '.' in line[0]:
            continue

        if '-' in line[0]:
            line[6] = '_'
            line[9] = '_'
            print('\t'.join(line), file=outf)
            continue

        print(lines[i], file=outf)
        i += 1
