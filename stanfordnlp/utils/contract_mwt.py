import sys

with open(sys.argv[2], 'w') as fout:
    with open(sys.argv[1], 'r') as fin:
        idx = 0
        mwt_begin = 0
        mwt_end = -1
        for line in fin:
            line = line.strip()

            if line.startswith('#'):
                print(line, file=fout)
                continue
            elif len(line) <= 0:
                print(line, file=fout)
                idx = 0
                mwt_begin = 0
                mwt_end = -1
                continue

            idx += 1
            line = line.split('\t')
            if '-' in line[0]:
                mwt_begin, mwt_end = [int(x) for x in line[0].split('-')]
                print(f"{idx}\t{"\t".join(line[1:-1])}\t{'MWT=Yes' if line[-1] == '_' else line[-1] + ',MWT=Yes'}", file=fout)
                idx -= 1
            elif mwt_begin <= idx <= mwt_end:
                continue
            else:
                print(f"{idx}\t{'\t'.join(line[1:]))}", file=fout)
