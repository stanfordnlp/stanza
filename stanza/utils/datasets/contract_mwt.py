import sys

def contract_mwt(infile, outfile, ignore_gapping=True):
    with open(outfile, 'w') as fout:
        with open(infile, 'r') as fin:
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
    
                line = line.split('\t')

                # ignore gapping word
                if ignore_gapping and '.' in line[0]:
                    continue

                idx += 1
                if '-' in line[0]:
                    mwt_begin, mwt_end = [int(x) for x in line[0].split('-')]
                    print("{}\t{}\t{}".format(idx, "\t".join(line[1:-1]), "MWT=Yes" if line[-1] == '_' else line[-1] + "|MWT=Yes"), file=fout)
                    idx -= 1
                elif mwt_begin <= idx <= mwt_end:
                    continue
                else:
                    print("{}\t{}".format(idx, "\t".join(line[1:])), file=fout)

if __name__ == '__main__':
    contract_mwt(sys.argv[1], sys.argv[2])

