"""
Look through 2 files, only output the common trees

pretty basic - could use some more options
"""

import sys

def main():
    in1 = sys.argv[1]
    with open(in1, encoding="utf-8") as fin:
        lines1 = fin.readlines()
    in2 = sys.argv[2]
    with open(in2, encoding="utf-8") as fin:
        lines2 = fin.readlines()

    common = [l1 for l1, l2 in zip(lines1, lines2) if l1 == l2]
    for l in common:
        print(l.strip())

if __name__ == '__main__':
    main()

