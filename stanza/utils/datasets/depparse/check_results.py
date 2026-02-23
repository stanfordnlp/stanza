"""
A small script to report the dev/test scores from a depparse run, along with averaging multiple runs at once.

Uses the expected log format from the depparse.  Will not work otherwise.
"""

import argparse
import re
import sys

dev_re = re.compile(".*INFO: step ([0-9]+).*dev_score = ([.0-9]+).*")


def main():
    parser = argparse.ArgumentParser(description="Grep through a list of files looking for the final results or best results up to a point")
    parser.add_argument("filenames", nargs="+", help="Files to check")
    parser.add_argument("--step", default=None, type=int, help="If set, stop checking at this step")
    args = parser.parse_args()

    filenames = args.filenames
    if len(filenames) == 0:
        return

    dev_scores = []
    test_scores = []

    best_step = None
    for filename in filenames:
        with open(filename, encoding="utf-8") as fin:
            lines = fin.readlines()
            dev_score = None
            test_score = None
            for line in lines:
                if line.find("Parser score") >= 0:
                    score = float(line.strip().split()[-1])
                    if "dev" in line:
                        dev_score = score
                    elif "test" in line:
                        test_score = score
                    else:
                        raise AssertionError("Did the parser score layout change?  Got an unexpected score line in %s" % filename)
                    best_step = None
                dev_match = dev_re.match(line)
                if dev_match:
                    step = int(dev_match.groups()[0])
                    if args.step is not None and step > args.step:
                        break
                    score = float(dev_match.groups()[1]) * 100
                    if dev_score is None or score > dev_score:
                        dev_score = score
                        best_step = step
            if dev_score is None:
                dev_score = "N/A"
            else:
                dev_scores.append(dev_score)
                dev_score = "%.2f" % dev_score
            if test_score is None:
                test_score = "N/A"
            else:
                test_scores.append(test_score)
                test_score = "%.2f" % test_score
            if best_step is not None:
                print("%s     %s  (%d)" % (filename, dev_score, best_step))
            else:
                print("%s     %s  %s" % (filename, dev_score, test_score))

    if len(dev_scores) > 0:
        dev_score = sum(dev_scores) / len(dev_scores)
        print("Avg dev score:  %.2f" % dev_score)
    if len(test_scores) > 0:
        test_score = sum(test_scores) / len(test_scores)
        print("Avg test score: %.2f" % test_score)

if __name__ == '__main__':
    main()
