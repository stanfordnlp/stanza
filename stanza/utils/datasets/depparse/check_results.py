"""
A small script to report the dev/test scores from a depparse run, along with averaging multiple runs at once.

Uses the expected log format from the depparse.  Will not work otherwise.
"""

import sys

def main():
    filenames = sys.argv[1:]
    if len(filenames) == 0:
        return

    dev_scores = []
    test_scores = []

    for filename in filenames:
        with open(filename, encoding="utf-8") as fin:
            lines = fin.readlines()
            dev_score = None
            test_score = None
            for line in lines:
                if line.find("Parser score") < 0:
                    continue
                score = float(line.strip().split()[-1])
                if "dev" in line:
                    dev_score = score
                elif "test" in line:
                    test_score = score
                else:
                    raise AssertionError("Did the parser score layout change?  Got an unexpected score line in %s" % filename)
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
            print("%s     %s  %s" % (filename, dev_score, test_score))

    if len(dev_scores) > 0:
        dev_score = sum(dev_scores) / len(dev_scores)
        print("Avg dev score: %.2f" % dev_score)
    if len(test_scores) > 0:
        test_score = sum(test_scores) / len(test_scores)
        print("Avg dev score: %.2f" % test_score)

if __name__ == '__main__':
    main()
