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
                raise AssertionError("Missing dev score in %s" % filename)
            if test_score is None:
                raise AssertionError("Missing test score in %s" % filename)
            print("%s     %.2f  %.2f" % (filename, dev_score, test_score))
            dev_scores.append(dev_score)
            test_scores.append(test_score)

    dev_score = sum(dev_scores) / len(dev_scores)
    print("Avg dev score: %.2f" % dev_score)
    test_score = sum(test_scores) / len(test_scores)
    print("Avg dev score: %.2f" % test_score)

if __name__ == '__main__':
    main()
