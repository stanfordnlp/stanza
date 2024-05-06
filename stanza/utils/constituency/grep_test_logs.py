import subprocess
import sys

filenames = sys.argv[1:]

total_score = 0.0
num_scores = 0

for filename in filenames:
    grep_cmd = ["grep", "F1 score.*test.*", filename]
    grep_result = subprocess.run(grep_cmd, stdout=subprocess.PIPE, encoding="utf-8")
    grep_result = grep_result.stdout.strip()
    if not grep_result:
        print("{}: no result".format(filename))
        continue

    score = float(grep_result.split()[-1])
    print("{}: {}".format(filename, score))
    total_score += score
    num_scores += 1

if num_scores > 0:
    avg = total_score / num_scores
    print("Avg: {}".format(avg))
