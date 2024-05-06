import subprocess
import sys

iteration = sys.argv[1]
filenames = sys.argv[2:]

total_score = 0.0
num_scores = 0

for filename in filenames:
    grep_cmd = ["grep", "Dev score.* %s[)]" % iteration, "-A1", filename]
    grep_result = subprocess.run(grep_cmd, stdout=subprocess.PIPE, encoding="utf-8")
    grep_result = grep_result.stdout.strip()
    if not grep_result:
        max_cmd = ["grep", "Dev score", filename]
        max_result = subprocess.run(max_cmd, stdout=subprocess.PIPE, encoding="utf-8")
        max_result = max_result.stdout.strip()
        if not max_result:
            print("{}: no result".format(filename))
        else:
            max_it = max_result.split("\n")[-1]
            max_it = int(max_it.split(":")[0].split("(")[-1][:-1])
            epoch_finished_string = "Epoch %d finished" % max_it
            finish_cmd = ["grep", epoch_finished_string, filename]
            finish_result = subprocess.run(finish_cmd, stdout=subprocess.PIPE, encoding="utf-8")
            finish_result = finish_result.stdout.strip()
            finish_time = finish_result.split(" INFO")[0]
            print("{}: no result.  max iteration: {}   finished at {}".format(filename, max_it, finish_time))
    else:
        grep_result = grep_result.split("\n")[-1]
        score = float(grep_result.split(":")[-1])
        best_iteration = int(grep_result.split(":")[-2][-6:-1])
        print("{}: {}  ({})".format(filename, score, best_iteration))
        total_score += score
        num_scores += 1

if num_scores > 0:
    avg = total_score / num_scores
    print("Avg: {}".format(avg))

