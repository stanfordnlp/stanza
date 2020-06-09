import csv
import glob
import os

def write_list(out_filename, dataset):
    with open(out_filename, 'w') as fout:
        for line in dataset:
            fout.write(line)
            fout.write("\n")

def clean_tokenized_tweet(line):
    line = list(line)
    if len(line) > 3 and line[0] == 'RT' and line[1][0] == '@' and line[2] == ':':
        line[0] = ' '
        line[1] = ' '
        line[2] = ' '
    if line[0][0] == '@':
        line[0] = ' '
    for i in range(len(line)):
        if line[i][0] == '@' or line[i][0] == '#':
            line[i] = line[i][1:]
        if line[i].startswith("http:") or line[i].startswith("https:"):
            line[i] = ' '
    return line

def get_scare_snippets(nlp, csv_dir_path, text_id_map, filename_pattern="*.csv"):
    num_short_items = 0

    snippets = []
    csv_files = glob.glob(os.path.join(csv_dir_path, filename_pattern))
    for csv_filename in csv_files:
        with open(csv_filename, newline='') as fin:
            cin = csv.reader(fin, delimiter='\t', quotechar='"')
            lines = list(cin)

            for line in lines:
                ann_id, begin, end, sentiment = [line[i] for i in [1, 2, 3, 6]]
                begin = int(begin)
                end = int(end)
                if sentiment.lower() == 'unknown':
                    continue
                elif sentiment.lower() == 'positive':
                    sentiment = 2
                elif sentiment.lower() == 'neutral':
                    sentiment = 1
                elif sentiment.lower() == 'negative':
                    sentiment = 0
                else:
                    raise ValueError("Tell John he screwed up and this is why he can't have Mox Opal: {}".format(sentiment))
                if ann_id not in text_id_map:
                    print("Found snippet which can't be found: {}-{}".format(csv_filename, ann_id))
                    continue
                snippet = text_id_map[ann_id][begin:end]
                doc = nlp(snippet)
                text = " ".join(sentence.text for sentence in doc.sentences)
                num_tokens = sum(len(sentence.tokens) for sentence in doc.sentences)
                if num_tokens < 4:
                    num_short_items = num_short_items + 1
                snippets.append("%d %s" % (sentiment, text))
    print("Number of short items: {}".format(num_short_items))
    return snippets
