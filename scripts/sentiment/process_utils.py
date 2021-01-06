import csv
import glob
import os

from collections import namedtuple

Split = namedtuple('Split', ['filename', 'weight'])

def write_list(out_filename, dataset):
    with open(out_filename, 'w') as fout:
        for line in dataset:
            fout.write(line)
            fout.write("\n")

def write_splits(out_directory, snippets, splits):
    total_weight = sum(split.weight for split in splits)
    divs = []
    subtotal = 0.0
    for split in splits:
        divs.append(int(len(snippets) * subtotal / total_weight))
        subtotal = subtotal + split.weight
    divs.append(len(snippets))

    for i, split in enumerate(splits):
        filename = os.path.join(out_directory, split.filename)
        print("Writing {}:{} to {}".format(divs[i], divs[i+1], filename))
        write_list(filename, snippets[divs[i]:divs[i+1]])

def clean_tokenized_tweet(line):
    line = list(line)
    if len(line) > 3 and line[0] == 'RT' and line[1][0] == '@' and line[2] == ':':
        line = line[3:]
    elif len(line) > 4 and line[0] == 'RT' and line[1] == '@' and line[3] == ':':
        line = line[4:]
    elif line[0][0] == '@':
        line = line[1:]
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
                text = " ".join(" ".join(token.text for token in sentence.tokens) for sentence in doc.sentences)
                num_tokens = sum(len(sentence.tokens) for sentence in doc.sentences)
                if num_tokens < 4:
                    num_short_items = num_short_items + 1
                snippets.append("%d %s" % (sentiment, text))
    print("Number of short items: {}".format(num_short_items))
    return snippets
