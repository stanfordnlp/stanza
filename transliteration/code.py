
"""
# read xml files from a directory and get the tags s1

import os
import xml.etree.ElementTree as ET
import re
import emoji
from collections import defaultdict
from Levenshtein import ratio


# compare the similarity between two words
def compare(word1, word2):
    # if the words are equal return 1
    if word1 == word2:
        return 1
    # if the words are not equal return the similarity between them
    else:
        return ratio(word1, word2)


def clean(text):
    # lower case
    text = text.lower()
    # remove emojis
    text = emoji.replace_emoji(text)
    # replace a word that contains a dot with a space between them
    text = re.sub(r'([a-zA-Z]+)\.([a-zA-Z]+)', r'\1 \2', text)
    # replace 3' with 1
    # remove emoticons that look like hearts <3 or <<3  or <333
    text = re.sub(r'<+3+', '', text)
    # remove emoticons that looks like :-) or :-( or :(
    text = re.sub(
        r'[:;=0oO>\\\/][\-o\*\'\[\]\|\:]*[())pPdD38xXGgyYoOsSIi/\:\}\{@\|\\*]{1,2}', '', text)
    # remove 3-)
    text = re.sub(
        r'\s[38][\-o\*\[\]\|\:]*[())pPdD38xXGgyYoOsS/\:\}\{@\|\\*]$', '', text)
    text = re.sub(
        r'\s[38][\-o\*\[\]\|\:]*[())pP38xXGgyYsS/\:\}\{@\|\\*]\s', '', text)
    # remove emoticons like xD or XD or xD
    text = re.sub(r'[xX][dD$]+', '', text)
    # remove emoticons that look like ._. or x_x or -_-
    text = re.sub(r'[.\-xX][_|.]+[.\-xX]', '', text)
    # remove emoticons that look like o.O or O.o or o.O
    text = re.sub(r'[oO]+[._-]+[oO]+', '', text)
    # remove emoticons that look like ^_^ or >_< or >_<
    text = re.sub(r'[><]+[._-]+[><]+', '', text)
    # remove emoticons that look like *_* or *_*
    text = re.sub(r'[*]+[._-]+[*]+', '', text)
    # remove punctuation except '
    text = re.sub(r'[^\w\s\']', '', text)
    # remove repeated characters
    text = re.sub(r'(.)\1+', r'\1\1', text)
    return text


def tokenize(text):
    return text.split()


counter = 0
corpus = []
tags = []
arz2ar = defaultdict(set)
with open("artext.txt", "w") as f:
    for filename in sorted(os.listdir("su_xml")):
        f.writelines(filename + "\n")
        tree = ET.parse(os.path.join("su_xml", filename))
        root = tree.getroot()
        for s1 in root.iter('su'):
            arz = []
            names = []
            # get subtag source
            for i, source in enumerate(s1.iter('token')):
                # get text
                text = source.text
                # clean text
                text = clean(text)
                if 'tag' in source.attrib:
                    # add to tags
                    tags.append(source.attrib['tag'])
                # if there is attribute tag
                if ('tag' in source.attrib) and (source.attrib['tag'] == 'name' or source.attrib['tag'] == 'arabicName'):
                    # add to names
                    names.append(i)
                # set arz
                if text != "":
                    arz.append(text)
            f.writelines(" ".join(arz) + "\n")
            for source in s1.iter('corrected_transliteration'):
                # get text
                text = source.text
                # remove punctuation except '
                text = re.sub(r'[^\w\s]', '', text)
                # tokenize text
                text = tokenize(text)
                for i in range(len(text)):
                    if i in names:
                        text[i] = "[" + text[i] + "]"
                # write to file
                if text != []:
                    f.writelines(" ".join(text) + "\n")
                if len(arz) != len(text):
                    print("Bad sentence: " + str(arz) + " " + str(text))
                    counter += 1
                if len(arz) == len(text):
                    for i in range(len(arz)):
                        arz2ar[arz[i]].add(text[i])
print(len(arz2ar))
print("Bad sentences: " + str(counter))
print(set(tags))
# get input from user
while True:
    # ge the list of keys
    keys = list(arz2ar.keys())
    sen = input("Enter a sentence: ")
    for word in sen.split():
        maxword = ""
        # in each key count the number of common words between the input and the key
        for i in range(len(keys)):
            if compare(word, keys[i]) > compare(word, maxword):
                maxword = keys[i]
        print(arz2ar[maxword])
"""

"""

import stanza
import os
import xml.etree.ElementTree as ET
import re

corpus = []
for filename in sorted(os.listdir("su_xml")):
    tree = ET.parse(os.path.join("su_xml", filename))
    root = tree.getroot()
    for s1 in root.iter('su'):
        for source in s1.iter('retokenized_transliteration'):
            # get text
            text = source.text
            # remove punctuation except '
            text = re.sub(r'[^\w\s]', '', text)
            # add to corpus
            corpus.append(text)

nlp = stanza.Pipeline(
    lang='ar', processors='tokenize,mwt,pos,lemma,depparse')
# create a new txt file
print(len(corpus))
with open("arabic.txt", "w") as f:
    # for each sentence in corpus
    for i, sen in enumerate(corpus):
        doc = nlp(sen)
        for sent in doc.sentences:
            for word in sent.words:
                # write the sentence to the file
                f.write(
                    f'{word.id};{word.text};{word.head};{sent.words[word.head-1].text if word.head > 0 else "root"};{word.deprel};{word.lemma};{word.upos};{word.xpos};{word.feats};{word.misc};{word.deps}')
                f.write("\n")
            f.write("\n")
        print(i)
"""


"""
s = set()
with open("postotal.txt", "r") as f:
    lines = f.readlines()
    for i in range(0, len(lines), 13):
        if "NOUN_PROP" in lines[i+9].split() or "FOREIGN" in lines[i+9].split():
            s.add(lines[i].split()[2])


with open("names.txt", "w") as namesF:
    for word in s:
        namesF.writelines(word + "\n")
print(len(s))
"""

"""

import os
with open("ldc2018/postotalbefore.txt", "w") as f:
    for filename in sorted(os.listdir("ldc2018/before")):
        try: 
            # read txt file
            with open(os.path.join("ldc2018/before", filename), "r") as posfile:
                # get lines
                lines = posfile.readlines()
                for i, line in enumerate(lines):
                    # write to file
                    if i % 13 == 2: 
                        f.writelines(line[:-1] + ";" + filename + "\n")
                    else:
                        f.writelines(line)
        except:
            continue

with open("ldc2018/postotalafter.txt", "w") as f:
    for filename in sorted(os.listdir("ldc2018/after")):
        try: 
            # read txt file
            with open(os.path.join("ldc2018/after", filename), "r") as posfile:
                # get lines
                lines = posfile.readlines()
                for i, line in enumerate(lines):
                    # write to file
                    if i % 10 == 3: 
                        f.writelines(line[:-1] + ";" + filename + "\n")
                    else:
                        f.writelines(line)
        except:
            continue
"""

"""
s = set()
with open("names.txt", "r") as f:
    lines = f.readlines()
    s = set([line.strip() for line in lines])
print(s)
with open("corpus.txt", "w") as namesF:
    with open("postotal.txt", "r") as f:
        lines = f.readlines()
        sen = []
        for i in range(0, len(lines), 13):
            if lines[i+3].split()[1].split("-")[0] == "0":
                namesF.writelines(" ".join(sen))
                namesF.writelines("\n")
                sen = []
            if "NOUN_PROP" in lines[i+9].split():
                sen.append("[" + lines[i].split()[2] + "]")
                print(lines[i].split()[2])
            else:
                sen.append(lines[i].split()[2])
"""

"""
# open the file postotalafter.txt
with open("corpusNER.txt", "w") as namesF:
    with open("postotalafter.txt", "r") as f:
        lines = f.readlines()
        sen = []
        contains = False
        nowords = []  # "نو", "من", "يأس",
        # "تودي", "توداي", "سوري", "يا", "تو", "آه", "يو", "إيه", "أيه", "عن"]
        names = {}
        for i in range(0, len(lines), 10):
            if lines[i+4].split()[1].split(",")[0] == "0":
                if contains:
                    namesF.writelines(" ".join(sen))
                    namesF.writelines("\n")
                sen = []
                contains = False
            if ("NOUN_PROP" in lines[i+7].split()[1]) and lines[i].split()[2] not in nowords:
                sen.append("[" + lines[i].split()[2] + "]")
                names[lines[i].split()[2]] = names.get(
                    lines[i].split()[2], 0) + 1
                contains = True
            else:
                if len(lines[i].split()) == 3:
                    sen.append(lines[i].split()[2])

    print(len(names))
    print(sorted(names.items(), key=lambda x: x[1], reverse=True))
"""


"""
import stanza
# download the arabic model
stanza.download('ar')
# create a pipeline
nlp = stanza.Pipeline(
    lang='ar', processors='tokenize,mwt,pos,lemma,depparse')
# tokenize a sentence
doc = nlp('إيه اخبار السقعة عندكم')
# print words and lemmas
for sent in doc.sentences:
    for word in sent.words:
        print(f'{word.text}')
"""

"""
import re
# open the file postotalafter.txt
with open("corpusTokenized.txt", "w") as namesF:
    with open("postotalafter.txt", "r") as f:
        lines = f.readlines()
        sen = []
        for i in range(0, len(lines), 10):
            if lines[i+4].split()[1].split(",")[0] == "0":
                if sen != []:
                    namesF.writelines(" ".join(sen))
                    namesF.writelines("\n")
                    sen = []
            if len(lines[i].split()) == 3:
                # remove punctuation
                text = lines[i].split()[2]
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'(.)\1+', r'\1\1', text)
                sen.append(text)
                if sen[-1] == "":
                    sen.pop()
"""


"""

# Skipgram model :
# train the model
import fasttext
model = fasttext.train_unsupervised(
    "corpusTokenizedTotal.txt", model='skipgram', dim=100, epoch=5000)
# save the model
model.save_model("skipgram.bin")
"""


"""
from gensim.models import FastText
model = FastText.load_fasttext_format('skipgram.bin')
print(model.wv.most_similar("محمود"))
"""

"""
# got throught pos after and get words that have det
with open("postotalafter.txt", "r") as f:
    dets = set()
    lines = f.readlines()
    for i in range(0, len(lines), 10):
        if lines[i+7].split()[1] == "DET":
            dets.add(lines[i].split()[2])
    print(dets)
"""

"""
import os
import random


with open("tags.txt", "r") as f:
    lines = f.readlines()
    utags = {
        "JJR": "ADJ",
        "JJ": "ADJ",
        "VN": "ADJ",
        "RB": "ADV",
        "WRB": "ADV",
        "CC": "CCONJ",
        "DT": "DET",
        "UH": "INTJ",
        "NOUN_QUANT": "NOUN",
        "VBG": "NOUN",
        "NN": "NOUN",
        "NNS": "NOUN",
        "ADJ_NUM": "NUM",
        "CD": "NUM",
        "RP": "PART",
        "PRP": "PRON",
        "PRP$": "PRON",
        "WP": "PRON",
        "NNP": "PROPN",
        "NNPS": "PROPN",
        "VB": "VERB",
        "VBD": "VERB",
        "VBN": "VERB",
        "VBP": "VERB",
    }
    tags = {}
    tag = ""
    for line in lines:
        # if the first char is a ;
        if line[0] == ";":
            # get the tag
            tag = line.split()[1]
        else:
            # get the word
            word = line.split("\t")[0]
            if word == "\n":
                continue
            # add the word to the tag
            tags[word] = utags.get(tag, "X")

tags["PREP"] = "ADP"
tags["SUB_CONJ"] = "SCONJ"
tags["PUNC"] = "PUNCT"
tags[";"] = "PUNCT"




# get all the file names in the before folder
filenames = []
for filename in sorted(os.listdir("before")):
    filenames.append(filename)
# split the files into train, dev and test
random.shuffle(filenames)
train, dev, test = filenames[:int(len(filenames)*0.8)], filenames[int(len(filenames)*0.8):int(len(filenames)*0.9)], filenames[int(len(filenames)*0.9):]

# open the files to write to
with open("tokenizer/t.train.gold.conllu", "w") as tokenizerDataTRAIN:
    with open("tokenizer/t.dev.gold.conllu", "w") as tokenizerDataDEV:
        with open("tokenizer/t.test.gold.conllu", "w") as tokenizerDataTEST:
            # open the files to read from
            with open("postotalafter.txt", "r") as after:
                with open("postotalbefore.txt", "r") as before:
                    # match after and before with their word position number
                    # for each word in before
                    beforelines = before.readlines()
                    afterlines = after.readlines()
                    last = 0
                    senCounter = 1
                    prevind = -1
                    prevword=""
                    # get the file to write to, and see whether it is in train, dev or test filenames created earlier 
                    if beforelines[2].split()[1].split(";")[1] in train:
                        tokenizerData = tokenizerDataTRAIN
                    elif beforelines[2].split()[1].split(";")[1] in dev:
                        tokenizerData = tokenizerDataDEV
                    else:
                        tokenizerData = tokenizerDataTEST
                    for i in range(0, len(beforelines), 13):
                        
                        # get the word
                        word = beforelines[i].split()[2]
                        # get the indices
                        indices = beforelines[i+3].split()[1].split("-")

                        if int(indices[0]) < int(prevind) and i != 0:
                            
                            # seek -1 to the end of the line

                            if prevword[-1] not in [".", "!", "?", "؟"]:
                                tokenizerData.seek(tokenizerData.tell() - 2, os.SEEK_SET)
                                tokenizerData.write("SpaceAfter=No\n")
                                tokenizerData.write(
                                "{0}\t.\t.\tPUNCT\tPUNC\t_\t1\tdep\t_\t_".format(senCounter) + "\n")
                            tokenizerData.write("\n")
                            if beforelines[i+2].split()[1].split(";")[1] in train:
                                tokenizerData = tokenizerDataTRAIN
                            elif beforelines[i+2].split()[1].split(";")[1] in dev:
                                tokenizerData = tokenizerDataDEV
                            else:
                                tokenizerData = tokenizerDataTEST
                            prevind=-1
                            senCounter = 1
                        # read from after until the indices are found
                        counter = 0
                        words = []
                        for j in range(last, len(afterlines), 10):
                            # read all the words till the end index
                            # get the word
                            if len(afterlines[j].split()) != 3:
                                continue
                            afterword = afterlines[j].split()[2]
                            # get the tag
                            tag = afterlines[j+7].split()[1]
                            words.append((senCounter, afterword, tag, afterlines[j+4].split()[1].split(",")))
                            counter += 1
                            senCounter += 1
                            if afterlines[j+4].split()[1].split(",")[1] == indices[1]:
                                last = j + 10
                                break
                        if counter > 1:
                            tokenizerData.write(
                                str(words[0][0]) + "-" + str(words[-1][0]) + "\t" + word + "\t_\t_\t_\t_\t_\t_\t_\t_" + "\n")
                        for w in words:
                            if prevind == w[3][0]:
                                tokenizerData.seek(tokenizerData.tell() - 2, os.SEEK_SET)
                                tokenizerData.write("SpaceAfter=No\n")
                            tokenizerData.write(
                                "{0}\t{1}\t_\t{2}\t{3}\t_\t{4}\t{5}\t_\t_".format(w[0], w[1], tags[w[2]], w[2], "0" if w[0] == 1 else "1", "root" if w[0] == 1 else "dep") + "\n")
                            prevind = w[3][1]
                            prevword = w[1]
            if prevword[-1] not in [".", "!", "?", "؟"]:
                tokenizerData.seek(tokenizerData.tell() - 2, os.SEEK_SET)
                tokenizerData.write("SpaceAfter=No\n")
                tokenizerData.write(
                "{0}\t.\t.\tPUNCT\tPUNC\t_\t1\tdep\t_\t_".format(senCounter))


# creating the corpus train, dev, test txt files for the tokenizer

# each file is treated as a different paragraph. All the sentences within a paragraph either end with . or ! or ? or ؟ (arabic question mark)
# The paragraphs are separated by a \n
import os
import xml.etree.ElementTree as ET
import re
# these lists will contain all the paragraphs for train, dev and test
trainlist, devlist, testlist = [], [], []
# su_xml contains all raw text files
for filename in sorted(os.listdir("ldc2021/su_xml")):
    # this "sentences" list will contain all the sentences in the paragraph/file
    sentences = []
    tree = ET.parse(os.path.join("ldc2021/su_xml", filename))
    root = tree.getroot()
    sub = set()
    for su in root.iter('su'):
        # get all the subtags in su
        subtags = su.findall("*")
        # get the text name of the subtags
        subtags = [subtag.tag for subtag in subtags]
        for subtag in subtags:
            sub.add(subtag)
    if sorted(list(sub)) == ['body', 'messages']:
        for su in root.iter('su'):
            # get all the subtags that are called body
            subtags = su.findall("body")
            # get the attributes of the subtags
            if len(subtags) == 1:
                #get text 
                if subtags[0].text is not None:
                    sentences.append(subtags[0].text)
            elif len(subtags) == 2:
                if subtags[0].text is not None:
                    sentences.append(subtags[1].text)
    elif "normalized_transliteration" in sub:
        for su in root.iter('su'):
            # get all the subtags that are called body
            subtags = su.findall("normalized_transliteration")
            # get text
            if subtags[0].text is not None:
                sentences.append(subtags[0].text)
    elif "retokenized_transliteration" in sub:
        for su in root.iter('su'):
            # get all the subtags that are called body
            subtags = su.findall("retokenized_transliteration")
            if len(subtags) == 0:
               continue
            # get text
            if subtags[0].text is not None:
                sentences.append(subtags[0].text)
    elif sorted(list(sub)) == ['messages', 'source']:
        for su in root.iter('su'):
            # get all the subtags that are called body
            subtags = su.findall("source")
            # get text
            if subtags[0].text is not None:
                sentences.append(subtags[0].text)
    # remove all the weird characters
    sentences = [re.sub(r"\u200E", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u200F", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u202C", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u202A", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u202B", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u202D", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u202E", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u2066", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u2067", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u2068", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u2069", "", sentence) for sentence in sentences]
    sentences = [re.sub(r"\u061C", "", sentence) for sentence in sentences]
    # remove trailing spaces
    sentences = [sentence.strip() for sentence in sentences]
    # join sentences on ". " but only if the sentence doens't end with a punctuation
    sentences = [sentence + " " if sentence[-1] in [".", "!", "?", "؟"] else sentence + ". " for sentence in sentences]
    if filename.replace(".su.xml", ".txt") in train:
        trainlist.append("".join(sentences)[:-1])
    elif filename.replace(".su.xml", ".txt") in dev:
        devlist.append("".join(sentences)[:-1])
    elif filename.replace(".su.xml", ".txt") in test:
        testlist.append("".join(sentences)[:-1])

with open("tokenizer/t.train.txt", "w") as tokenizerTxt:
    tokenizerTxt.write("\n".join(trainlist))
with open("tokenizer/t.dev.txt", "w") as tokenizerTxt:
    tokenizerTxt.write("\n".join(devlist))
with open("tokenizer/t.test.txt", "w") as tokenizerTxt:
    tokenizerTxt.write("\n".join(testlist))




import prepare_tokenizer_data
def mwt_name(base_dir, short_name, dataset):
    return os.path.join(base_dir, f"{short_name}-ud-{dataset}-mwt.json")

def prepare_tokenizer_dataset_labels(input_txt, input_conllu, tokenizer_dir, short_name, dataset):
    prepare_tokenizer_data.main([input_txt,
                                 input_conllu,
                                 "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                 "-m", mwt_name(tokenizer_dir, short_name, dataset)])

def prepare_tokenizer_treebank_labels(tokenizer_dir, short_name):

    for dataset in ["train", "dev", "test"]:
        output_txt = f"{tokenizer_dir}/{short_name}.{dataset}.txt"
        output_conllu = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
        try:
            prepare_tokenizer_dataset_labels(output_txt, output_conllu, tokenizer_dir, short_name, dataset)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print("Failed to convert %s to %s" % (output_txt, output_conllu))
            raise

prepare_tokenizer_treebank_labels("tokenizer", "t")

"""


"""
import stanza

# download the arabic model
stanza.download('ar')
# depency parser
nlp = stanza.Pipeline('ar', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)
def ss(sen):    
    doc = nlp(sen)
    print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')

"""

"""

from stanza.utils.datasets.contract_mwt import contract_mwt
import shutil

short_name = "arz_penn"
tokenizer_dir = "tokenizer"
mwt_dir = "mwt"
def copy_conllu(tokenizer_dir, mwt_dir, short_name, dataset, particle):
    input_conllu_tokenizer = f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu"
    input_conllu_mwt = f"{mwt_dir}/{short_name}.{dataset}.{particle}.conllu"
    shutil.copyfile(input_conllu_tokenizer, input_conllu_mwt)

copy_conllu(tokenizer_dir, mwt_dir, short_name, "train", "in")
copy_conllu(tokenizer_dir, mwt_dir, short_name, "dev", "gold")
copy_conllu(tokenizer_dir, mwt_dir, short_name, "test", "gold")

contract_mwt(f"{mwt_dir}/{short_name}.dev.gold.conllu", f"{mwt_dir}/{short_name}.dev.in.conllu")
contract_mwt(f"{mwt_dir}/{short_name}.test.gold.conllu", f"{mwt_dir}/{short_name}.test.in.conllu")
"""

"""
import stanza
pipe=stanza.Pipeline("arz",
                         processors="tokenize,mwt",
                         allow_unknown_language=True,
                         tokenize_model_path="saved_models/tokenize/arz_penn_tokenizer.pt",
                         mwt_model_path="saved_models/mwt/arz_penn_mwt_expander.pt",
                         download_method=None)
# tokenize the following sentence
doc = pipe("والمحميه جميله فشخ")
print(doc)

"""

"""

# get all the files in the directory su_xml
import os

def get_files(directory, remove):
    files = set()
    for filename in os.listdir(directory):
        files.add(filename.replace(remove, ""))
    return files

print(len(get_files("/Users/ammas/stanford/research/bolt_arz--df/data/su_xml","")))
"""