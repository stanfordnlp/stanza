
import os
import random
import xml.etree.ElementTree as ET
import re
import prepare_tokenizer_data
from collections import defaultdict

folder = "ldc2018"

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

#extras
tags["PREP"] = "ADP"
tags["SUB_CONJ"] = "SCONJ"
tags["PUNC"] = "PUNCT"
tags[";"] = "PUNCT"




# get all the file names in the before folder
filenames = []
for filename in sorted(os.listdir(folder + "/before")):
    filenames.append(filename)
# split the files into train, dev and test
random.seed(10)
random.shuffle(filenames)
train, dev, test = filenames[:int(len(filenames)*0.8)], filenames[int(len(filenames)*0.8):int(len(filenames)*0.9)], filenames[int(len(filenames)*0.9):]

# open the files to write to
with open("tokenizer/arz_penn.train.gold.conllu", "w") as tokenizerDataTRAIN:
    with open("tokenizer/arz_penn.dev.gold.conllu", "w") as tokenizerDataDEV:
        with open("tokenizer/arz_penn.test.gold.conllu", "w") as tokenizerDataTEST:
            # open the files to read from
            with open(folder + "/postotalafter.txt", "r") as after:
                with open(folder + "/postotalbefore.txt", "r") as before:
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



######## creating the corpus train, dev, test txt files for the tokenizer

# each file is treated as a different paragraph. All the sentences within a paragraph either end with . or ! or ? or ؟ (arabic question mark)
# The paragraphs are separated by a \n

# these lists will contain all the paragraphs for train, dev and test
trainlist, devlist, testlist = [], [], []

# dictionary of dictionaries. Each dictionary will contain list
ninc = defaultdict(lambda: defaultdict(list))

################ for ldc2018 only
if folder == "ldc2018":

    # read the not-included file
    with open(folder + "/not-included.txt", "r") as notincluded:
        # read the lines
        lines = notincluded.readlines()
        # for each line
        for line in lines:
            fname = line.split(":")[0]
            # get the sentence number
            sennum = int(line.split(":")[1].split("[")[0])
            # get the indices
            indices = line.split(":")[1].split("[")[1].split("]")[0]
            # string
            string = line.split(":")[1].split("[")[1].split("]")[1]
            # indices are either [1] which means from 1 to the end of the sentence or [1,2] which means from 1 to 2
            if "," in indices:
                start = int(indices.split(",")[0])
                end = int(indices.split(",")[1])
            else:
                start = int(indices)
                end = -1
            # add the sentence number and the indices to the dictionary
            ninc[fname][sennum].append((start, end, string))
################

# su_xml contains all raw text files
# they are structured as follows:
# <conversation>
#   <su>  # this is a sentence unit and it contains a message
#      here is can either be <messages> and <body> (we want the body)
#      or <messages> and <body type="original"> and <body type="transliterated"> (we want the transliterated one)
#      or <messages> and <source> (we want the source)
#      or {'annotated_arabizi', 'source', 'corrected_transliteration', 'normalized_transliteration', 'messages', 'auto_transliteration'} (we want the normalized_transliteration)
#      or {'annotated_arabizi', 'source', 'corrected_transliteration', 'retokenized_transliteration', 'messages', 'auto_transliteration'} (we want the retokenized_transliteration)
#      or text is written directly
for filename in sorted(os.listdir(folder + "/su_xml")):
    # skip ds_store
    if filename == ".DS_Store":
        continue
    # this "sentences" list will contain all the sentences in the paragraph/file
    sentences = []
    tree = ET.parse(os.path.join(folder + "/su_xml", filename))
    root = tree.getroot()
    sub = set()
    for su in root.iter('su'):
        # get all the subtags in su
        subtags = su.findall("*")
        # get the text name of the subtags
        subtags = [subtag.tag for subtag in subtags]
        for subtag in subtags:
            sub.add(subtag)
    def formatted_sentence(s):
        if s is None:
            return None

        # remove all the weird characters and the spaces at the beginning and end of the sentence
        s = re.sub(r"[\u202A-\u202E]|[\u200E-\u200F]|[\u2066-\u2069]|\u061C", " ", s).strip()
        if folder == "ldc2018":
            s = s.replace("\"", " ")
            s = s.replace(")", " ")
            s = s.replace("(", " ")
            s = s.replace("+", " ")
            s = s.replace("=", " ")
            s = s.replace("*", " ")
        if s == "":
            return None
        # join sentences on ". " but only if the sentence doens't end with a punctuation
        if folder == "ldc2021":
            s = s + " " if s[-1] in [".", "!", "?", "؟"] else s + ". "
        return s

    # check the format of the xml and see which one is it according to the above formats
    if folder == "ldc2021":
        if sorted(list(sub)) == ['body', 'messages']:
            for su in root.iter('su'):
                # get all the subtags that are called body inside the su tag
                subtags = su.findall("body")
                # if it's only <body>
                if len(subtags) == 1:
                    #get text 
                    if formatted_sentence(subtags[0].text) is not None:
                        sentences.append(formatted_sentence(subtags[0].text))
                # if it's <body type="original"> and <body type="transliterated">
                elif len(subtags) == 2:
                    # get text
                    if formatted_sentence(subtags[1].text) is not None:
                        sentences.append(formatted_sentence(subtags[1].text))
        elif "normalized_transliteration" in sub:
            for su in root.iter('su'):
                # get all the subtags that are called normalized_transliteration
                subtags = su.findall("normalized_transliteration")
                # get text
                if formatted_sentence(subtags[0].text) is not None:
                    sentences.append(formatted_sentence(subtags[0].text))
        elif "retokenized_transliteration" in sub:
            for su in root.iter('su'):
                # get all the subtags that are called retokenized_transliteration
                subtags = su.findall("retokenized_transliteration")
                if len(subtags) == 0:
                    continue
                # get text
                if formatted_sentence(subtags[0].text) is not None:
                    sentences.append(formatted_sentence(subtags[0].text))
        elif sorted(list(sub)) == ['messages', 'source']:
            for su in root.iter('su'):
                # get all the subtags that are called source
                subtags = su.findall("source")
                # get text
                if formatted_sentence(subtags[0].text) is not None:
                    sentences.append(formatted_sentence(subtags[0].text))
    elif folder == "ldc2018":
        for i, su in enumerate(root.iter('su')):
            # get su text
            if su.text is not None:
                sen = formatted_sentence(su.text)
                if filename.replace(".xml", "") in ninc:
                    if i+1 in ninc[filename.replace(".xml", "")]:
                        for start, end, string in ninc[filename.replace(".xml", "")][i+1]:
                            
                            kk = string
                            if kk.strip() != "\"" and kk.strip() != ")" and kk.strip() != "(" and kk.strip() != "+" and kk.strip() != "=" and kk.strip() != "*":
                                
                                if end == -1:
                                    sen = sen[:start]
                                else:
                                    sen = sen[:start] + " " * (end - start) + sen[end:]
                        sen = sen.strip()
                if sen != "":
                    sen = sen + " " if sen[-1] in [".", "!", "?", "؟"] else sen + ". "
                    sentences.append(sen)
   
    # concatenate and the [:-1] is to remove the last space
    sentences = "".join(sentences)[:-1]

    ################# only for ldc2021
    if "ldc2021" in folder:
        ### some sentences that have extra characters that are incorrectly annotates
        # remove those extra characters case by case
        sentences = sentences.replace("إيه رأيك في الرياض ه دي", "إيه رأيك في الرياض دي")
        sentences = sentences.replace("اهو كده الرسائل الحلوه عشان كانا نروح من غ ير مطر", "اهو كده الرسائل الحلوه عشان كانا نروح من ير مطر")
        sentences = sentences.replace("حلوة دى ؟؟؟؟؟؟ ج", "حلوة دى ؟؟؟؟؟؟")
        sentences = sentences.replace("اهو كده الكلام وانت دخلت في الخط اهو بس بعد اذنك كمالته هي ولو ياك ل نصه عشان انا جعان والجعان يحلم باكل العيش", "اهو كده الكلام وانت دخلت في الخط اهو بس بعد اذنك كمالته هي ولو ياك نصه عشان انا جعان والجعان يحلم باكل العيش")
        sentences = sentences.replace("الحمد لله كانت كويسه خالص وامبارح كنا معزومين عند ناس اصحابنا وكان ت قعده حلوه", "الحمد لله كانت كويسه خالص وامبارح كنا معزومين عند ناس اصحابنا وكان قعده حلوه")

        # if (2/2) is in s. 
        sentences = sentences.replace("(2/2)", "2/2")
        sentences = sentences.replace("علشان يعرف 2/2يتحرك", "علشان يعرف (2/2)يتحرك")
        sentences = sentences.replace("بس دى 2/2اقل", "بس دى (2/2)اقل")
        sentences = sentences.replace("لأ. ]توطير", "لأ ]توطير")
        sentences = sentences.replace("حلوة دى ؟؟؟؟؟؟.", "حلوة دى ؟؟؟؟؟؟")
    #################


    # check if the file is in train, dev or test then add the paragraph/file to the corresponding list
    if filename.replace(".su.xml", ".txt") in train or filename.replace(".xml", ".txt") in train:
        trainlist.append(sentences)
    elif filename.replace(".su.xml", ".txt") in dev or filename.replace(".xml", ".txt") in dev:
        devlist.append(sentences)
    elif filename.replace(".su.xml", ".txt") in test or filename.replace(".xml", ".txt") in test:
        testlist.append(sentences)
# write the paragraphs to the corresponding files separated by a \n
with open("tokenizer/arz_penn.train.txt", "w") as tokenizerTxt:
    tokenizerTxt.write("\n".join(trainlist))
with open("tokenizer/arz_penn.dev.txt", "w") as tokenizerTxt:
    tokenizerTxt.write("\n".join(devlist))
with open("tokenizer/arz_penn.test.txt", "w") as tokenizerTxt:
    tokenizerTxt.write("\n".join(testlist))





###### prepare the data for the tokenizer

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

prepare_tokenizer_treebank_labels("tokenizer", "arz_penn")