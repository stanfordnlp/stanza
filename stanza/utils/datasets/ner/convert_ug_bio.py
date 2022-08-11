import numpy as np
import sys
import os
import xml.etree.ElementTree as ET
import stanza

path = '/Users/arman/Desktop/CURIS/UyNeRel/uynereldata'
TAGS = ['PER', 'ORG', 'GPE', 'FAC', 'LOC', 'TTL']

TRAIN_SPLIT = 0.8
DEV_SPLIT = 0.1
TEST_SPLIT = 0.1

tagged_tuples = []

for filename in os.listdir(path):
    fd = os.path.join(path, filename)

    myroot = ET.parse(fd)
    text = myroot.find('TEXT').text

    pipe = stanza.Pipeline("Uyghur", processors="tokenize", tokenize_no_ssplit=False)  # assuming there is one line per sentence

    doc = pipe(text)
    tokens_ = []
    for sentence in doc.sentences:
        for t in sentence.tokens:
            start, end = t.start_char, t.end_char
            tokens_.append((start, end, t.text))
        tokens_.append((0, 0, 'NEW_LINE'))

    tags = myroot.find('TAGS')
    tags_lst = [tags.findall(t) for t in TAGS]
    tags_cummulative = []
    for lst_i in range(len(tags_lst)):
        tag_lst = []
        for i in tags_lst[lst_i]:
            
            start, end = i.get('spans').split('~')

            # tag_lst.append((int(start), int(end), i.get('text'), TAGS[lst_i]))
            tags_cummulative.append((int(start), int(end), i.get('text'), TAGS[lst_i]))
        
        # tags_cummulative.append(tag_lst)
        # print(f'-------------- TAG: {TAGS[lst_i]} Processed. -----------')

    tags_cummulative.sort(key=lambda x: x[0])

    if len(tags_cummulative) != 0:
        sorted_tags = [tags_cummulative[0]]
        for i in range(1, len(tags_cummulative)):
            if sorted_tags[-1][0] != tags_cummulative[i][0]:
                sorted_tags.append(tags_cummulative[i])

        tags_cummulative = sorted_tags

    for i in range(len(tokens_)):
        t = tokens_[i]
        start, end, text = t
        print(f"{i} TOKEN: {text}: start: {start}, end: {end}")


    for i in range(len(tags_cummulative)):
        tuple_ = tags_cummulative[i]
        print(f"{tuple_[0]} : {tuple_[1]}, text : {tuple_[2]}, tag: {tuple_[3]}")

    idx = 0
    i = 0

    while i < len(tokens_):
        t = tokens_[i]
        start, end = t[0], t[1]
        if start == 0 and end == 0 and t[2] == 'NEW_LINE':
            tagged_tuples.append(('\n', 'NEW_LINE'))

        if idx == len(tags_cummulative) or end <= tags_cummulative[idx][0]:
            tagged_tuples.append((t[2], 'O'))

        elif start == tags_cummulative[idx][0] or start - 1 == tags_cummulative[idx][0] or (start <= tags_cummulative[idx][0] and end <= tags_cummulative[idx][1]):
            tagged_tuples.append((t[2], f'B-{tags_cummulative[idx][3]}'))
            while end < tags_cummulative[idx][1]:
                i += 1
                t = tokens_[i]
                start, end = t[0], t[1]
                tagged_tuples.append((t[2], f'I-{tags_cummulative[idx][3]}'))   
            idx += 1

        else:
            print("RED ALERT!")
            print(f"Tried to match the {i}th token :{t[2]} -- Span: {start}~{end}")
            print(f"Next TAG to be matched: {tags_cummulative[idx][2]} -- Span: {tags_cummulative[idx][0]}~{tags_cummulative[idx][1]}, TAG: {tags_cummulative[idx][3]}")
        i += 1

    print(tagged_tuples[-1])
    tagged_tuples.pop()
    tagged_tuples.append(('\n\n', 'END_OF_DOCUMENT'))

print(tagged_tuples)
print(len(tagged_tuples))

    
#     myroot = ET.parse(fd)
#     text = myroot.find('TEXT').text
#     print(text)

#     tags = myroot.find('TAGS')
#     tags_lst = [tags.findall(t) for t in TAGS]
#     print(tags_lst)
#     for lst in tags_lst:
#         for i in lst:
#             print(i.attrib)

#     # per_lst = tags.findall('PER')
#     # for i in per_lst:
#     #     print(i.attrib)
    
#     pipe = stanza.Pipeline("Uyghur", processors="tokenize", tokenize_no_ssplit=True)  # assuming there is one line per sentence
#     doc = pipe(text)
#     sentence = doc.sentences[0]
#     for token in sentence.tokens:
#         print(token)


# print(text)

#     with open(os.path.join(path, filename), 'r') as f:
#        fd = f.read()
#        mytree = ET.parse(fd)


# myroot = mytree.getroot()

#print(myroot.find('TEXT').text)


# """
# Experimental 1057CON.XML 
# """
# exp_path = path + '/1057CON.XML'
# myroot = ET.parse(exp_path)
# text = myroot.find('TEXT').text

# pipe = stanza.Pipeline("Uyghur", processors="tokenize", tokenize_no_ssplit=False)  # assuming there is one line per sentence

# doc = pipe(text)
# tokens_ = []
# for sentence in doc.sentences:
#     for t in sentence.tokens:
#         start, end = t.start_char, t.end_char
#         tokens_.append((start, end, t.text))
#     tokens_.append((0, 0, '\n'))

# tags = myroot.find('TAGS')
# tags_lst = [tags.findall(t) for t in TAGS]
# tags_cummulative = []
# for lst_i in range(len(tags_lst)):
#     tag_lst = []
#     for i in tags_lst[lst_i]:
        
#         start, end = i.get('spans').split('~')

#         # tag_lst.append((int(start), int(end), i.get('text'), TAGS[lst_i]))
#         tags_cummulative.append((int(start), int(end), i.get('text'), TAGS[lst_i]))
    
#     # tags_cummulative.append(tag_lst)
#     print(f'-------------- TAG: {TAGS[lst_i]} Processed. -----------')

# tags_cummulative.sort(key=lambda x: x[0])
# sorted_tags = [tags_cummulative[0]]
# for i in range(1, len(tags_cummulative)):
#     if sorted_tags[-1][0] != tags_cummulative[i][0]:
#         sorted_tags.append(tags_cummulative[i])

# tags_cummulative = sorted_tags

# tagged_tuples = []
# for i in range(65):
#     t = tokens_[i]
#     start, end, text = t
#     print(f"{i} TOKEN: {text}: start: {start}, end: {end}")


# for i in range(len(tags_cummulative)):
#     tuple_ = tags_cummulative[i]
#     print(f"{tuple_[0]} : {tuple_[1]}, text : {tuple_[2]}, tag: {tuple_[3]}")

# idx = 0
# i = 0

# while i < len(tokens_):
#     t = tokens_[i]
#     start, end = t[0], t[1]
#     if start == 0 and end == 0 and t[2] == '\n':
#         tagged_tuples.append(('\n', 'NEW_LINE'))

#     if idx == len(tags_cummulative) or end <= tags_cummulative[idx][0]:
#         tagged_tuples.append((t[2], 'O'))

#     elif start == tags_cummulative[idx][0] or start - 1 == tags_cummulative[idx][0]:
#         tagged_tuples.append((t[2], f'B-{tags_cummulative[idx][3]}'))
#         while end < tags_cummulative[idx][1]:
#             i += 1
#             t = tokens_[i]
#             start, end = t[0], t[1]
#             tagged_tuples.append((t[2], f'I-{tags_cummulative[idx][3]}'))   
#         idx += 1

#     else:
#         print("RED ALERT!")
#         print(f"Tried to match the {i}th token :{t[2]} -- Span: {start}~{end}")
#         print(f"Next TAG to be matched: {tags_cummulative[idx][0]} -- Span: {tags_cummulative[idx][1]}~{tags_cummulative[idx][2]}, TAG: {tags_cummulative[idx][3]}")
#     i += 1

# tagged_tuples.pop()
# print(tagged_tuples)
# print(len(tagged_tuples))

# output_dir = '/Users/arman/Desktop/CURIS/UyNeRel/bio_uynereldata'
# SHARDS = ('train', 'dev', 'test')

# def split_write_bio(dataset, short_name, shard):
#     output_filename = os.path.join(output_dir, "%s.%s.bio" % (short_name, shard))
#     with open(output_filename, "w", encoding="utf-8") as fout:
#         for word, tag in dataset:
#             fout.write("%s\t%s\n", word, tag)
#             for sentence in dataset:
#                 for word in sentence:
#                     fout.write("%s\t%s\n", word, tag)
#                 fout.write("\n")



