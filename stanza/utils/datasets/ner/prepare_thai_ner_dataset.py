import os

TRAIN_PATH = '/u/nlp/data/thai/LST20_Corpus/train'
EVAL_PATH = '/u/nlp/data/thai/LST20_Corpus/eval'
TEST_PATH = '/u/nlp/data/thai/LST20_Corpus/test'

TRAIN_OUTPUT_PATH = '/nlp/scr/kkotchum/NER/thai/train'
EVAL_OUTPUT_PATH = '/nlp/scr/kkotchum/NER/thai/eval'
TEST_OUTPUT_PATH = '/nlp/scr/kkotchum/NER/thai/test'

INCLUDE_SPACE_CHAR = False

paths = [(TRAIN_PATH, TRAIN_OUTPUT_PATH), (EVAL_PATH, EVAL_OUTPUT_PATH), (TEST_PATH, TEST_OUTPUT_PATH)]

for path_in, path_out in paths:
    text_list = ['/{}'.format(text) for text in os.listdir(path_in) if text[0] == 'T']

    if INCLUDE_SPACE_CHAR:
        if path_in == TRAIN_PATH:
            text_path = '/train.txt'
        elif path_in == EVAL_PATH:
            text_path = '/eval.txt'
        else:
            text_path = '/test.txt'

    else:
        if path_in == TRAIN_PATH:
            text_path = '/train_no_ws.txt'
        elif path_in == EVAL_PATH:
            text_path = '/eval_no_ws.txt'
        else:
            text_path = '/test_no_ws.txt'

    with open(path_out + text_path, 'w', encoding='utf-8') as f1:
        for text in text_list:
            lst = []
            with open(path_in + text, 'r', encoding='utf-8') as f2:
                for line in f2:
                    x = line.split()
                    if len(x) > 0:
                        if x[0] == '_' and not INCLUDE_SPACE_CHAR:
                            continue
                        else:
                            f1.write('{}\t{}'.format(x[0], x[2]))
                            f1.write('\n')
                    else:
                        f1.write('\n')
    print('Done')
