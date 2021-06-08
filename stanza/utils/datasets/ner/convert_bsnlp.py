import argparse
import glob
import os
import logging
import random
import re

import stanza

logger = logging.getLogger('stanza')

AVAILABLE_LANGUAGES = ("bg", "cs", "pl", "ru")

def normalize_bg_entity(text, entity, raw):
    entity = entity.strip()
    # sanity check that the token is in the original text
    if text.find(entity) >= 0:
        return entity

    # some entities have quotes, but the quotes are different from those in the data file
    # for example:
    #   training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_458.txt
    #     'Съвета "Общи въпроси"'
    #   training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1002.txt
    #     'Съвет "Общи въпроси"'
    if sum(1 for x in entity if x == '"') == 2:
        quote_entity = entity.replace('"', '“')
        if text.find(quote_entity) >= 0:
            logger.info("searching for '%s' instead of '%s' in %s" % (quote_entity, entity, raw))
            return quote_entity

        quote_entity = entity.replace('"', '„', 1).replace('"', '“')
        if text.find(quote_entity) >= 0:
            logger.info("searching for '%s' instead of '%s' in %s" % (quote_entity, entity, raw))
            return quote_entity

    if sum(1 for x in entity if x == '"') == 1:
        quote_entity = entity.replace('"', '„', 1)
        if text.find(quote_entity) >= 0:
            logger.info("searching for '%s' instead of '%s' in %s" % (quote_entity, entity, raw))
            return quote_entity

    if entity.find("'") >= 0:
        quote_entity = entity.replace("'", "’")
        if text.find(quote_entity) >= 0:
            logger.info("searching for '%s' instead of '%s' in %s" % (quote_entity, entity, raw))
            return quote_entity

    lower_idx = text.lower().find(entity.lower())
    if lower_idx >= 0:
        fixed_entity = text[lower_idx:lower_idx+len(entity)]
        logger.info("lowercase match found.  Searching for '%s' instead of '%s' in %s" % (fixed_entity, entity, raw))
        return fixed_entity

    substitution_pairs = {
        # this exact error happens in:
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_67.txt
        'Съвет по общи въпроси': 'Съвета по общи въпроси',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_214.txt
        'Сумимото Мицуи файненшъл груп': 'Сумитомо Мицуи файненшъл груп',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_245.txt
        'С и Д': 'С&Д',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_348.txt
        'законопроекта за излизане на Великобритания за излизане от Европейския съюз': 'законопроекта за излизане на Великобритания от Европейския съюз',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_771.txt
        'Унивеситета в Есекс': 'Университета в Есекс',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_779.txt
        'Съвет за сигурност на ООН': 'Съвета за сигурност на ООН',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_787.txt
        'Федерика Могерини': 'Федереика Могерини',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_938.txt
        'Уайстейбъл': 'Уайтстейбъл',
        'Партията за независимост на Обединеното кралство': 'Партията на независимостта на Обединеното кралство',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_972.txt
        'Европейска банка за възстановяване и развитие': 'Европейската банка за възстановяване и развитие',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1065.txt
        'Харолд Уилсон': 'Харолд Уилсън',
        'Манчестърски университет': 'Манчестърския университет',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1096.txt
        'Обединеното кралство в променящата се Европа': 'Обединеното кралство в променяща се Европа',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1175.txt
        'The Daily Express': 'Daily Express',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1186.txt
        'демократичната юнионистка партия': 'демократична юнионистка партия',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1192.txt
        'Европейската агенция за безопасността на полетите': 'Европейската агенция за сигурността на полетите',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1219.txt
        'пресцентъра на Външно министертво': 'пресцентъра на Външно министерство',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1281.txt
        'Европейска агенциа за безопасността на полетите': 'Европейската агенция за сигурността на полетите',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1415.txt
        'Хонк Конг': 'Хонг Конг',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1663.txt
        'Лейбъристка партия': 'Лейбъристката партия',
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1963.txt
        'Найджъл Фараж': 'Найджъл Фарадж',
        'Фараж': 'Фарадж',

        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1773.txt has an entity which is mixed Cyrillic and Ascii
        'Tescо': 'Tesco',
    }

    if entity in substitution_pairs and text.find(substitution_pairs[entity]) >= 0:
        fixed_entity = substitution_pairs[entity]
        logger.info("searching for '%s' instead of '%s' in %s" % (fixed_entity, entity, raw))
        return fixed_entity

    # oops, can't find it anywhere
    # want to raise ValueError but there are just too many in the train set for BG
    logger.error("Could not find '%s' in %s" % (entity, raw))

def fix_bg_typos(text, raw_filename):
    typo_pairs = {
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_202.txt is not exactly a typo, but the word is mixed cyrillic and ascii characters
        'brexit_bg.txt_file_202.txt':  ('Вlооmbеrg', 'Bloomberg'),
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_261.txt has a typo: Telegaph instead of Telegraph
        'brexit_bg.txt_file_261.txt':  ('Telegaph', 'Telegraph'),
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_574.txt has a typo: politicalskrapbook instead of politicalscrapbook
        'brexit_bg.txt_file_574.txt':  ('politicalskrapbook', 'politicalscrapbook'),
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_861.txt has a mix of cyrillic and ascii
        'brexit_bg.txt_file_861.txt':  ('Съвета „Общи въпроси“', 'Съветa "Общи въпроси"'),
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_992.txt is not exactly a typo, but the word is mixed cyrillic and ascii characters
        'brexit_bg.txt_file_992.txt':  ('The Guardiаn', 'The Guardian'),
        # training_pl_cs_ru_bg_rc1/raw/bg/brexit_bg.txt_file_1856.txt has a typo: Southerb instead of Southern
        'brexit_bg.txt_file_1856.txt': ('Southerb', 'Southern'),
    }

    filename = os.path.split(raw_filename)[1]
    if filename in typo_pairs:
        replacement = typo_pairs.get(filename)
        text = text.replace(replacement[0], replacement[1])

    return text

def get_sentences(language, pipeline, annotated, raw):
    if language == 'bg':
        normalize_entity = normalize_bg_entity
        fix_typos = fix_bg_typos
    else:
        raise AssertionError("Please build a normalize_%s_entity and fix_%s_typos first" % language)

    annotated_sentences = []
    with open(raw) as fin:
        lines = fin.readlines()
    if len(lines) < 5:
        raise ValueError("Unexpected format in %s" % raw)
    text = "\n".join(lines[4:])
    text = fix_typos(text, raw)

    entities = {}
    with open(annotated) as fin:
        # first line
        header = fin.readline().strip()
        if len(header.split("\t")) > 1:
            raise ValueError("Unexpected missing header line in %s" % annotated)
        for line in fin:
            pieces = line.strip().split("\t")
            if len(pieces) < 3 or len(pieces) > 4:
                raise ValueError("Unexpected annotation format in %s" % annotated)

            entity = normalize_entity(text, pieces[0], raw)
            if not entity:
                continue
            if entity in entities:
                if entities[entity] != pieces[2]:
                    # would like to make this an error, but it actually happens and it's not clear how to fix
                    # annotated/nord_stream/bg/nord_stream_bg.txt_file_119.out
                    logger.warn("found multiple definitions for %s in %s" % (pieces[0], annotated))
                    entities[entity] = pieces[2]
            else:
                entities[entity] = pieces[2]

    tokenized = pipeline(text)
    # The benefit of doing these one at a time, instead of all at once,
    # is that nested entities won't clobber previously labeled entities.
    # For example, the file
    #   training_pl_cs_ru_bg_rc1/annotated/bg/brexit_bg.txt_file_994.out
    # has each of:
    #   Северна Ирландия
    #   Република Ирландия
    #   Ирландия
    # By doing the larger ones first, we can detect and skip the ones
    # we already labeled when we reach the shorter one
    regexes = [re.compile(re.escape(x)) for x in sorted(entities.keys(), key=len, reverse=True)]

    bad_sentences = set()

    for regex in regexes:
        for match in regex.finditer(text):
            start_char, end_char = match.span()
            # this is inefficient, but for something only run once, it shouldn't matter
            start_token = None
            start_sloppy = False
            end_token = None
            end_sloppy = False
            for token in tokenized.iter_tokens():
                if token.start_char <= start_char and token.end_char > start_char:
                    start_token = token
                    if token.start_char != start_char:
                        start_sloppy = True
                if token.start_char <= end_char and token.end_char >= end_char:
                    end_token = token
                    if token.end_char != end_char:
                        end_sloppy = True
                    break
            if start_token is None or end_token is None:
                raise RuntimeError("Match %s did not align with any tokens in %s" % (match.group(0), raw))
            if not start_token.sent is end_token.sent:
                bad_sentences.add(start_token.sent.id)
                bad_sentences.add(end_token.sent.id)
                logger.warn("match %s spanned sentences %d and %d in document %s" % (match.group(0), start_token.sent.id, end_token.sent.id, raw))
                continue

            # ids start at 1, not 0, so we have to subtract 1
            # then the end token is included, so we add back the 1
            # TODO: verify that this is correct if the language has MWE - cs, pl, for example
            tokens = start_token.sent.tokens[start_token.id[0]-1:end_token.id[0]]
            if all(token.ner for token in tokens):
                # skip matches which have already been made
                # this has the nice side effect of not complaining if
                # a smaller match is found after a larger match
                # earlier set the NER on those tokens
                continue

            if start_sloppy and end_sloppy:
                bad_sentences.add(start_token.sent.id)
                logger.warn("match %s matched in the middle of a token in %s" % (match.group(0), raw))
                continue
            if start_sloppy:
                bad_sentences.add(end_token.sent.id)
                logger.warn("match %s started matching in the middle of a token in %s" % (match.group(0), raw))
                #print(start_token)
                #print(end_token)
                #print(start_char, end_char)
                continue
            if end_sloppy:
                bad_sentences.add(start_token.sent.id)
                logger.warn("match %s ended matching in the middle of a token in %s" % (match.group(0), raw))
                #print(start_token)
                #print(end_token)
                #print(start_char, end_char)
                continue
            match_text = match.group(0)
            if match_text not in entities:
                raise RuntimeError("Matched %s, which is not in the entities from %s" % (match_text, annotated))
            ner_tag = entities[match_text]
            tokens[0].ner = "B-" + ner_tag
            for token in tokens[1:]:
                token.ner = "I-" + ner_tag

    for sentence in tokenized.sentences:
        if not sentence.id in bad_sentences:
            annotated_sentences.append(sentence)

    return annotated_sentences

def write_sentences(output_filename, annotated_sentences):
    logger.info("Writing %d sentences to %s" % (len(annotated_sentences), output_filename))
    with open(output_filename, "w") as fout:
        for sentence in annotated_sentences:
            for token in sentence.tokens:
                ner_tag = token.ner
                if not ner_tag:
                    ner_tag = "O"
                fout.write("%s\t%s\n" % (token.text, ner_tag))
            fout.write("\n")


def convert_bsnlp(language, base_input_path, output_filename, split_filename=None):
    """
    Converts the BSNLP dataset for the given language.

    If only one output_filename is provided, all of the output goes to that file.
    If split_filename is provided as well, 15% of the output chosen randomly
      goes there instead.  The dataset has no dev set, so this helps
      divide the data into train/dev/test.
    Note that the custom error fixes are only done for BG currently.
    Please manually correct the data as appropriate before using this
      for another language.
    """
    if language not in AVAILABLE_LANGUAGES:
        raise ValueError("The current BSNLP datasets only include the following languages: %s" % ",".join(AVAILABLE_LANGUAGES))
    if language != "bg":
        raise ValueError("There were quite a few data fixes needed to get the data correct for BG.  Please work on similar fixes before using the model for %s" % language.upper())
    pipeline = stanza.Pipeline(language, processors="tokenize")
    random.seed(1234)

    annotated_path = os.path.join(base_input_path, "annotated", "*", language, "*")
    annotated_files = sorted(glob.glob(annotated_path))
    raw_path = os.path.join(base_input_path, "raw", "*", language, "*")
    raw_files = sorted(glob.glob(raw_path))

    # if the instructions for downloading the data from the
    # process_ner_dataset script are followed, there will be two test
    # directories of data and a separate training directory of data.
    if len(annotated_files) == 0 and len(raw_files) == 0:
        logger.info("Could not find files in %s" % annotated_path)
        annotated_path = os.path.join(base_input_path, "annotated", language, "*")
        logger.info("Trying %s instead" % annotated_path)
        annotated_files = sorted(glob.glob(annotated_path))
        raw_path = os.path.join(base_input_path, "raw", language, "*")
        raw_files = sorted(glob.glob(raw_path))

    if len(annotated_files) != len(raw_files):
        raise ValueError("Unexpected differences in the file lists between %s and %s" % (annotated_files, raw_files))

    for i, j in zip(annotated_files, raw_files):
        if os.path.split(i)[1][:-4] != os.path.split(j)[1][:-4]:
            raise ValueError("Unexpected differences in the file lists: found %s instead of %s" % (i, j))

    annotated_sentences = []
    if split_filename:
        split_sentences = []
    for annotated, raw in zip(annotated_files, raw_files):
        new_sentences = get_sentences(language, pipeline, annotated, raw)
        if not split_filename or random.random() < 0.85:
            annotated_sentences.extend(new_sentences)
        else:
            split_sentences.extend(new_sentences)

    write_sentences(output_filename, annotated_sentences)
    if split_filename:
        write_sentences(split_filename, split_sentences)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default="bg", help="Language to process")
    parser.add_argument('--input_path', type=str, default="/home/john/extern_data/ner/bsnlp2019", help="Where to find the files")
    parser.add_argument('--output_path', type=str, default="/home/john/stanza/data/ner/bg_bsnlp.test.csv", help="Where to output the results")
    parser.add_argument('--dev_path', type=str, default=None, help="A secondary output path - 15% of the data will go here")
    args = parser.parse_args()

    convert_bsnlp(args.language, args.input_path, args.output_path, args.dev_path)
