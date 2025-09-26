from collections import defaultdict
import json
import os
import re
import glob

from stanza.utils.default_paths import get_default_paths
from stanza.utils.get_tqdm import get_tqdm
from stanza.utils.datasets.coref.utils import find_cconj_head

from stanza.utils.conll import CoNLL

from random import Random

import argparse

augment_random = Random(7)
split_random = Random(8)

tqdm = get_tqdm()
IS_UDCOREF_FORMAT = True
UDCOREF_ADDN = 0 if not IS_UDCOREF_FORMAT else 1

def process_documents(docs, augment=False):
    processed_section = []

    for idx, (doc, doc_id, lang) in enumerate(tqdm(docs)):
        # drop the last token 10% of the time
        if augment:
            for i in doc.sentences:
                if len(i.words) > 1:
                    if augment_random.random() < 0.1:
                        i.tokens = i.tokens[:-1]
                        i.words = i.words[:-1]

        # extract the entities
        # get sentence words and lengths
        sentences = [[j.text for j in i.words]
                    for i in doc.sentences]
        sentence_lens = [len(x.words) for x in doc.sentences]

        cased_words = [] 
        for x in sentences:
            if augment:
                # modify case of the first word with 50% chance
                if augment_random.random() < 0.5:
                    x[0] = x[0].lower()

            for y in x:
                cased_words.append(y)

        sent_id = [y for idx, sent_len in enumerate(sentence_lens) for y in [idx] * sent_len]

        word_total = 0
        heads = []
        # TODO: does SD vs UD matter?
        deprel = []
        for sentence in doc.sentences:
            for word in sentence.words:
                deprel.append(word.deprel)
                if word.head == 0:
                    heads.append("null")
                else:
                    heads.append(word.head - 1 + word_total)
            word_total += len(sentence.words)

        span_clusters = defaultdict(list)
        word_clusters = defaultdict(list)
        head2span = []
        word_total = 0
        SPANS = re.compile(r"(\(\w+|[%\w]+\))")
        for parsed_sentence in doc.sentences:
            # spans regex
            # parse the misc column, leaving on "Entity" entries
            misc = [[k.split("=")
                    for k in j
                    if k.split("=")[0] == "Entity"]
                    for i in parsed_sentence.words
                    for j in [i.misc.split("|") if i.misc else []]]
            # and extract the Entity entry values
            entities = [i[0][1] if len(i) > 0 else None for i in misc]
            # extract reference information
            refs = [SPANS.findall(i) if i else [] for i in entities]
            # and calculate spans: the basic rule is (e... begins a reference
            # and ) without e before ends the most recent reference
            # every single time we get a closing element, we pop it off
            # the refdict and insert the pair to final_refs
            refdict = defaultdict(list)
            final_refs = defaultdict(list)
            last_ref = None
            for indx, i in enumerate(refs):
                for j in i:
                    # this is the beginning of a reference
                    if j[0] == "(":
                        refdict[j[1+UDCOREF_ADDN:]].append(indx)
                        last_ref = j[1+UDCOREF_ADDN:]
                    # at the end of a reference, if we got exxxxx, that ends
                    # a particular refereenc; otherwise, it ends the last reference
                    elif j[-1] == ")" and j[UDCOREF_ADDN:-1].isnumeric():
                        if (not UDCOREF_ADDN) or j[0] == "e":
                            try:
                                final_refs[j[UDCOREF_ADDN:-1]].append((refdict[j[UDCOREF_ADDN:-1]].pop(-1), indx))
                            except IndexError:
                                # this is probably zero anaphora
                                continue
                    elif j[-1] == ")":
                        final_refs[last_ref].append((refdict[last_ref].pop(-1), indx))
                        last_ref = None
            final_refs = dict(final_refs)
            # convert it to the right format (specifically, in (ref, start, end) tuples)
            coref_spans = []
            for k, v in final_refs.items():
                for i in v:
                    coref_spans.append([int(k), i[0], i[1]])
            sentence_upos = [x.upos for x in parsed_sentence.words]
            sentence_heads = [x.head - 1 if x.head > 0 else None for x in parsed_sentence.words]
            for span in coref_spans:
                # input is expected to be start word, end word + 1
                # counting from 0
                # whereas the OntoNotes coref_span is [start_word, end_word] inclusive
                span_start = span[1] + word_total
                span_end = span[2] + word_total + 1
                candidate_head = find_cconj_head(sentence_heads, sentence_upos, span[1], span[2]+1)
                if candidate_head is None:
                    for candidate_head in range(span[1], span[2] + 1):
                        # stanza uses 0 to mark the head, whereas OntoNotes is counting
                        # words from 0, so we have to subtract 1 from the stanza heads
                        #print(span, candidate_head, parsed_sentence.words[candidate_head].head - 1)
                        # treat the head of the phrase as the first word that has a head outside the phrase
                        if (parsed_sentence.words[candidate_head].head - 1 < span[1] or
                            parsed_sentence.words[candidate_head].head - 1 > span[2]):
                            break
                    else:
                        # if none have a head outside the phrase (circular??)
                        # then just take the first word
                        candidate_head = span[1]
                #print("----> %d" % candidate_head)
                candidate_head += word_total
                span_clusters[span[0]].append((span_start, span_end))
                word_clusters[span[0]].append(candidate_head)
                head2span.append((candidate_head, span_start, span_end))
            word_total += len(parsed_sentence.words)
        span_clusters = sorted([sorted(values) for _, values in span_clusters.items()])
        word_clusters = sorted([sorted(values) for _, values in word_clusters.items()])
        head2span = sorted(head2span)

        processed = {
            "document_id": doc_id,
            "cased_words": cased_words,
            "sent_id": sent_id,
            "part_id": idx,
            # "pos": pos,
            "deprel": deprel,
            "head": heads,
            "span_clusters": span_clusters,
            "word_clusters": word_clusters,
            "head2span": head2span,
            "lang": lang
        }
        processed_section.append(processed)
    return processed_section

def process_dataset(short_name, coref_output_path, split_test, train_files, dev_files):
    section_names = ('train', 'dev')
    section_filenames = [train_files, dev_files]
    sections = []

    test_sections = []

    for section, filenames in zip(section_names, section_filenames):
        input_file = []
        for load in filenames:
            lang = load.split("/")[-1].split("_")[0]
            print("Ingesting %s from %s of lang %s" % (section, load, lang))
            docs = CoNLL.conll2multi_docs(load)
            print("  Ingested %d documents" % len(docs))
            if split_test and section == 'train':
                test_section = []
                train_section = []
                for i in docs:
                    # reseed for each doc so that we can attempt to keep things stable in the event
                    # of different file orderings or some change to the number of documents
                    split_random = Random(i.sentences[0].doc_id + i.sentences[0].text)
                    if split_random.random() < split_test:
                        test_section.append((i, i.sentences[0].doc_id, lang))
                    else:
                        train_section.append((i, i.sentences[0].doc_id, lang))
                if len(test_section) == 0 and len(train_section) >= 2:
                    idx = split_random.randint(0, len(train_section) - 1)
                    test_section = [train_section[idx]]
                    train_section = train_section[:idx] + train_section[idx+1:]
                print("  Splitting %d documents from %s for test" % (len(test_section), load))
                input_file.extend(train_section)
                test_sections.append(test_section)
            else:
                for i in docs:
                    input_file.append((i, i.sentences[0].doc_id, lang))
        print("Ingested %d total documents" % len(input_file))
        sections.append(input_file)

    if split_test:
        section_names = ('train', 'dev', 'test')
        full_test_section = []
        for filename, test_section in zip(filenames, test_sections):
            # TODO: could write dataset-specific test sections as well
            full_test_section.extend(test_section)
        sections.append(full_test_section)


    for section_data, section_name in zip(sections, section_names):
        converted_section = process_documents(section_data, augment=(section_name=="train"))

        os.makedirs(coref_output_path, exist_ok=True)
        output_filename = os.path.join(coref_output_path, "%s.%s.json" % (short_name, section_name))
        with open(output_filename, "w", encoding="utf-8") as fout:
            json.dump(converted_section, fout, indent=2)

def get_dataset_by_language(coref_input_path, langs):
    conll_path = os.path.join(coref_input_path, "CorefUD-1.2-public", "data")
    train_filenames = []
    dev_filenames = []
    for lang in langs:
        train_filenames.extend(glob.glob(os.path.join(conll_path, "*%s*" % lang, "*train.conllu")))
        dev_filenames.extend(glob.glob(os.path.join(conll_path, "*%s*" % lang, "*dev.conllu")))
    train_filenames = sorted(train_filenames)
    dev_filenames = sorted(dev_filenames)
    return train_filenames, dev_filenames

def main():
    paths = get_default_paths()
    parser = argparse.ArgumentParser(
        prog='Convert UDCoref Data',
    )
    parser.add_argument('--split_test', default=None, type=float, help='How much of the data to randomly split from train to make a test set')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--directory', type=str, help="the name of the subfolder for data conversion")
    group.add_argument('--project', type=str, help="Look for and use a set of datasets for data conversion - Slavic or Hungarian")

    args = parser.parse_args()
    coref_input_path = paths['COREF_BASE']
    coref_output_path = paths['COREF_DATA_DIR']

    if args.project:
        if args.project == 'slavic':
            project = "slavic_udcoref"
            langs = ('Polish', 'Russian', 'Czech')
            train_filenames, dev_filenames = get_dataset_by_language(coref_input_path, langs)
        elif args.project == 'hungarian':
            project = "hu_udcoref"
            langs = ('Hungarian',)
            train_filenames, dev_filenames = get_dataset_by_language(coref_input_path, langs)
        elif args.project == 'gerrom':
            project = "gerrom_udcoref"
            langs = ('Catalan', 'English', 'French', 'German', 'Norwegian', 'Spanish')
            train_filenames, dev_filenames = get_dataset_by_language(coref_input_path, langs)
        elif args.project == 'germanic':
            project = "germanic_udcoref"
            langs = ('English', 'German', 'Norwegian')
            train_filenames, dev_filenames = get_dataset_by_language(coref_input_path, langs)
        elif args.project == 'norwegian':
            project = "norwegian_udcoref"
            langs = ('Norwegian',)
            train_filenames, dev_filenames = get_dataset_by_language(coref_input_path, langs)
    else:
        project = args.directory
        conll_path = os.path.join(coref_input_path, project)
        if not os.path.exists(conll_path) and os.path.exists(project):
            conll_path = args.directory
        train_filenames = sorted(glob.glob(os.path.join(conll_path, f"*train.conllu")))
        dev_filenames = sorted(glob.glob(os.path.join(conll_path, f"*dev.conllu")))
    process_dataset(project, coref_output_path, args.split_test, train_filenames, dev_filenames)

if __name__ == '__main__':
    main()

