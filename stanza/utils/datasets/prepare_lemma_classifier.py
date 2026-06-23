from collections import namedtuple
import os
import sys

from stanza.utils.datasets.common import find_treebank_dataset_file, UnknownDatasetError
from stanza.utils.default_paths import get_default_paths
from stanza.models.lemma_classifier import prepare_dataset
from stanza.models.common.short_name_to_treebank import short_name_to_treebank
from stanza.utils.conll import CoNLL

SECTIONS = ("train", "dev", "test")

Target = namedtuple("Target", "word upos filename lemmas")

def process_treebank(paths, short_name, word, upos, allowed_lemmas, sections=SECTIONS, dataset_to_use=None):
    if dataset_to_use is None:
        treebank = short_name_to_treebank(short_name)
    else:
        treebank = short_name_to_treebank(dataset_to_use)
    udbase_dir = paths["UDBASE"]

    output_dir = paths["LEMMA_CLASSIFIER_DATA_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    output_filenames = []

    for section in sections:
        filename = find_treebank_dataset_file(treebank, udbase_dir, section, "conllu", fail=True)
        output_filename = os.path.join(output_dir, "%s.%s.lemma" % (short_name, section))
        args = ["--conll_path", filename,
                "--target_word", word,
                "--target_upos", upos,
                "--output_path", output_filename]
        if allowed_lemmas is not None:
            args.extend(["--allowed_lemmas", allowed_lemmas])
        prepare_dataset.main(args)
        output_filenames.append(output_filename)

    return output_filenames

def convert_combined_docs(docs, target_word, target_upos, allowed_lemmas=".*"):
    sentences = [ [], [], [] ]

    for section_docs, section_sentences in zip(docs, sentences):
        for filename, doc in section_docs:
            processor = prepare_dataset.DataProcessor(target_word=target_word, target_upos=target_upos, allowed_lemmas=allowed_lemmas)
            new_sentences = processor.process_document(doc, save_name=None)
            print("Extracted %d sentences from %s" % (len(new_sentences), filename))
            section_sentences.extend(new_sentences)

    return sentences

def process_en_combined(paths, short_name):
    udbase_dir = paths["UDBASE"]
    output_dir = paths["LEMMA_CLASSIFIER_DATA_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    train_treebanks = ["UD_English-EWT", "UD_English-GUM", "UD_English-GUMReddit", "UD_English-LinES"]
    test_treebanks = ["UD_English-PUD", "UD_English-Pronouns"]

    docs = [ [], [], [] ]
    for treebank in train_treebanks:
        for section_idx, section in enumerate(SECTIONS):
            filename = find_treebank_dataset_file(treebank, udbase_dir, section, "conllu", fail=True)
            doc = CoNLL.conll2doc(filename)
            docs[section_idx].append((filename, doc))
            print("Loaded %s" % filename)
    for treebank in test_treebanks:
        section = "test"
        filename = find_treebank_dataset_file(treebank, udbase_dir, section, "conllu", fail=True)
        doc = CoNLL.conll2doc(filename)
        # only test set for these documents
        docs[2].append((filename, doc))
        print("Loaded %s" % filename)

    for target_word, target_upos, target_filename, _ in DATASET_TARGETS["en_combined"]:
        print("Processing %s_%s" % (target_word, target_upos))
        sentences = convert_combined_docs(docs, target_word, target_upos)

        for section, section_sentences in zip(SECTIONS, sentences):
            output_filename = os.path.join(output_dir, "%s.%s.%s.lemma" % (short_name, target_filename, section))
            prepare_dataset.DataProcessor.write_output_file(output_filename, target_upos, section_sentences)
            print("Wrote %s sentences to %s" % (len(section_sentences), output_filename))

def process_ja_gsd(paths, short_name):
    # this one looked promising, but only has 10 total dev & test cases
    # 行っ VERB Counter({'行う': 60, '行く': 38})
    # could possibly do
    # ない AUX Counter({'ない': 383, '無い': 99})
    # なく AUX Counter({'無い': 53, 'ない': 42})
    # currently this one has enough in the dev & test data
    # and functions well
    # だ AUX Counter({'だ': 237, 'た': 67})
    word = "だ"
    upos = "AUX"
    allowed_lemmas = None

    # both the base GSD and 'ja_combined' with extra training sentences will use the ja_gsd dataset
    process_treebank(paths, short_name, word, upos, allowed_lemmas, dataset_to_use='ja_gsd')

def process_fa_perdt(paths, short_name):
    word = "شد"
    upos = "VERB"
    allowed_lemmas = "کرد|شد"

    process_treebank(paths, short_name, word, upos, allowed_lemmas)

def process_hi_hdtb(paths, short_name):
    word = "के"
    upos = "ADP"
    allowed_lemmas = "का|के"

    process_treebank(paths, short_name, word, upos, allowed_lemmas)

def process_ar_padt(paths, short_name):
    word = "أن"
    upos = "SCONJ"
    allowed_lemmas = "أَن|أَنَّ"

    process_treebank(paths, short_name, word, upos, allowed_lemmas)

def process_el_gdt(paths, short_name):
    """
    All of the Greek lemmas for these words are εγώ or μου

    τους PRON Counter({'μου': 118, 'εγώ': 32})
    μας PRON Counter({'μου': 89, 'εγώ': 32})
    του PRON Counter({'μου': 82, 'εγώ': 8})
    της PRON Counter({'μου': 80, 'εγώ': 2})
    σας PRON Counter({'μου': 34, 'εγώ': 24})
    μου PRON Counter({'μου': 45, 'εγώ': 10})
    """
    word = "τους|μας|του|της|σας|μου"
    upos = "PRON"
    allowed_lemmas = None

    process_treebank(paths, short_name, word, upos, allowed_lemmas)

def process_sl(paths, short_name):
    # TODO: could refactor some with the en equivalent
    udbase_dir = paths["UDBASE"]
    output_dir = paths["LEMMA_CLASSIFIER_DATA_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    if short_name == 'sl_ssj':
        train_treebanks = ["UD_Slovenian-SSJ"]
        extra_files = []
    elif short_name == 'sl_sst':
        train_treebanks = ["UD_Slovenian-SST"]
        extra_files = []
    elif short_name == 'sl_combined':
        train_treebanks = ["UD_Slovenian-SSJ", "UD_Slovenian-SST"]

        extra_path = os.path.join(paths["STANZA_EXTERN_DIR"], "slovenian", "SUK.CoNLL-U")
        if not os.path.exists(extra_path):
            raise FileNotFoundError("Cannot find SUK extra data.  Please download from https://www.clarin.si/repository/xmlui/handle/11356/1959 and put it in %s" % extra_path)
        extra_files = []
        for filename in ["ssj500k-tag.ud.conllu", "ambiga.ud.conllu"]:
            extra_files.append(os.path.join(extra_path, filename))
            if not os.path.exists(extra_files[-1]):
                raise FileNotFoundError("Could not find expected extra path %s" % extra_files[-1])

    docs = [ [], [], [] ]
    for treebank in train_treebanks:
        for section_idx, section in enumerate(SECTIONS):
            filename = find_treebank_dataset_file(treebank, udbase_dir, section, "conllu", fail=True)
            doc = CoNLL.conll2doc(filename)
            docs[section_idx].append((filename, doc))
            print("Loaded %s" % filename)
    for filename in extra_files:
        doc = CoNLL.conll2doc(filename)
        docs[0].append((filename, doc))
        print("Loaded %s" % filename)

    for target_word, target_upos, target_filename, allowed_lemmas in DATASET_TARGETS[short_name]:
        print("Processing %s_%s" % (target_word, target_upos))
        sentences = convert_combined_docs(docs, target_word, target_upos, allowed_lemmas)

        for section, section_sentences in zip(SECTIONS, sentences):
            output_filename = os.path.join(output_dir, "%s.%s.%s.lemma" % (short_name, target_filename, section))
            prepare_dataset.DataProcessor.write_output_file(output_filename, target_upos, section_sentences)
            print("Wrote %s sentences to %s" % (len(section_sentences), output_filename))


DATASET_MAPPING = {
    "ar_padt":           process_ar_padt,
    "el_gdt":            process_el_gdt,
    "en_combined":       process_en_combined,
    "fa_perdt":          process_fa_perdt,
    "hi_hdtb":           process_hi_hdtb,
    "ja_combined":       process_ja_gsd,
    "ja_gsd":            process_ja_gsd,

    "sl_combined":       process_sl,
    "sl_ssj":            process_sl,
    "sl_sst":            process_sl,
}

DATASET_TARGETS = {
    "en_combined":       [
        Target("'s",  "AUX",  "s",    None),   # since we don't want filenames with 's
        Target("her", "PRON", "her",  None),
    ],
    "sl_combined":       [
        Target("(?i:rok|roka|roke|roki)",         "NOUN",  "rok",   "rok|roka"),
        Target("(?i:dela|delu|delom|delih|deli)", "NOUN",  "del",   "del|delo"),
    ]
}

DATASET_TARGETS["sl_ssj"] = DATASET_TARGETS["sl_combined"]
DATASET_TARGETS["sl_sst"] = DATASET_TARGETS["sl_combined"]

def main(dataset_name):
    paths = get_default_paths()
    print("Processing lemma_classifier %s" % dataset_name)

    # obviously will want to multiplex to multiple languages / datasets
    if dataset_name in DATASET_MAPPING:
        DATASET_MAPPING[dataset_name](paths, dataset_name)
    else:
        raise UnknownDatasetError(dataset_name, f"dataset {dataset_name} currently not handled by prepare_lemma_classifier.py")
    print("Done processing lemma_classifier %s" % dataset_name)

if __name__ == '__main__':
    main(sys.argv[1])
