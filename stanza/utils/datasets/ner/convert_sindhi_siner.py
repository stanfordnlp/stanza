"""
Converts the raw data from SiNER to .json for the Stanza NER system

https://aclanthology.org/2020.lrec-1.361.pdf
"""

from stanza.utils.datasets.ner.utils import write_dataset

def fix_sentence(sentence):
    """
    Fix some of the mistags in the dataset

    This covers 11 sentences: 1 P-PERSON, 2 with line breaks in the middle of the tag, and 8 with no B- or I-
    """
    new_sentence = []
    for word_idx, word in enumerate(sentence):
        if word[1] == 'P-PERSON':
            new_sentence.append((word[0], 'B-PERSON'))
        elif word[1] == 'B-OT"':
            new_sentence.append((word[0], 'B-OTHERS'))
        elif word[1] == 'B-T"':
            new_sentence.append((word[0], 'B-TITLE'))
        elif word[1] in ('GPE', 'LOC', 'OTHERS'):
            if len(new_sentence) > 0 and new_sentence[-1][1][:2] in ('B-', 'I-') and new_sentence[-1][1][2:] == word[1]:
                # one example... no idea if it should be a break or
                # not, but the last word translates to "Corporation",
                # so probably not: ميٽرو پوليٽن ڪارپوريشن
                new_sentence.append((word[0], 'I-' + word[1]))
            else:
                new_sentence.append((word[0], 'B-' + word[1]))
        else:
            new_sentence.append(word)
    return new_sentence

def convert_sindhi_siner(in_filename, out_directory, short_name, train_frac=0.8, dev_frac=0.1):
    """
    Read lines from the dataset, crudely separate sentences based on . or !, and write the dataset
    """
    with open(in_filename, encoding="utf-8") as fin:
        lines = fin.readlines()

    lines = [x.strip().split("\t") for x in lines]
    lines = [(x[0].strip(), x[1].strip()) for x in lines if len(x) == 2]
    print("Read %d words from %s" % (len(lines), in_filename))
    sentences = []
    prev_idx = 0
    for sent_idx, line in enumerate(lines):
        # maybe also handle line[0] == '،', "Arabic comma"?
        if line[0] in ('.', '!'):
            sentences.append(lines[prev_idx:sent_idx+1])
            prev_idx=sent_idx+1

    # in case the file doesn't end with punctuation, grab the last few lines
    if prev_idx < len(lines):
        sentences.append(lines[prev_idx:])

    print("Found %d sentences before splitting" % len(sentences))
    sentences = [fix_sentence(x) for x in sentences]
    assert not any('"' in x[1] or x[1].startswith("P-") or x[1] in ("GPE", "LOC", "OTHERS") for sentence in sentences for x in sentence)

    train_len = int(len(sentences) * train_frac)
    dev_len = int(len(sentences) * (train_frac+dev_frac))
    train_sentences = sentences[:train_len]
    dev_sentences = sentences[train_len:dev_len]
    test_sentences = sentences[dev_len:]

    datasets = (train_sentences, dev_sentences, test_sentences)
    write_dataset(datasets, out_directory, short_name, suffix="bio")

