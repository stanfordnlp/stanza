from collections import Counter
from copy import copy
import json
import numpy as np
import re
import logging
import os

import stanza.utils.default_paths as default_paths
from stanza.models.common.utils import ud_scores, harmonic_mean
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import *

logger = logging.getLogger('stanza')
paths = default_paths.get_default_paths()

def create_dictionary(lexicon=None):
    """
    This function is to create a new dictionary used for improving tokenization model for multi-syllable words languages
    such as vi, zh or th. This function takes the lexicon as input and output a dictionary that contains three set:
    words, prefixes and suffixes where prefixes set should contains all the prefixes in the lexicon and similar for suffixes.
    The point of having prefixes/suffixes sets in the  dictionary is just to make it easier to check during data preparation.

    :param shorthand - language and dataset, eg: vi_vlsp, zh_gsdsimp
    :param lexicon - set of words used to create dictionary
    :return a dictionary object that contains words and their prefixes and suffixes.
    """
    
    dictionary = {"words":set(), "prefixes":set(), "suffixes":set()}
    
    def add_word(word):
        if word not in dictionary["words"]:
            dictionary["words"].add(word)
            prefix = ""
            suffix = ""
            for i in range(0,len(word)-1):
                prefix = prefix + word[i]
                suffix = word[len(word) - i - 1] + suffix
                dictionary["prefixes"].add(prefix)
                dictionary["suffixes"].add(suffix)

    for word in lexicon:
        if len(word)>1:
            add_word(word)

    return dictionary
def create_lexicon(shorthand=None, train_path=None, external_path=None):
    """
    This function is to create a lexicon to store all the words from the training set and external dictionary.
    This lexicon will be saved with the model and will be used to create dictionary when the model is loaded.
    The idea of separating lexicon and dictionary in two different phases is a good tradeoff between time and space.
    Note that we eliminate all the long words but less frequently appeared in the lexicon by only taking 95-percentile
    list of words.

    :param shorthand - language and dataset, eg: vi_vlsp, zh_gsdsimp
    :param train_path - path to conllu train file
    :param external_path - path to extenral dict, expected to be inside the training dataset dir with format of: SHORTHAND-externaldict.txt
    :return a set lexicon object that contains all distinct words
    """
    lexicon = set()
    length_freq = []
    #this regex is to check if a character is an actual Thai character as seems .isalpha() python method doesn't pick up Thai accent characters..
    pattern_thai = re.compile(r"(?:[^\d\W]+)|\s")
    
    def check_valid_word(shorthand, word):
        """
        This function is to check if the word are multi-syllable words and not numbers. 
        For vi, whitespaces are syllabe-separator.
        """
        if shorthand.startswith("vi_"):
            return True if len(word.split(" ")) > 1 and any(map(str.isalpha, word)) and not any(map(str.isdigit, word)) else False
        elif shorthand.startswith("th_"):
            return True if len(word) > 1 and any(map(pattern_thai.match, word)) and not any(map(str.isdigit, word)) else False
        else:
            return True if len(word) > 1 and any(map(str.isalpha, word)) and not any(map(str.isdigit, word)) else False

    #checking for words in the training set to add them to lexicon.
    if train_path is not None:
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"Cannot open train set at {train_path}")

        doc_conll,_ = CoNLL.conll2dict(input_file=train_path)

        for sent_conll in doc_conll:
            for token_conll in sent_conll:
                word = token_conll['text'].lower()
                if check_valid_word(shorthand, word) and word not in lexicon:
                    lexicon.add(word)
                    length_freq.append(len(word))
        count_word = len(lexicon)
        logger.info(f"Added {count_word} words from the training data to the lexicon.")

    #checking for external dictionary and add them to lexicon.
    if external_path is not None:
        if not os.path.isfile(external_path):
            raise FileNotFoundError(f"Cannot open external dictionary at {external_path}")

        external_file = open(external_path, "r", encoding="utf-8")
        lines = external_file.readlines()
        for line in lines:
            word = line.lower()
            word = word.replace("\n","")
            if check_valid_word(shorthand, word) and word not in lexicon:
                lexicon.add(word)
                length_freq.append(len(word))
        external_file.close()
        logger.info(f"Added another {len(lexicon) - count_word} words from the external dict to dictionary.")
        

    #automatically calculate the number of dictionary features (window size to look for words) based on the frequency of word length
    #take the length at 95-percentile to eliminate all the longest (maybe) compounds words in the lexicon
    num_dict_feat = int(np.percentile(length_freq, 95))
    lexicon = {word for word in lexicon if len(word) <= num_dict_feat }
    logger.info(f"Final lexicon consists of {len(lexicon)} words after getting rid of long words.")

    return lexicon, num_dict_feat

def load_lexicon(args):
    """
    This function is to create a new dictionary and load it to training.
    The external dictionary is expected to be inside the training dataset dir with format of: SHORTHAND-externaldict.txt
    For example, vi_vlsp-externaldict.txt
    """
    shorthand = args["shorthand"]
    tokenize_dir = paths["TOKENIZE_DATA_DIR"]
    train_path = f"{tokenize_dir}/{shorthand}.train.gold.conllu"
    external_dict_path = f"{tokenize_dir}/{shorthand}-externaldict.txt"
    if not os.path.exists(external_dict_path):
        logger.info("External dictionary not found! Checking training data...")
        external_dict_path = None
    if not os.path.exists(train_path):
        logger.info(f"Training dataset does not exist, thus cannot create dictionary {shorthand}")
        train_path = None
    if train_path is None and external_dict_path is None:
        raise FileNotFoundError(f"Cannot find training set / external dictionary at {train_path} and {external_dict_path}")

    return create_lexicon(shorthand, train_path, external_dict_path)


def load_mwt_dict(filename):
    if filename is not None:
        with open(filename, 'r') as f:
            mwt_dict0 = json.load(f)

        mwt_dict = dict()
        for item in mwt_dict0:
            (key, expansion), count = item

            if key not in mwt_dict or mwt_dict[key][1] < count:
                mwt_dict[key] = (expansion, count)

        return mwt_dict
    else:
        return

def process_sentence(sentence, mwt_dict=None):
    sent = []
    i = 0
    for tok, p, position_info in sentence:
        expansion = None
        if (p == 3 or p == 4) and mwt_dict is not None:
            # MWT found, (attempt to) expand it!
            if tok in mwt_dict:
                expansion = mwt_dict[tok][0]
            elif tok.lower() in mwt_dict:
                expansion = mwt_dict[tok.lower()][0]
        if expansion is not None:
            sent.append({ID: (i+1, i+len(expansion)), TEXT: tok})
            if position_info is not None:
                sent[-1][START_CHAR] = position_info[0]
                sent[-1][END_CHAR] = position_info[1]
            for etok in expansion:
                sent.append({ID: (i+1, ), TEXT: etok})
                i += 1
        else:
            if len(tok) <= 0:
                continue
            sent.append({ID: (i+1, ), TEXT: tok})
            if position_info is not None:
                sent[-1][START_CHAR] = position_info[0]
                sent[-1][END_CHAR] = position_info[1]
            if p == 3 or p == 4:# MARK
                sent[-1][MISC] = 'MWT=Yes'
            i += 1
    return sent


# https://stackoverflow.com/questions/201323/how-to-validate-an-email-address-using-a-regular-expression
EMAIL_RAW_RE = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""

# https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
# modification: disallow " as opposed to all ^\s
URL_RAW_RE = r"""(?:https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s"]{2,}|www\.[a-zA-Z0-9]+\.[^\s"]{2,})"""

MASK_RE = re.compile(f"(?:{EMAIL_RAW_RE}|{URL_RAW_RE})")

def find_spans(raw):
    """
    Return spans of text which don't contain <PAD> and are split by <PAD>
    """
    pads = [idx for idx, char in enumerate(raw) if char == '<PAD>']
    if len(pads) == 0:
        spans = [(0, len(raw))]
    else:
        prev = 0
        spans = []
        for pad in pads:
            if pad != prev:
                spans.append( (prev, pad) )
            prev = pad + 1
        if prev < len(raw):
            spans.append( (prev, len(raw)) )
    return spans

def update_pred_regex(raw, pred):
    """
    Update the results of a tokenization batch by checking the raw text against a couple regular expressions

    Currently, emails and urls are handled
    TODO: this might work better as a constraint on the inference

    for efficiency pred is modified in place
    """
    spans = find_spans(raw)

    for span_begin, span_end in spans:
        text = "".join(raw[span_begin:span_end])
        for match in MASK_RE.finditer(text):
            match_begin, match_end = match.span()
            # first, update all characters touched by the regex to not split
            # with the exception of the last character...
            for char in range(match_begin+span_begin, match_end+span_begin-1):
                pred[char] = 0
            # if the last character is not currently a split, make it a word split
            if pred[match_end+span_begin-1] == 0:
                pred[match_end+span_begin-1] = 1

    return pred

SPACE_RE = re.compile(r'\s')
SPACE_SPLIT_RE = re.compile(r'( *[^ ]+)')

def output_predictions(output_file, trainer, data_generator, vocab, mwt_dict, max_seqlen=1000, orig_text=None, no_ssplit=False, use_regex_tokens=True):
    paragraphs = []
    for i, p in enumerate(data_generator.sentences):
        start = 0 if i == 0 else paragraphs[-1][2]
        length = sum([len(x[0]) for x in p])
        paragraphs += [(i, start, start+length, length)] # para idx, start idx, end idx, length

    paragraphs = list(sorted(paragraphs, key=lambda x: x[3], reverse=True))

    all_preds = [None] * len(paragraphs)
    all_raw = [None] * len(paragraphs)

    eval_limit = max(3000, max_seqlen)

    batch_size = trainer.args['batch_size']
    skip_newline = trainer.args['skip_newline']
    batches = int((len(paragraphs) + batch_size - 1) / batch_size)

    for i in range(batches):
        # At evaluation time, each paragraph is treated as a single "sentence", and a batch of `batch_size` paragraphs 
        # are tokenized together. `offsets` here are used by the data generator to identify which paragraphs to use
        # for the next batch of evaluation.
        batchparas = paragraphs[i * batch_size : (i + 1) * batch_size]
        offsets = [x[1] for x in batchparas]

        batch = data_generator.next(eval_offsets=offsets)
        raw = batch[3]

        N = len(batch[3][0])
        if N <= eval_limit:
            pred = np.argmax(trainer.predict(batch), axis=2)
        else:
            idx = [0] * len(batchparas)
            adv = [0] * len(batchparas)
            Ns = [p[3] for p in batchparas]
            pred = [[] for _ in batchparas]
            while True:
                ens = [min(N - idx1, eval_limit) for idx1, N in zip(idx, Ns)]
                en = max(ens)
                batch1 = batch[0][:, :en], batch[1][:, :en], batch[2][:, :en], [x[:en] for x in batch[3]]
                pred1 = np.argmax(trainer.predict(batch1), axis=2)

                for j in range(len(batchparas)):
                    sentbreaks = np.where((pred1[j] == 2) + (pred1[j] == 4))[0]
                    if len(sentbreaks) <= 0 or idx[j] >= Ns[j] - eval_limit:
                        advance = ens[j]
                    else:
                        advance = np.max(sentbreaks) + 1

                    pred[j] += [pred1[j, :advance]]
                    idx[j] += advance
                    adv[j] = advance

                if all([idx1 >= N for idx1, N in zip(idx, Ns)]):
                    break
                # once we've made predictions on a certain number of characters for each paragraph (recorded in `adv`),
                # we skip the first `adv` characters to make the updated batch
                batch = data_generator.next(eval_offsets=adv, old_batch=batch)

            pred = [np.concatenate(p, 0) for p in pred]

        for j, p in enumerate(batchparas):
            len1 = len([1 for x in raw[j] if x != '<PAD>'])
            if pred[j][len1-1] < 2:
                pred[j][len1-1] = 2
            elif pred[j][len1-1] > 2:
                pred[j][len1-1] = 4
            if use_regex_tokens:
                all_preds[p[0]] = update_pred_regex(raw[j], pred[j][:len1])
            else:
                all_preds[p[0]] = pred[j][:len1]
            all_raw[p[0]] = raw[j]

    offset = 0
    oov_count = 0
    doc = []

    text = SPACE_RE.sub(' ', orig_text) if orig_text is not None else None
    char_offset = 0
    use_la_ittb_shorthand = trainer.args['shorthand'] == 'la_ittb'

    UNK_ID = vocab.unit2id('<UNK>')

    # Once everything is fed through the tokenizer model, it's time to decode the predictions
    # into actual tokens and sentences that the rest of the pipeline uses
    for j in range(len(paragraphs)):
        raw = all_raw[j]
        pred = all_preds[j]

        current_tok = ''
        current_sent = []

        for t, p in zip(raw, pred):
            if t == '<PAD>':
                break
            # hack la_ittb
            if use_la_ittb_shorthand and t in (":", ";"):
                p = 2
            offset += 1
            if vocab.unit2id(t) == UNK_ID:
                oov_count += 1

            current_tok += t
            if p >= 1:
                tok = vocab.normalize_token(current_tok)
                assert '\t' not in tok, tok
                if len(tok) <= 0:
                    current_tok = ''
                    continue
                if orig_text is not None:
                    st = -1
                    tok_len = 0
                    for part in SPACE_SPLIT_RE.split(current_tok):
                        if len(part) == 0: continue
                        if skip_newline:
                            part_pattern = re.compile(r'\s*'.join(re.escape(c) for c in part))
                            match = part_pattern.search(text, char_offset)
                            st0 = match.start(0) - char_offset
                            partlen = match.end(0) - match.start(0)
                        else:
                            st0 = text.index(part, char_offset) - char_offset
                            partlen = len(part)
                        lstripped = part.lstrip()
                        if st < 0:
                            st = char_offset + st0 + (len(part) - len(lstripped))
                        char_offset += st0 + partlen
                    position_info = (st, char_offset)
                else:
                    position_info = None
                current_sent.append((tok, p, position_info))
                current_tok = ''
                if (p == 2 or p == 4) and not no_ssplit:
                    doc.append(process_sentence(current_sent, mwt_dict))
                    current_sent = []

        assert(len(current_tok) == 0)
        if len(current_sent):
            doc.append(process_sentence(current_sent, mwt_dict))

    if output_file: CoNLL.dict2conll(doc, output_file)
    return oov_count, offset, all_preds, doc

def eval_model(args, trainer, batches, vocab, mwt_dict):
    oov_count, N, all_preds, doc = output_predictions(args['conll_file'], trainer, batches, vocab, mwt_dict, args['max_seqlen'])

    all_preds = np.concatenate(all_preds, 0)
    labels = [y[1] for x in batches.data for y in x]
    counter = Counter(zip(all_preds, labels))

    def f1(pred, gold, mapping):
        pred = [mapping[p] for p in pred]
        gold = [mapping[g] for g in gold]

        lastp = -1; lastg = -1
        tp = 0; fp = 0; fn = 0
        for i, (p, g) in enumerate(zip(pred, gold)):
            if p == g > 0 and lastp == lastg:
                lastp = i
                lastg = i
                tp += 1
            elif p > 0 and g > 0:
                lastp = i
                lastg = i
                fp += 1
                fn += 1
            elif p > 0:
                # and g == 0
                lastp = i
                fp += 1
            elif g > 0:
                lastg = i
                fn += 1

        if tp == 0:
            return 0
        else:
            return 2 * tp / (2 * tp + fp + fn)

    f1tok = f1(all_preds, labels, {0:0, 1:1, 2:1, 3:1, 4:1})
    f1sent = f1(all_preds, labels, {0:0, 1:0, 2:1, 3:0, 4:1})
    f1mwt = f1(all_preds, labels, {0:0, 1:1, 2:1, 3:2, 4:2})
    logger.info(f"{args['shorthand']}: token F1 = {f1tok*100:.2f}, sentence F1 = {f1sent*100:.2f}, mwt F1 = {f1mwt*100:.2f}")
    return harmonic_mean([f1tok, f1sent, f1mwt], [1, 1, .01])

