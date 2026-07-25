from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, MutableSequence, Sequence
import json
import numpy as np
import re
import logging
import os
from typing import Callable, Optional, Protocol, SupportsInt, TYPE_CHECKING, TypedDict, Union

import torch
from torch.utils.data import DataLoader as TorchDataLoader

import stanza.utils.default_paths as default_paths
from stanza.models.common.utils import harmonic_mean
from stanza.models.common.doc import (
    Document,
    TokenEntry,
    TokenEntryId,
    TokenEntryIdComponents,
    END_CHAR,
    ID,
    MEXP,
    MISC,
    START_CHAR,
    TEXT,
)
from stanza.utils.conll import CoNLL, OutputTarget
from stanza.models.tokenization.data import (
    SortedDataset,
    TokenizationDataset,
    WHITESPACE_RE,
)

logger = logging.getLogger('stanza')
paths = default_paths.get_default_paths()

if TYPE_CHECKING:
    from numpy.typing import NDArray

    PredictionLogits = NDArray[np.float32]
    PredictionArray = NDArray[np.intp]
else:
    # ``numpy.typing`` was added in NumPy 1.20.  Keep it out of the runtime
    # import path because Stanza's NumPy dependency has no matching lower
    # bound; Pyright still sees the precise aliases above.
    PredictionLogits = np.ndarray
    PredictionArray = np.ndarray

class TokenizerDictionary(TypedDict):
    words: set[str]
    prefixes: set[str]
    suffixes: set[str]


class TokenizerInferenceArgs(TypedDict):
    batch_size: int
    shorthand: str
    skip_newline: bool


class TokenizerEvaluationArgs(TypedDict):
    conll_file: Optional[OutputTarget]
    max_seqlen: int
    shorthand: str


TokenizationBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[list[str]],
]
TokenizedDocument = list[list[TokenEntry]]
# A bare string is also a Sequence[str], but it is not an MWT expansion.
MwtExpansionWords = Union[list[str], tuple[str, ...]]
MwtDictionary = Mapping[str, tuple[MwtExpansionWords, int]]
TokenPosition = tuple[int, int]
DecodedToken = tuple[str, int, Optional[TokenPosition]]
Predictions = Union[PredictionArray, Sequence[SupportsInt]]
MutablePredictions = Union[PredictionArray, MutableSequence[int]]
PostprocessorEntryValue = Union[
    None,
    bool,
    str,
    TokenEntryId,
    TokenEntryIdComponents,
]
PostprocessorDocumentEntry = Union[
    TokenEntry,
    Mapping[str, PostprocessorEntryValue],
]
PostprocessorDraftToken = Union[str, tuple[str, bool]]
PostprocessorToken = Union[
    str,
    Sequence[Union[str, bool, Iterable[str]]],
]
TokenizerPostprocessor = Callable[
    [list[list[PostprocessorDraftToken]]],
    Iterable[Iterable[PostprocessorToken]],
]


class TokenizerTrainer(Protocol):
    @property
    def args(self) -> Optional[TokenizerInferenceArgs]:
        ...

    def predict(self, inputs: TokenizationBatch) -> PredictionLogits:
        ...


class TokenizerVocab(Protocol):
    def unit2id(self, unit: str) -> int:
        ...

    def normalize_token(self, token: str) -> str:
        ...


def create_dictionary(lexicon: Iterable[str]) -> TokenizerDictionary:
    """
    This function is to create a new dictionary used for improving tokenization model for multi-syllable words languages
    such as vi, zh or th. This function takes the lexicon as input and output a dictionary that contains three set:
    words, prefixes and suffixes where prefixes set should contains all the prefixes in the lexicon and similar for suffixes.
    The point of having prefixes/suffixes sets in the  dictionary is just to make it easier to check during data preparation.

    :param shorthand - language and dataset, eg: vi_vlsp, zh_gsdsimp
    :param lexicon - set of words used to create dictionary
    :return a dictionary object that contains words and their prefixes and suffixes.
    """
    
    dictionary: TokenizerDictionary = {
        "words": set(),
        "prefixes": set(),
        "suffixes": set(),
    }
    
    def add_word(word: str) -> None:
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


def create_lexicon(
        shorthand: Optional[str] = None,
        train_path: Optional[str] = None,
        external_path: Optional[str] = None,
    ) -> tuple[set[str], int]:
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
    lexicon: set[str] = set()
    length_freq: list[int] = []
    count_word = 0
    #this regex is to check if a character is an actual Thai character as seems .isalpha() python method doesn't pick up Thai accent characters..
    pattern_thai = re.compile(r"(?:[^\d\W]+)|\s")
    
    def check_valid_word(
            current_shorthand: Optional[str],
            word: str,
        ) -> bool:
        """
        This function is to check if the word are multi-syllable words and not numbers. 
        For vi, whitespaces are syllabe-separator.
        """
        if current_shorthand is None:
            # Preserve the failure mode of the previous direct ``startswith`` call
            # for callers which omit a shorthand while providing input data.
            raise AttributeError("'NoneType' object has no attribute 'startswith'")
        if current_shorthand.startswith("vi_"):
            return True if len(word.split(" ")) > 1 and any(map(str.isalpha, word)) and not any(map(str.isdigit, word)) else False
        elif current_shorthand.startswith("th_"):
            return True if len(word) > 1 and any(map(pattern_thai.match, word)) and not any(map(str.isdigit, word)) else False
        else:
            return True if len(word) > 1 and any(map(str.isalpha, word)) and not any(map(str.isdigit, word)) else False

    #checking for words in the training set to add them to lexicon.
    if train_path is not None:
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"Cannot open train set at {train_path}")

        train_doc = CoNLL.conll2doc(input_file=train_path)

        for train_sent in train_doc.sentences:
            train_words = [x.text for x in train_sent.tokens if x.is_mwt()] + [x.text for x in train_sent.words]
            for word in train_words:
                word = word.lower()
                if check_valid_word(shorthand, word) and word not in lexicon:
                    lexicon.add(word)
                    length_freq.append(len(word))
        count_word = len(lexicon)
        logger.info(f"Added {count_word} words from the training data to the lexicon.")

    #checking for external dictionary and add them to lexicon.
    if external_path is not None:
        if not os.path.isfile(external_path):
            raise FileNotFoundError(f"Cannot open external dictionary at {external_path}")

        with open(external_path, "r", encoding="utf-8") as external_file:
            lines = external_file.readlines()
        for line in lines:
            word = line.lower()
            word = word.replace("\n","")
            if check_valid_word(shorthand, word) and word not in lexicon:
                lexicon.add(word)
                length_freq.append(len(word))
        logger.info(f"Added another {len(lexicon) - count_word} words from the external dict to dictionary.")
        

    #automatically calculate the number of dictionary features (window size to look for words) based on the frequency of word length
    #take the length at 95-percentile to eliminate all the longest (maybe) compounds words in the lexicon
    num_dict_feat = int(np.percentile(length_freq, 95))
    lexicon = {word for word in lexicon if len(word) <= num_dict_feat }
    logger.info(f"Final lexicon consists of {len(lexicon)} words after getting rid of long words.")

    return lexicon, num_dict_feat


def load_lexicon(args: Mapping[str, str]) -> tuple[set[str], int]:
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
        logger.info(f"External dictionary not found! Looked in {external_dict_path}  Checking training data...")
        external_dict_path = None
    if not os.path.exists(train_path):
        logger.info(f"Training dataset does not exist, thus cannot create dictionary {shorthand}")
        train_path = None
    if train_path is None and external_dict_path is None:
        raise FileNotFoundError(f"Cannot find training set / external dictionary at {train_path} and {external_dict_path}")

    return create_lexicon(shorthand, train_path, external_dict_path)


def load_mwt_dict(filename: Optional[str]) -> Optional[MwtDictionary]:
    """
    Returns a dict from an MWT to its most common expansion and count.

    Other less common expansions are discarded.
    """
    if filename is None:
        return None

    with open(filename, 'r', encoding='utf-8') as f:
        raw_mwt_dict = json.load(f)

    if not isinstance(raw_mwt_dict, list):
        raise ValueError("MWT dictionary must be a list")

    mwt_dict: dict[str, tuple[MwtExpansionWords, int]] = {}
    for raw_item in raw_mwt_dict:
        if not isinstance(raw_item, (list, tuple)) or len(raw_item) != 2:
            raise ValueError("Each MWT dictionary entry must contain a key and count")
        raw_key_and_expansion, count = raw_item
        if not isinstance(raw_key_and_expansion, (list, tuple)) or len(raw_key_and_expansion) != 2:
            raise ValueError("Each MWT dictionary key must contain a token and expansion")
        key, expansion = raw_key_and_expansion
        if not isinstance(key, str):
            raise ValueError("MWT dictionary keys must be strings")
        if not isinstance(expansion, list) or not all(isinstance(word, str) for word in expansion):
            raise ValueError("MWT dictionary expansions must be lists of strings")
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("MWT dictionary counts must be integers")

        if key not in mwt_dict or mwt_dict[key][1] < count:
            mwt_dict[key] = (expansion, count)

    return mwt_dict

def process_sentence(
        sentence: Sequence[DecodedToken],
        mwt_dict: Optional[MwtDictionary] = None,
    ) -> list[TokenEntry]:
    sent: list[TokenEntry] = []
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
URL_RAW_RE = r"""(?:https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s"]{2,}|www\.[a-zA-Z0-9]+\.[^\s"]{2,})|[a-zA-Z0-9]+\.(?:gov|org|edu|net|com|co)(?:\.[^\s"]{2,})"""

MASK_RE = re.compile(f"(?:{EMAIL_RAW_RE}|{URL_RAW_RE})")

def find_spans(raw: Sequence[str]) -> list[tuple[int, int]]:
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


def update_pred_regex(
        raw: Sequence[str],
        pred: MutablePredictions,
    ) -> MutablePredictions:
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

SPACE_SPLIT_RE = re.compile(r'( *[^ ]+)')

def predict(
        trainer: TokenizerTrainer,
        data_generator: TokenizationDataset,
        batch_size: int,
        max_seqlen: int,
        use_regex_tokens: bool,
        num_workers: int,
    ) -> tuple[list[PredictionArray], list[list[str]]]:
    """
    The guts of the prediction method

    Calls trainer.predict() over and over until we have predictions for all of the text
    """
    all_preds: list[PredictionArray] = []
    all_raw: list[list[str]] = []

    sorted_data = SortedDataset(data_generator)
    dataloader = TorchDataLoader(
        sorted_data,
        batch_size=batch_size,
        collate_fn=sorted_data.collate,
        num_workers=num_workers,
    )
    for batch_idx, batch in enumerate(dataloader):
        num_sentences = len(batch[3])
        # being sorted by descending length, we need to use 0 as the longest sentence
        N = len(batch[3][0])
        for paragraph in batch[3]:
            all_raw.append(list(paragraph))

        if N <= max_seqlen:
            predicted_rows = np.argmax(trainer.predict(batch), axis=2)
            pred = [predicted_rows[row_idx] for row_idx in range(num_sentences)]
        else:
            # TODO: we could shortcircuit some processing of
            # long strings of PAD by tracking which rows are finished
            idx = [0] * num_sentences
            adv = [0] * num_sentences
            para_lengths = [x.index('<PAD>') for x in batch[3]]
            prediction_chunks: list[list[PredictionArray]] = [
                [] for _ in range(num_sentences)
            ]
            while True:
                ens = [min(N - idx1, max_seqlen) for idx1, N in zip(idx, para_lengths)]
                en = max(ens)
                batch1 = batch[0][:, :en], batch[1][:, :en], batch[2][:, :en], [x[:en] for x in batch[3]]
                pred1 = np.argmax(trainer.predict(batch1), axis=2)

                for j in range(num_sentences):
                    sentbreaks = np.where((pred1[j] == 2) + (pred1[j] == 4))[0]
                    if len(sentbreaks) <= 0 or idx[j] >= para_lengths[j] - max_seqlen:
                        advance = ens[j]
                    else:
                        advance = int(np.max(sentbreaks)) + 1

                    prediction_chunks[j].append(pred1[j, :advance])
                    idx[j] += advance
                    adv[j] = advance

                if all([idx1 >= N for idx1, N in zip(idx, para_lengths)]):
                    break
                # once we've made predictions on a certain number of characters for each paragraph (recorded in `adv`),
                # we skip the first `adv` characters to make the updated batch
                batch = data_generator.advance_old_batch(adv, batch)

            pred = [np.concatenate(chunks, 0) for chunks in prediction_chunks]

        for par_idx in range(num_sentences):
            offset = batch_idx * batch_size + par_idx

            raw = all_raw[offset]
            par_len = raw.index('<PAD>')
            raw = raw[:par_len]
            all_raw[offset] = raw
            if pred[par_idx][par_len-1] < 2:
                pred[par_idx][par_len-1] = 2
            elif pred[par_idx][par_len-1] > 2:
                pred[par_idx][par_len-1] = 4
            if use_regex_tokens:
                regex_prediction = update_pred_regex(
                    raw,
                    pred[par_idx][:par_len],
                )
                if not isinstance(regex_prediction, np.ndarray):
                    raise TypeError("Tokenizer predictions must be numpy arrays")
                all_preds.append(regex_prediction)
            else:
                all_preds.append(pred[par_idx][:par_len])

    all_preds = sorted_data.unsort(all_preds)
    all_raw = sorted_data.unsort(all_raw)

    return all_preds, all_raw

def output_predictions(
        output_file: Optional[OutputTarget],
        trainer: TokenizerTrainer,
        data_generator: TokenizationDataset,
        vocab: Optional[TokenizerVocab],
        mwt_dict: Optional[MwtDictionary],
        max_seqlen: int = 1000,
        orig_text: Optional[str] = None,
        no_ssplit: bool = False,
        use_regex_tokens: bool = True,
        num_workers: int = 0,
        postprocessor: Optional[TokenizerPostprocessor] = None,
    ) -> tuple[int, int, list[PredictionArray], TokenizedDocument]:
    trainer_args = trainer.args
    if trainer_args is None:
        raise ValueError("Tokenizer trainer arguments have not been initialized")
    batch_size = trainer_args['batch_size']
    max_seqlen = max(1000, max_seqlen)

    all_preds, all_raw = predict(trainer, data_generator, batch_size, max_seqlen, use_regex_tokens, num_workers)

    use_la_ittb_shorthand = trainer_args['shorthand'] == 'la_ittb'
    skip_newline = trainer_args['skip_newline']
    oov_count, offset, doc = decode_predictions(vocab, mwt_dict, orig_text, all_raw, all_preds, no_ssplit, skip_newline, use_la_ittb_shorthand)

    # If we are provided a postprocessor, we prepare a list of pre-tokenized words and mwt flags and
    # call the postprocessor for analysis.
    if postprocessor:
        doc = postprocess_doc(doc, postprocessor, orig_text, all_raw)

    if output_file: CoNLL.dict2conll(doc, output_file)
    return oov_count, offset, all_preds, doc

def postprocess_doc(
        doc: Sequence[Sequence[PostprocessorDocumentEntry]],
        postprocessor: TokenizerPostprocessor,
        orig_text: Optional[str] = None,
        all_raw: Optional[Sequence[Sequence[str]]] = None,
    ) -> TokenizedDocument:
    """Applies a postprocessor on the doc"""

    # get a list of all the words in the "draft" document to pass to the postprocessor
    # the words array looks like [["words, "words", "words"], ["words, ("i_am_a_mwt", True), "I_am_not"]]
    # and the postprocessor is expected to return in the same format
    words: list[list[PostprocessorDraftToken]] = []
    for sentence in doc:
        sentence_words: list[PostprocessorDraftToken] = []
        for word in sentence:
            token_text = word.get(TEXT)
            if not isinstance(token_text, str):
                raise ValueError("Tokenizer document entries must contain text strings")
            sentence_words.append(
                (token_text, True)
                if word.get(MISC) == "MWT=Yes"
                else token_text
            )
        words.append(sentence_words)
    if orig_text is None:
        if all_raw is None:
            raise ValueError("all_raw is required when orig_text is not provided")
        raw_text = "".join("".join(i) for i in all_raw) # template to compare the stitched text against
    else:
        raw_text = orig_text

    # perform correction with the postprocessor
    postprocessor_return = postprocessor(words)

    # collect the words and MWTs separately
    corrected_words: list[list[str]] = []
    corrected_mwts: list[list[bool]] = []
    corrected_expansions: list[list[Optional[str]]] = []

    # for each word, if its just a string (without the ("word", mwt_bool) format)
    # we default that the word is not a MWT.
    for sent in postprocessor_return:
        sent_words: list[str] = []
        sent_mwts: list[bool] = []
        sent_expansions: list[Optional[str]] = []
        for word in sent:
            if isinstance(word, str):
                sent_words.append(word)
                sent_mwts.append(False)
                sent_expansions.append(None)
            else:
                if len(word) != 2:
                    raise ValueError(
                        "Postprocessor token sequences must contain exactly two items"
                    )
                token_text = word[0]
                if not isinstance(token_text, str):
                    raise TypeError(
                        "The first item in a postprocessor token sequence must be a string"
                    )
                mwt_value = word[1]
                if isinstance(mwt_value, bool):
                    sent_words.append(token_text)
                    sent_mwts.append(mwt_value)
                    sent_expansions.append(None)
                else:
                    if not isinstance(mwt_value, Iterable):
                        raise TypeError(
                            "The second item in a postprocessor token sequence must be "
                            "a bool or an iterable of strings"
                        )
                    expansion_words = list(mwt_value)
                    if not all(isinstance(expansion_word, str) for expansion_word in expansion_words):
                        raise TypeError(
                            "Postprocessor token expansions must contain only strings"
                        )
                    sent_words.append(token_text)
                    sent_mwts.append(True)
                    # expansions are marked in a space-separated list, which
                    # `stanza.common.doc.set_mwt_expansions` reads and splits again
                    # by splitting by spaces. Therefore, to serialize the users' supplied MWT
                    # information, we join them by spaces to be split later by
                    # `set_mwt_expansions`.
                    sent_expansions.append(" ".join(expansion_words))
        corrected_words.append(sent_words)
        corrected_mwts.append(sent_mwts)
        corrected_expansions.append(sent_expansions)
        
    # check postprocessor output
    token_lens = [len(i) for i in corrected_words]
    mwt_lens = [len(i) for i in corrected_mwts]
    assert token_lens == mwt_lens, "Postprocessor returned token and MWT lists of different length! Token list lengths %s, MWT list lengths %s" % (token_lens, mwt_lens)
    
    # reassemble document. offsets and oov shouldn't change
    doc = reassemble_doc_from_tokens(corrected_words, corrected_mwts,
                                     corrected_expansions, raw_text)

    return doc

def reassemble_doc_from_tokens(
        tokens: Sequence[Sequence[str]],
        mwts: Sequence[Sequence[bool]],
        expansions: Sequence[Sequence[Optional[str]]],
        raw_text: str,
    ) -> TokenizedDocument:
    """Assemble a Stanza document list format from a list of string tokens, calculating offsets as needed.

    Parameters
    ----------
    tokens : List[List[str]]
        A list of sentences, which includes string tokens.
    mwts : List[List[bool]]
        Whether or not each of the tokens are MWTs to be analyzed by the MWT system.
    expansions : List[List[Optional[List[str]]]]
        A list of possible expansions for MWTs, or None if no user-defined expansion
        is given.
    parser_text : str
        The raw text off of which we can compare offsets.

    Returns
    -------
    List[List[Dict]]
        List of words and their offsets, used as `doc`.
    """

    # oov count and offset stays the same; doc gets regenerated
    new_offset = 0
    corrected_doc: TokenizedDocument = []

    for sent_words, sent_mwts, sent_expansions in zip(tokens, mwts, expansions):
        sentence_doc: list[TokenEntry] = []

        for indx, (word, mwt, expansion) in enumerate(zip(sent_words, sent_mwts, sent_expansions)):
            try:
                offset_index = raw_text.index(word, new_offset)
            except ValueError as e:
                sub_start = max(0, new_offset - 20)
                sub_end = min(len(raw_text), new_offset + 20)
                sub = raw_text[sub_start:sub_end]
                raise ValueError("Could not find word |%s| starting from char_offset %d.  Surrounding text: |%s|. \n Hint: did you accidentally add/subtract a symbol/character such as a space when combining tokens?" % (word, new_offset, sub)) from e

            wd: TokenEntry = {
                "id": (indx+1,), "text": word,
                "start_char":  offset_index,
                "end_char":    offset_index+len(word)
            }
            if expansion:
                wd[MEXP] = True
            elif mwt:
                wd["misc"] = "MWT=Yes"

            sentence_doc.append(wd)

            # start the next search after the previous word ended
            new_offset = offset_index+len(word)

        corrected_doc.append(sentence_doc)

    # use the built in MWT system to expand MWTs
    doc = Document(corrected_doc, raw_text)
    doc.set_mwt_expansions([j
                            for i in expansions
                            for j in i if j],
                           process_manual_expanded=True)
    return doc.to_dict()

def decode_predictions(
        vocab: Optional[TokenizerVocab],
        mwt_dict: Optional[MwtDictionary],
        orig_text: Optional[str],
        all_raw: Sequence[Sequence[str]],
        all_preds: Sequence[Predictions],
        no_ssplit: bool,
        skip_newline: bool,
        use_la_ittb_shorthand: bool,
    ) -> tuple[int, int, TokenizedDocument]:
    """
    Decode the predictions into a document of words

    Once everything is fed through the tokenizer model, it's time to decode the predictions
    into actual tokens and sentences that the rest of the pipeline uses
    """
    offset = 0
    oov_count = 0
    doc: TokenizedDocument = []

    text = WHITESPACE_RE.sub(' ', orig_text) if orig_text is not None else None
    char_offset = 0

    unk_id = vocab.unit2id('<UNK>') if vocab is not None else None

    for raw, pred in zip(all_raw, all_preds):
        current_tok = ''
        current_sent: list[DecodedToken] = []

        for t, raw_prediction in zip(raw, pred):
            p = int(raw_prediction)
            if t == '<PAD>':
                break
            # hack la_ittb
            if use_la_ittb_shorthand and t in (":", ";"):
                p = 2
            offset += 1
            if vocab is not None and vocab.unit2id(t) == unk_id:
                oov_count += 1

            current_tok += t
            if p >= 1:
                if vocab is not None:
                    tok = vocab.normalize_token(current_tok)
                else:
                    tok = current_tok
                assert '\t' not in tok, tok
                if len(tok) <= 0:
                    current_tok = ''
                    continue
                if text is not None:
                    st = -1
                    tok_len = 0
                    for part in SPACE_SPLIT_RE.split(current_tok):
                        if len(part) == 0: continue
                        if skip_newline:
                            part_pattern = re.compile(r'\s*'.join(re.escape(c) for c in part))
                            match = part_pattern.search(text, char_offset)
                            if match is None:
                                raise ValueError(
                                    "Could not find |%s| starting from char_offset %d"
                                    % (part, char_offset)
                                )
                            st0 = match.start(0) - char_offset
                            partlen = match.end(0) - match.start(0)
                            lstripped = match.group(0).lstrip()
                        else:
                            try:
                                st0 = text.index(part, char_offset) - char_offset
                            except ValueError as e:
                                sub_start = max(0, char_offset - 20)
                                sub_end = min(len(text), char_offset + 20)
                                sub = text[sub_start:sub_end]
                                raise ValueError("Could not find |%s| starting from char_offset %d.  Surrounding text: |%s|" % (part, char_offset, sub)) from e
                            partlen = len(part)
                            lstripped = part.lstrip()
                        if st < 0:
                            st = char_offset + st0 + (partlen - len(lstripped))
                        char_offset += st0 + partlen
                    position_info = (st, char_offset)
                else:
                    position_info = None
                current_sent.append((tok, p, position_info))
                current_tok = ''
                if (p == 2 or p == 4) and not no_ssplit:
                    doc.append(process_sentence(current_sent, mwt_dict))
                    current_sent = []

        if len(current_tok) > 0:
            raise ValueError("Finished processing tokens, but there is still text left!")
        if len(current_sent):
            doc.append(process_sentence(current_sent, mwt_dict))

    return oov_count, offset, doc

def match_tokens_with_text(
        sentences: Sequence[Sequence[str]],
        orig_text: str,
    ) -> Document:
    """
    Turns pretokenized text and the original text into a Doc object

    sentences: list of list of string
    orig_text: string, where the text must be exactly the sentences
      concatenated with 0 or more whitespace characters

    if orig_text deviates in any way, a ValueError will be thrown
    """
    text = "".join(["".join(x) for x in sentences])
    all_raw = list(text)
    all_preds = [0] * len(all_raw)
    offset = 0
    for sentence in sentences:
        for word in sentence:
            offset += len(word)
            all_preds[offset-1] = 1
        all_preds[offset-1] = 2
    _, _, doc = decode_predictions(None, None, orig_text, [all_raw], [all_preds], False, False, False)
    doc = Document(doc, orig_text)

    # check that all the orig_text was used up by the tokens
    offset = doc.sentences[-1].tokens[-1].end_char
    remainder = orig_text[offset:].strip()
    if len(remainder) > 0:
        raise ValueError("Finished processing tokens, but there is still text left!")

    return doc


def eval_model(
        args: TokenizerEvaluationArgs,
        trainer: TokenizerTrainer,
        batches: TokenizationDataset,
        vocab: Optional[TokenizerVocab],
        mwt_dict: Optional[MwtDictionary],
    ) -> float:
    oov_count, N, all_preds, doc = output_predictions(args['conll_file'], trainer, batches, vocab, mwt_dict, args['max_seqlen'])

    all_preds = np.concatenate(all_preds, 0)
    labels = np.concatenate(batches.labels())
    counter = Counter(zip(all_preds, labels))

    def f1(
            pred: Iterable[SupportsInt],
            gold: Iterable[SupportsInt],
            mapping: Mapping[int, int],
        ) -> float:
        mapped_pred = [mapping[int(value)] for value in pred]
        mapped_gold = [mapping[int(value)] for value in gold]

        lastp = -1; lastg = -1
        tp = 0; fp = 0; fn = 0
        for i, (p, g) in enumerate(zip(mapped_pred, mapped_gold)):
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
            return 0.0
        else:
            return 2 * tp / (2 * tp + fp + fn)

    f1tok = f1(all_preds, labels, {0:0, 1:1, 2:1, 3:1, 4:1})
    f1sent = f1(all_preds, labels, {0:0, 1:0, 2:1, 3:0, 4:1})
    f1mwt = f1(all_preds, labels, {0:0, 1:1, 2:1, 3:2, 4:2})
    logger.info(f"{args['shorthand']}: token F1 = {f1tok*100:.2f}, sentence F1 = {f1sent*100:.2f}, mwt F1 = {f1mwt*100:.2f}")
    return float(harmonic_mean([f1tok, f1sent, f1mwt], [1, 1, .01]))
