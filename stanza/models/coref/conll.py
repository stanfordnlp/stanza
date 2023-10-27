""" Contains functions to produce conll-formatted output files with
predicted spans and their clustering """

from collections import defaultdict
from contextlib import contextmanager
import os
from typing import List, TextIO

from coref.config import Config
from coref.const import Doc, Span


# pylint: disable=too-many-locals
def write_conll(doc: Doc,
                clusters: List[List[Span]],
                f_obj: TextIO):
    """ Writes span/cluster information to f_obj, which is assumed to be a file
    object open for writing """
    placeholder = "  -" * 7
    doc_id = doc["document_id"]
    words = doc["cased_words"]
    part_id = doc["part_id"]
    sents = doc["sent_id"]

    max_word_len = max(len(w) for w in words)

    starts = defaultdict(lambda: [])
    ends = defaultdict(lambda: [])
    single_word = defaultdict(lambda: [])

    for cluster_id, cluster in enumerate(clusters):
        for start, end in cluster:
            if end - start == 1:
                single_word[start].append(cluster_id)
            else:
                starts[start].append(cluster_id)
                ends[end - 1].append(cluster_id)

    f_obj.write(f"#begin document ({doc_id}); part {part_id:0>3d}\n")

    word_number = 0
    for word_id, word in enumerate(words):

        cluster_info_lst = []
        for cluster_marker in starts[word_id]:
            cluster_info_lst.append(f"({cluster_marker}")
        for cluster_marker in single_word[word_id]:
            cluster_info_lst.append(f"({cluster_marker})")
        for cluster_marker in ends[word_id]:
            cluster_info_lst.append(f"{cluster_marker})")
        cluster_info = "|".join(cluster_info_lst) if cluster_info_lst else "-"

        if word_id == 0 or sents[word_id] != sents[word_id - 1]:
            f_obj.write("\n")
            word_number = 0

        f_obj.write(f"{doc_id}  {part_id}  {word_number:>2}"
                    f"  {word:>{max_word_len}}{placeholder}  {cluster_info}\n")

        word_number += 1

    f_obj.write("#end document\n\n")


@contextmanager
def open_(config: Config, epochs: int, data_split: str):
    """ Opens conll log files for writing in a safe way. """
    base_filename = f"{config.section}_{data_split}_e{epochs}"
    conll_dir = config.conll_log_dir
    kwargs = {"mode": "w", "encoding": "utf8"}

    os.makedirs(conll_dir, exist_ok=True)

    with open(os.path.join(  # type: ignore
            conll_dir, f"{base_filename}.gold.conll"), **kwargs) as gold_f:
        with open(os.path.join(  # type: ignore
                conll_dir, f"{base_filename}.pred.conll"), **kwargs) as pred_f:
            yield (gold_f, pred_f)
