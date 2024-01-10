import os

import pytest

import stanza.models.lemma_classifier.utils as utils
import stanza.utils.datasets.prepare_lemma_classifier as prepare_lemma_classifier

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

EWT_ONE_SENTENCE = """
# sent_id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-0002
# newpar id = weblog-blogspot.com_rigorousintuition_20050518101500_ENG_20050518_101500-p0002
# text = Here's a Miami Herald interview
1-2	Here's	_	_	_	_	_	_	_	_
1	Here	here	ADV	RB	PronType=Dem	0	root	0:root	_
2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	1	cop	1:cop	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	6	det	6:det	_
4	Miami	Miami	PROPN	NNP	Number=Sing	5	compound	5:compound	_
5	Herald	Herald	PROPN	NNP	Number=Sing	6	compound	6:compound	_
6	interview	interview	NOUN	NN	Number=Sing	1	nsubj	1:nsubj	_
""".lstrip()


def test_convert_one_sentence(tmp_path):
    ud_path = tmp_path / "ud"
    input_path = ud_path / "UD_English-EWT"
    output_path = tmp_path / "data" / "lemma_classifier"

    os.makedirs(input_path, exist_ok=True)
    sample_file = input_path / "en_ewt-ud-train.conllu"
    with open(sample_file, "w", encoding="utf-8") as fout:
        fout.write(EWT_ONE_SENTENCE)

    paths = {"UDBASE": ud_path,
             "LEMMA_CLASSIFIER_DATA_DIR": output_path}

    converted_files = prepare_lemma_classifier.process_treebank(paths, "en_ewt", "'s", "AUX", "be|have", ["train"])
    assert len(converted_files) == 1

    text_batches, idx_batches, upos_batches, label_batches, counts, label_decoder, upos_to_id = utils.load_dataset(converted_files[0], get_counts=True, batch_size=10)
    assert text_batches == [[['Here', "'s", 'a', 'Miami', 'Herald', 'interview']]]
    assert label_decoder == {'be': 0}
    id_to_upos = {y: x for x, y in upos_to_id.items()}
    upos = [id_to_upos[x] for x in upos_batches[0][0]]
    assert upos == ['ADV', 'AUX', 'DET', 'PROPN', 'PROPN', 'NOUN']
