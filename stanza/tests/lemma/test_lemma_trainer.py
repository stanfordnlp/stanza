"""
Test a couple basic functions - load & save an existing model
"""

import pytest

import glob
import os
import tempfile

import torch

from stanza.models import lemmatizer
from stanza.models.lemma import trainer
from stanza.tests import *
from stanza.utils.training.common import choose_lemma_charlm, build_charlm_args

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

@pytest.fixture(scope="module")
def english_model():
    models_path = os.path.join(TEST_MODELS_DIR, "en", "lemma", "*")
    models = glob.glob(models_path)
    # we expect at least one English model downloaded for the tests
    assert len(models) >= 1
    model_file = models[0]
    return trainer.Trainer(model_file=model_file)

def test_load_model(english_model):
    """
    Does nothing, just tests that loading works
    """

def test_save_load_model(english_model):
    """
    Load, save, and load again
    """
    with tempfile.TemporaryDirectory() as tempdir:
        save_file = os.path.join(tempdir, "resaved", "lemma.pt")
        english_model.save(save_file)
        reloaded = trainer.Trainer(model_file=save_file)

TRAIN_DATA = """
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0003
# text = DPA: Iraqi authorities announced that they had busted up 3 terrorist cells operating in Baghdad.
1	DPA	DPA	PROPN	NNP	Number=Sing	0	root	0:root	SpaceAfter=No
2	:	:	PUNCT	:	_	1	punct	1:punct	_
3	Iraqi	Iraqi	ADJ	JJ	Degree=Pos	4	amod	4:amod	_
4	authorities	authority	NOUN	NNS	Number=Plur	5	nsubj	5:nsubj	_
5	announced	announce	VERB	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	1	parataxis	1:parataxis	_
6	that	that	SCONJ	IN	_	9	mark	9:mark	_
7	they	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	9	nsubj	9:nsubj	_
8	had	have	AUX	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	9	aux	9:aux	_
9	busted	bust	VERB	VBN	Tense=Past|VerbForm=Part	5	ccomp	5:ccomp	_
10	up	up	ADP	RP	_	9	compound:prt	9:compound:prt	_
11	3	3	NUM	CD	NumForm=Digit|NumType=Card	13	nummod	13:nummod	_
12	terrorist	terrorist	ADJ	JJ	Degree=Pos	13	amod	13:amod	_
13	cells	cell	NOUN	NNS	Number=Plur	9	obj	9:obj	_
14	operating	operate	VERB	VBG	VerbForm=Ger	13	acl	13:acl	_
15	in	in	ADP	IN	_	16	case	16:case	_
16	Baghdad	Baghdad	PROPN	NNP	Number=Sing	14	obl	14:obl:in	SpaceAfter=No
17	.	.	PUNCT	.	_	1	punct	1:punct	_

# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0004
# text = Two of them were being run by 2 officials of the Ministry of the Interior!
1	Two	two	NUM	CD	NumForm=Word|NumType=Card	6	nsubj:pass	6:nsubj:pass	_
2	of	of	ADP	IN	_	3	case	3:case	_
3	them	they	PRON	PRP	Case=Acc|Number=Plur|Person=3|PronType=Prs	1	nmod	1:nmod:of	_
4	were	be	AUX	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	6	aux	6:aux	_
5	being	be	AUX	VBG	VerbForm=Ger	6	aux:pass	6:aux:pass	_
6	run	run	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	0	root	0:root	_
7	by	by	ADP	IN	_	9	case	9:case	_
8	2	2	NUM	CD	NumForm=Digit|NumType=Card	9	nummod	9:nummod	_
9	officials	official	NOUN	NNS	Number=Plur	6	obl	6:obl:by	_
10	of	of	ADP	IN	_	12	case	12:case	_
11	the	the	DET	DT	Definite=Def|PronType=Art	12	det	12:det	_
12	Ministry	Ministry	PROPN	NNP	Number=Sing	9	nmod	9:nmod:of	_
13	of	of	ADP	IN	_	15	case	15:case	_
14	the	the	DET	DT	Definite=Def|PronType=Art	15	det	15:det	_
15	Interior	Interior	PROPN	NNP	Number=Sing	12	nmod	12:nmod:of	SpaceAfter=No
16	!	!	PUNCT	.	_	6	punct	6:punct	_

""".lstrip()

DEV_DATA = """
1	From	from	ADP	IN	_	3	case	3:case	_
2	the	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
3	AP	AP	PROPN	NNP	Number=Sing	4	obl	4:obl:from	_
4	comes	come	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
5	this	this	DET	DT	Number=Sing|PronType=Dem	6	det	6:det	_
6	story	story	NOUN	NN	Number=Sing	4	nsubj	4:nsubj	_
7	:	:	PUNCT	:	_	4	punct	4:punct	_

""".lstrip()

class TestLemmatizer:
    @pytest.fixture(scope="class")
    def charlm_args(self):
        charlm = choose_lemma_charlm("en", "test", "default")
        charlm_args = build_charlm_args("en", charlm, model_dir=TEST_MODELS_DIR)
        return charlm_args


    def run_training(self, tmp_path, train_text, dev_text, extra_args=None):
        """
        Run the training for a few iterations, load & return the model
        """
        pred_file = str(tmp_path / "pred.conllu")

        save_name = "test_tagger.pt"
        save_file = str(tmp_path / save_name)

        train_file = str(tmp_path / "train.conllu")
        with open(train_file, "w", encoding="utf-8") as fout:
            fout.write(train_text)

        dev_file = str(tmp_path / "dev.conllu")
        with open(dev_file, "w", encoding="utf-8") as fout:
            fout.write(dev_text)

        args = ["--train_file", train_file,
                "--eval_file", dev_file,
                "--gold_file", dev_file,
                "--output_file", pred_file,
                "--num_epoch", "2",
                "--log_step", "10",
                "--save_dir", str(tmp_path),
                "--save_name", save_name,
                "--shorthand", "en_test"]
        if extra_args is not None:
            args = args + extra_args
        lemmatizer.main(args)

        assert os.path.exists(save_file)
        saved_model = trainer.Trainer(model_file=save_file)
        return saved_model

    def test_basic_train(self, tmp_path):
        """
        Simple test of a few 'epochs' of lemmatizer training
        """
        self.run_training(tmp_path, TRAIN_DATA, DEV_DATA)

    def test_charlm_train(self, tmp_path, charlm_args):
        """
        Simple test of a few 'epochs' of lemmatizer training
        """
        saved_model = self.run_training(tmp_path, TRAIN_DATA, DEV_DATA, extra_args=charlm_args)

        # check that the charlm wasn't saved in here
        args = saved_model.args
        save_name = os.path.join(args['save_dir'], args['save_name'])
        checkpoint = torch.load(save_name, lambda storage, loc: storage, weights_only=True)
        assert not any(x.startswith("contextual_embedding") for x in checkpoint['model'].keys())
