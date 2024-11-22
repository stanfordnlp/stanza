"""
Run the tagger for a couple iterations on some fake data

Uses a couple sentences of UD_English-EWT as training/dev data
"""

import os
import pytest

import torch

import stanza
from stanza.models import tagger
from stanza.models.common import pretrain
from stanza.models.pos.trainer import Trainer
from stanza.tests import TEST_WORKING_DIR, TEST_MODELS_DIR
from stanza.utils.training.common import choose_pos_charlm, build_charlm_args

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

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

TRAIN_DATA_2 = """
# sent_id = 11
# text = It's all hers!
# previous = Which person owns this?
# comment = predeterminer modifier
1	It	it	PRON	PRP	Number=Sing|Person=3|PronType=Prs	4	nsubj	_	SpaceAfter=No
2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	cop	_	_
3	all	all	DET	DT	Case=Nom	4	det:predet	_	_
4	hers	hers	PRON	PRP	Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs	0	root	_	SpaceAfter=No
5	!	!	PUNCT	.	_	4	punct	_	_

""".lstrip()

TRAIN_DATA_NO_UPOS = """
# sent_id = 11
# text = It's all hers!
# previous = Which person owns this?
# comment = predeterminer modifier
1	It	it	_	PRP	Number=Sing|Person=3|PronType=Prs	4	nsubj	_	SpaceAfter=No
2	's	be	_	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	cop	_	_
3	all	all	_	DT	Case=Nom	4	det:predet	_	_
4	hers	hers	_	PRP	Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs	0	root	_	SpaceAfter=No
5	!	!	_	.	_	4	punct	_	_

""".lstrip()

TRAIN_DATA_NO_XPOS = """
# sent_id = 11
# text = It's all hers!
# previous = Which person owns this?
# comment = predeterminer modifier
1	It	it	PRON	_	Number=Sing|Person=3|PronType=Prs	4	nsubj	_	SpaceAfter=No
2	's	be	AUX	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	cop	_	_
3	all	all	DET	_	Case=Nom	4	det:predet	_	_
4	hers	hers	PRON	_	Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs	0	root	_	SpaceAfter=No
5	!	!	PUNCT	_	_	4	punct	_	_

""".lstrip()

TRAIN_DATA_NO_FEATS = """
# sent_id = 11
# text = It's all hers!
# previous = Which person owns this?
# comment = predeterminer modifier
1	It	it	PRON	PRP	_	4	nsubj	_	SpaceAfter=No
2	's	be	AUX	VBZ	_	4	cop	_	_
3	all	all	DET	DT	_	4	det:predet	_	_
4	hers	hers	PRON	PRP	_	0	root	_	SpaceAfter=No
5	!	!	PUNCT	.	_	4	punct	_	_

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

class TestTagger:
    @pytest.fixture(scope="class")
    def wordvec_pretrain_file(self):
        return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

    @pytest.fixture(scope="class")
    def charlm_args(self):
        charlm = choose_pos_charlm("en", "test", "default")
        charlm_args = build_charlm_args("en", charlm, model_dir=TEST_MODELS_DIR)
        return charlm_args

    def run_training(self, tmp_path, wordvec_pretrain_file, train_text, dev_text, augment_nopunct=False, extra_args=None):
        """
        Run the training for a few iterations, load & return the model
        """
        dev_file = str(tmp_path / "dev.conllu")
        pred_file = str(tmp_path / "pred.conllu")

        save_name = "test_tagger.pt"
        save_file = str(tmp_path / save_name)

        if isinstance(train_text, str):
            train_text = [train_text]
        train_files = []
        for idx, train_blob in enumerate(train_text):
            train_file = str(tmp_path / ("train_%d.conllu" % idx))
            with open(train_file, "w", encoding="utf-8") as fout:
                fout.write(train_blob)
            train_files.append(train_file)
        train_file = ";".join(train_files)

        with open(dev_file, "w", encoding="utf-8") as fout:
            fout.write(dev_text)

        args = ["--wordvec_pretrain_file", wordvec_pretrain_file,
                "--train_file", train_file,
                "--eval_file", dev_file,
                "--output_file", pred_file,
                "--log_step", "10",
                "--eval_interval", "20",
                "--max_steps", "100",
                "--shorthand", "en_test",
                "--save_dir", str(tmp_path),
                "--save_name", save_name,
                "--lang", "en"]
        if not augment_nopunct:
            args.extend(["--augment_nopunct", "0.0"])
        if extra_args is not None:
            args = args + extra_args
        tagger.main(args)

        assert os.path.exists(save_file)
        pt = pretrain.Pretrain(wordvec_pretrain_file)
        saved_model = Trainer(pretrain=pt, model_file=save_file)
        return saved_model

    def test_train(self, tmp_path, wordvec_pretrain_file, augment_nopunct=True):
        """
        Simple test of a few 'epochs' of tagger training
        """
        self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA)

    def test_vocab_cutoff(self, tmp_path, wordvec_pretrain_file):
        """
        Test that the vocab cutoff leaves words we expect in the vocab, but not rare words
        """
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=["--word_cutoff", "3"])
        word_vocab = trainer.vocab['word']
        assert 'of' in word_vocab
        assert 'officials' in TRAIN_DATA
        assert 'officials' not in word_vocab

    def test_multiple_files(self, tmp_path, wordvec_pretrain_file):
        """
        Test that multiple train files works

        Checks for evidence of it working by looking for words from the second file in the vocab
        """
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, [TRAIN_DATA, TRAIN_DATA_2 * 3], DEV_DATA, extra_args=["--word_cutoff", "3"])
        word_vocab = trainer.vocab['word']
        assert 'of' in word_vocab
        assert 'officials' in TRAIN_DATA
        assert 'officials' not in word_vocab

        assert '	hers	' not in TRAIN_DATA
        assert '	hers	' in TRAIN_DATA_2
        assert 'hers' in word_vocab

    def test_train_zero_augment(self, tmp_path, wordvec_pretrain_file):
        """
        Train with the punct augmentation set to zero

        Distinguishs cases where training works w/ or w/o augmentation
        """
        extra_args = ['--augment_nopunct', '0.0']
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=extra_args)

    def test_train_100_augment(self, tmp_path, wordvec_pretrain_file):
        """
        Train with the punct augmentation set to 1.0

        Distinguishs cases where training works w/ or w/o augmentation
        """
        extra_args = ['--augment_nopunct', '1.0']
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=extra_args)

    def test_train_charlm(self, tmp_path, wordvec_pretrain_file, charlm_args):
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=charlm_args)

    def test_train_charlm_projection(self, tmp_path, wordvec_pretrain_file, charlm_args):
        extra_args = charlm_args + ['--charlm_transform_dim', '100']
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=extra_args)

    def test_missing_column(self, tmp_path, wordvec_pretrain_file):
        """
        Test that using train files with missing columns works

        In this test, we create three separate files, each with a single training entry.
        We then train on an amalgam of those three files with a batch size of 1, saving after each batch.
        This will ensure that only one item is used for each training loop and we can inspect the models which were saved.

        Since each of the three files have exactly one column missing
        from the training data, we expect to see the output maps for
        each column stay unchanged in one iteration and change in the
        other two.
        """
        # use SGD because some old versions of pytorch with Adam keep
        # learning a value even if the loss is 0 in subsequent steps
        # (perhaps it had a momentum by default?)
        extra_args = ['--save_each', '--eval_interval', '1', '--max_steps', '3', '--batch_size', '1', '--optim', 'sgd']
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, [TRAIN_DATA_NO_UPOS, TRAIN_DATA_NO_XPOS, TRAIN_DATA_NO_FEATS], DEV_DATA, extra_args=extra_args)
        save_each_name = tagger.save_each_file_name(trainer.args)
        model_files = [save_each_name % i for i in range(4)]
        assert all(os.path.exists(x) for x in model_files)
        pt = pretrain.Pretrain(wordvec_pretrain_file)
        saved_trainers = [Trainer(pretrain=pt, model_file=model_file) for model_file in model_files]

        upos_unchanged = 0
        xpos_unchanged = 0
        ufeats_unchanged = 0
        for t1, t2 in zip(saved_trainers[:-1], saved_trainers[1:]):
            upos_unchanged += torch.allclose(t1.model.upos_clf.weight, t2.model.upos_clf.weight)
            xpos_unchanged += torch.allclose(t1.model.xpos_clf.W_bilin.weight, t2.model.xpos_clf.W_bilin.weight)
            ufeats_unchanged += all(torch.allclose(f1.W_bilin.weight, f2.W_bilin.weight) for f1, f2 in zip(t1.model.ufeats_clf, t2.model.ufeats_clf))
        upos_norms = [torch.linalg.norm(t.model.upos_clf.weight) for t in saved_trainers]
        assert upos_unchanged == 1, "Unchanged: {} {} {} {}".format(upos_unchanged, xpos_unchanged, ufeats_unchanged, upos_norms)
        assert xpos_unchanged == 1, "Unchanged: %d %d %d" % (upos_unchanged, xpos_unchanged, ufeats_unchanged)
        assert ufeats_unchanged == 1, "Unchanged: %d %d %d" % (upos_unchanged, xpos_unchanged, ufeats_unchanged)

    def test_save_each(self, tmp_path, wordvec_pretrain_file):
        extra_args = ['--save_each']
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=extra_args)
        save_each_name = tagger.save_each_file_name(trainer.args)
        expected_models = sorted(set([save_each_name % i for i in range(0, trainer.args['max_steps']+1, trainer.args['eval_interval'])]))
        assert len(expected_models) == 6
        for model_name in expected_models:
            assert os.path.exists(model_name)


    def test_with_bert(self, tmp_path, wordvec_pretrain_file):
        self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert'])

    def test_with_bert_nlayers(self, tmp_path, wordvec_pretrain_file):
        self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert', '--bert_hidden_layers', '2'])

    def test_with_bert_finetune(self, tmp_path, wordvec_pretrain_file):
        self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert', '--bert_finetune', '--bert_learning_rate', '0.01', '--bert_hidden_layers', '2'])

    def test_bert_pipeline(self, tmp_path, wordvec_pretrain_file):
        """
        Test training the tagger, then using it in a pipeline

        The pipeline use of the tagger also tests the longer-than-maxlen workaround for the transformer
        """
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert'])
        save_name = trainer.args['save_name']
        save_file = str(tmp_path / save_name)
        assert os.path.exists(save_file)

        pipe = stanza.Pipeline("en", processors="tokenize,pos", models_dir=TEST_MODELS_DIR, pos_model_path=save_file, pos_pretrain_path=wordvec_pretrain_file)
        trainer = pipe.processors['pos'].trainer
        assert trainer.args['save_name'] == save_name

        # these should be one chunk only
        doc = pipe("foo " * 100)
        doc = pipe("foo " * 500)
        # this is two chunks of bert embedding
        doc = pipe("foo " * 1000)
        # this is multiple chunks
        doc = pipe("foo " * 2000)
