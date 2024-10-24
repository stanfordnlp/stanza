"""
Run the tagger for a couple iterations on some fake data

Uses a couple sentences of UD_English-EWT as training/dev data
"""

import os
import pytest

import torch

from stanza.models import parser
from stanza.models.common import pretrain
from stanza.models.depparse.trainer import Trainer
from stanza.tests import TEST_WORKING_DIR

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


DEV_DATA = """
1	From	from	ADP	IN	_	3	case	3:case	_
2	the	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
3	AP	AP	PROPN	NNP	Number=Sing	4	obl	4:obl:from	_
4	comes	come	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
5	this	this	DET	DT	Number=Sing|PronType=Dem	6	det	6:det	_
6	story	story	NOUN	NN	Number=Sing	4	nsubj	4:nsubj	_
7	:	:	PUNCT	:	_	4	punct	4:punct	_

""".lstrip()



class TestParser:
    @pytest.fixture(scope="class")
    def wordvec_pretrain_file(self):
        return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

    def run_training(self, tmp_path, wordvec_pretrain_file, train_text, dev_text, augment_nopunct=False, extra_args=None):
        """
        Run the training for a few iterations, load & return the model
        """
        train_file = str(tmp_path / "train.conllu")
        dev_file = str(tmp_path / "dev.conllu")
        pred_file = str(tmp_path / "pred.conllu")

        save_name = "test_parser.pt"
        save_file = str(tmp_path / save_name)

        with open(train_file, "w", encoding="utf-8") as fout:
            fout.write(train_text)

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
                # in case we are doing a bert test
                "--bert_start_finetuning", "10",
                "--bert_warmup_steps", "10",
                "--lang", "en"]
        if not augment_nopunct:
            args.extend(["--augment_nopunct", "0.0"])
        if extra_args is not None:
            args = args + extra_args
        trainer = parser.main(args)

        assert os.path.exists(save_file)
        pt = pretrain.Pretrain(wordvec_pretrain_file)
        # test loading the saved model
        saved_model = Trainer(pretrain=pt, model_file=save_file)
        return trainer

    def test_train(self, tmp_path, wordvec_pretrain_file):
        """
        Simple test of a few 'epochs' of tagger training
        """
        self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA)

    def test_with_bert_nlayers(self, tmp_path, wordvec_pretrain_file):
        self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert', '--bert_hidden_layers', '2'])

    def test_with_bert_finetuning(self, tmp_path, wordvec_pretrain_file):
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert', '--bert_finetune', '--bert_hidden_layers', '2'])
        assert 'bert_optimizer' in trainer.optimizer.keys()
        assert 'bert_scheduler' in trainer.scheduler.keys()

    def test_with_bert_finetuning_resaved(self, tmp_path, wordvec_pretrain_file):
        """
        Check that if we save, then load, then save a model with a finetuned bert, that bert isn't lost
        """
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert', '--bert_finetune', '--bert_hidden_layers', '2'])
        assert 'bert_optimizer' in trainer.optimizer.keys()
        assert 'bert_scheduler' in trainer.scheduler.keys()

        save_name = trainer.args['save_name']
        filename = tmp_path / save_name
        assert os.path.exists(filename)
        checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
        assert any(x.startswith("bert_model") for x in checkpoint['model'].keys())

        # Test loading the saved model, saving it, and still having bert in it
        # even if we have set bert_finetune to False for this incarnation
        pt = pretrain.Pretrain(wordvec_pretrain_file)
        args = {"bert_finetune": False}
        saved_model = Trainer(pretrain=pt, model_file=filename, args=args)

        saved_model.save(filename)

        # This is the part that would fail if the force_bert_saved option did not exist
        checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
        assert any(x.startswith("bert_model") for x in checkpoint['model'].keys())

    def test_with_peft(self, tmp_path, wordvec_pretrain_file):
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--bert_model', 'hf-internal-testing/tiny-bert', '--bert_finetune', '--bert_hidden_layers', '2', '--use_peft'])
        assert 'bert_optimizer' in trainer.optimizer.keys()
        assert 'bert_scheduler' in trainer.scheduler.keys()

    def test_single_optimizer_checkpoint(self, tmp_path, wordvec_pretrain_file):
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--optim', 'adam'])

        save_dir = trainer.args['save_dir']
        save_name = trainer.args['save_name']
        checkpoint_name = trainer.args["checkpoint_save_name"]

        assert os.path.exists(os.path.join(save_dir, save_name))
        assert checkpoint_name is not None
        assert os.path.exists(checkpoint_name)

        assert len(trainer.optimizer) == 1
        for opt in trainer.optimizer.values():
            assert isinstance(opt, torch.optim.Adam)

        pt = pretrain.Pretrain(wordvec_pretrain_file)
        checkpoint = Trainer(args=trainer.args, pretrain=pt, model_file=checkpoint_name)
        assert checkpoint.optimizer is not None
        assert len(checkpoint.optimizer) == 1
        for opt in checkpoint.optimizer.values():
            assert isinstance(opt, torch.optim.Adam)

    def test_two_optimizers_checkpoint(self, tmp_path, wordvec_pretrain_file):
        trainer = self.run_training(tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA, extra_args=['--optim', 'adam', '--second_optim', 'sgd', '--second_optim_start_step', '40'])

        save_dir = trainer.args['save_dir']
        save_name = trainer.args['save_name']
        checkpoint_name = trainer.args["checkpoint_save_name"]

        assert os.path.exists(os.path.join(save_dir, save_name))
        assert checkpoint_name is not None
        assert os.path.exists(checkpoint_name)

        assert len(trainer.optimizer) == 1
        for opt in trainer.optimizer.values():
            assert isinstance(opt, torch.optim.SGD)

        pt = pretrain.Pretrain(wordvec_pretrain_file)
        checkpoint = Trainer(args=trainer.args, pretrain=pt, model_file=checkpoint_name)
        assert checkpoint.optimizer is not None
        assert len(checkpoint.optimizer) == 1
        for opt in trainer.optimizer.values():
            assert isinstance(opt, torch.optim.SGD)

