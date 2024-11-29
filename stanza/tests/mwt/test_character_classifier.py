import os
import pytest

from stanza.models import mwt_expander
from stanza.models.mwt.character_classifier import CharacterClassifier
from stanza.models.mwt.data import DataLoader
from stanza.models.mwt.trainer import Trainer
from stanza.utils.conll import CoNLL

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

ENG_TRAIN = """
# text = Elena's motorcycle tour
1-2	Elena's	_	_	_	_	_	_	_	_
1	Elena	Elena	PROPN	NNP	Number=Sing	4	nmod:poss	4:nmod:poss	_
2	's	's	PART	POS	_	1	case	1:case	_
3	motorcycle	motorcycle	NOUN	NN	Number=Sing	4	compound	4:compound	_
4	tour	tour	NOUN	NN	Number=Sing	0	root	0:root	_


# text = women's reproductive health
1-2	women's	_	_	_	_	_	_	_	_
1	women	woman	NOUN	NNS	Number=Plur	4	nmod:poss	4:nmod:poss	_
2	's	's	PART	POS	_	1	case	1:case	_
3	reproductive	reproductive	ADJ	JJ	Degree=Pos	4	amod	4:amod	_
4	health	health	NOUN	NN	Number=Sing	0	root	0:root	SpaceAfter=No


# text = The Chernobyl Children's Project
1	The	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
2	Chernobyl	Chernobyl	PROPN	NNP	Number=Sing	3	compound	3:compound	_
3-4	Children's	_	_	_	_	_	_	_	_
3	Children	Children	PROPN	NNP	Number=Sing	5	nmod:poss	5:nmod:poss	_
4	's	's	PART	POS	_	3	case	3:case	_
5	Project	Project	PROPN	NNP	Number=Sing	0	root	0:root	_

""".lstrip()

ENG_DEV = """
# text = The Chernobyl Children's Project
1	The	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
2	Chernobyl	Chernobyl	PROPN	NNP	Number=Sing	3	compound	3:compound	_
3-4	Children's	_	_	_	_	_	_	_	_
3	Children	Children	PROPN	NNP	Number=Sing	5	nmod:poss	5:nmod:poss	_
4	's	's	PART	POS	_	3	case	3:case	_
5	Project	Project	PROPN	NNP	Number=Sing	0	root	0:root	_

""".lstrip()

def test_train(tmp_path):
    test_train = str(os.path.join(tmp_path, "en_test.train.conllu"))
    with open(test_train, "w") as fout:
        fout.write(ENG_TRAIN)

    test_dev = str(os.path.join(tmp_path, "en_test.dev.conllu"))
    with open(test_dev, "w") as fout:
        fout.write(ENG_DEV)

    test_output = str(os.path.join(tmp_path, "en_test.dev.pred.conllu"))
    model_name = "en_test_mwt.pt"

    args = [
        "--data_dir", str(tmp_path),
        "--train_file", test_train,
        "--eval_file", test_dev,
        "--gold_file", test_dev,
        "--lang", "en",
        "--shorthand", "en_test",
        "--output_file", test_output,
        "--save_dir", str(tmp_path),
        "--save_name", model_name,
        "--num_epoch", "10",
    ]

    mwt_expander.main(args=args)

    model = Trainer(model_file=os.path.join(tmp_path, model_name))
    assert model.model is not None
    assert isinstance(model.model, CharacterClassifier)

    doc = CoNLL.conll2doc(input_str=ENG_DEV)
    dataloader = DataLoader(doc, 10, model.args, vocab=model.vocab, evaluation=True, expand_unk_vocab=True)
    preds = []
    for i, batch in enumerate(dataloader.to_loader()):
        assert i == 0 # there should only be one batch
        preds += model.predict(batch, never_decode_unk=True, vocab=dataloader.vocab)
    assert len(preds) == 1
    # it is possible to make a version of the test where this happens almost every time
    # for example, running for 100 epochs makes the model succeed 30 times in a row
    # (never saw a failure)
    # but the one time that failure happened, it would be really annoying
    #assert preds[0] == "Children 's"
