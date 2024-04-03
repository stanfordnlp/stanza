
import pytest

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

from stanza.utils.datasets.prepare_mwt_treebank import check_mwt_composition

SAMPLE_GOOD_TEXT = """
# sent_id = weblog-typepad.com_ripples_20040407125600_ENG_20040407_125600-0057
# text = The Chernobyl Children's Project (http://www.adiccp.org/home/default.asp) offers several ways to help the children of that region.
1	The	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
2	Chernobyl	Chernobyl	PROPN	NNP	Number=Sing	3	compound	3:compound	_
3-4	Children's	_	_	_	_	_	_	_	_
3	Children	Children	PROPN	NNP	Number=Sing	5	nmod:poss	5:nmod:poss	_
4	's	's	PART	POS	_	3	case	3:case	_
5	Project	Project	PROPN	NNP	Number=Sing	9	nsubj	9:nsubj	_
6	(	(	PUNCT	-LRB-	_	7	punct	7:punct	SpaceAfter=No
7	http://www.adiccp.org/home/default.asp	http://www.adiccp.org/home/default.asp	X	ADD	_	5	appos	5:appos	SpaceAfter=No
8	)	)	PUNCT	-RRB-	_	7	punct	7:punct	_
9	offers	offer	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
10	several	several	ADJ	JJ	Degree=Pos	11	amod	11:amod	_
11	ways	way	NOUN	NNS	Number=Plur	9	obj	9:obj	_
12	to	to	PART	TO	_	13	mark	13:mark	_
13	help	help	VERB	VB	VerbForm=Inf	11	acl	11:acl:to	_
14	the	the	DET	DT	Definite=Def|PronType=Art	15	det	15:det	_
15	children	child	NOUN	NNS	Number=Plur	13	obj	13:obj	_
16	of	of	ADP	IN	_	18	case	18:case	_
17	that	that	DET	DT	Number=Sing|PronType=Dem	18	det	18:det	_
18	region	region	NOUN	NN	Number=Sing	15	nmod	15:nmod:of	SpaceAfter=No
19	.	.	PUNCT	.	_	9	punct	9:punct	_
""".lstrip()

SAMPLE_BAD_TEXT = """
# sent_id = weblog-typepad.com_ripples_20040407125600_ENG_20040407_125600-0057
# text = The Chernobyl Children's Project (http://www.adiccp.org/home/default.asp) offers several ways to help the children of that region.
1	The	the	DET	DT	Definite=Def|PronType=Art	3	det	3:det	_
2	Chernobyl	Chernobyl	PROPN	NNP	Number=Sing	3	compound	3:compound	_
3-4	Children's	_	_	_	_	_	_	_	_
3	Childrez	Children	PROPN	NNP	Number=Sing	5	nmod:poss	5:nmod:poss	_
4	's	's	PART	POS	_	3	case	3:case	_
5	Project	Project	PROPN	NNP	Number=Sing	9	nsubj	9:nsubj	_
6	(	(	PUNCT	-LRB-	_	7	punct	7:punct	SpaceAfter=No
7	http://www.adiccp.org/home/default.asp	http://www.adiccp.org/home/default.asp	X	ADD	_	5	appos	5:appos	SpaceAfter=No
8	)	)	PUNCT	-RRB-	_	7	punct	7:punct	_
9	offers	offer	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	0:root	_
10	several	several	ADJ	JJ	Degree=Pos	11	amod	11:amod	_
11	ways	way	NOUN	NNS	Number=Plur	9	obj	9:obj	_
12	to	to	PART	TO	_	13	mark	13:mark	_
13	help	help	VERB	VB	VerbForm=Inf	11	acl	11:acl:to	_
14	the	the	DET	DT	Definite=Def|PronType=Art	15	det	15:det	_
15	children	child	NOUN	NNS	Number=Plur	13	obj	13:obj	_
16	of	of	ADP	IN	_	18	case	18:case	_
17	that	that	DET	DT	Number=Sing|PronType=Dem	18	det	18:det	_
18	region	region	NOUN	NN	Number=Sing	15	nmod	15:nmod:of	SpaceAfter=No
19	.	.	PUNCT	.	_	9	punct	9:punct	_
""".lstrip()

def test_check_mwt_composition(tmp_path):
    mwt_file = tmp_path / "good.mwt"
    with open(mwt_file, "w", encoding="utf-8") as fout:
        fout.write(SAMPLE_GOOD_TEXT)
    check_mwt_composition(mwt_file)

    mwt_file = tmp_path / "bad.mwt"
    with open(mwt_file, "w", encoding="utf-8") as fout:
        fout.write(SAMPLE_BAD_TEXT)
    with pytest.raises(ValueError):
        check_mwt_composition(mwt_file)
