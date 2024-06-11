"""
Constants used for visualization tooling
"""

# Ssurgeon constants
SAMPLE_SSURGEON_DOC = """
    # sent_id = 271
    # text = Hers is easy to clean.
    # previous = What did the dealer like about Alex's car?
    # comment = extraction/raising via "tough extraction" and clausal subject
    1	Hers	hers	PRON	PRP	Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs	3	nsubj	_	_
    2	is	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	cop	_	_
    3	easy	easy	ADJ	JJ	Degree=Pos	0	root	_	_
    4	to	to	PART	TO	_	5	mark	_	_
    5	clean	clean	VERB	VB	VerbForm=Inf	3	csubj	_	SpaceAfter=No
    6	.	.	PUNCT	.	_	5	punct	_	_
    """

# Semgrex constants
DEFAULT_SAMPLE_TEXT = "Banning opal removed artifact decks from the meta."
DEFAULT_SEMGREX_QUERY = "{pos:NN}=object <obl {}=action, {cpos:NOUN}=thing <obj {cpos:VERB}=action"


