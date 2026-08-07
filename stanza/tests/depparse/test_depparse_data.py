"""
Test some pieces of the depparse dataloader
"""
import random
from collections import Counter

import pytest
from stanza.models import parser
from stanza.models.depparse.data import (
    data_to_batches, DataLoader, Dataset, DepparseBatchSampler, InfiniteBatch, to_int,
    record_ends_with_punct, record_can_augment_nopunct,
    record_starts_with_mark, record_can_drop_initial_mark,
)
from stanza.utils.conll import CoNLL

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

def make_fake_data(*lengths):
    data = []
    for i, length in enumerate(lengths):
        word = chr(ord('A') + i)
        chunk = [[word] * length]
        data.append(chunk)
    return data

def check_batches(batched_data, expected_sizes, expected_order):
    for chunk, size in zip(batched_data, expected_sizes):
        assert sum(len(x[0]) for x in chunk) == size
    word_order = []
    for chunk in batched_data:
        for sentence in chunk:
            word_order.append(sentence[0][0])
    assert word_order == expected_order

def test_data_to_batches_eval_mode():
    """
    Tests the chunking of batches in eval_mode

    A few options are tested, such as whether or not to sort and the maximum sentence size
    """
    data = make_fake_data(1, 2, 3)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [5, 1], ['C', 'B', 'A'])

    data = make_fake_data(1, 2, 6)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [6, 3], ['C', 'B', 'A'])

    data = make_fake_data(3, 2, 1)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [5, 1], ['A', 'B', 'C'])

    data = make_fake_data(3, 5, 2)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=True, min_length_to_batch_separately=None)
    check_batches(batched_data[0], [5, 5], ['B', 'A', 'C'])

    data = make_fake_data(3, 5, 2)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=False, min_length_to_batch_separately=3)
    check_batches(batched_data[0], [3, 5, 2], ['A', 'B', 'C'])

    data = make_fake_data(4, 1, 1)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=False, min_length_to_batch_separately=3)
    check_batches(batched_data[0], [4, 2], ['A', 'B', 'C'])

    data = make_fake_data(1, 4, 1)
    batched_data = data_to_batches(data, batch_size=5, eval_mode=True, sort_during_eval=False, min_length_to_batch_separately=3)
    check_batches(batched_data[0], [1, 4, 1], ['A', 'B', 'C'])


EWT_PUNCT_SAMPLE = """
# sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0048
# text = Bush asked for permission to go to Alabama to work on a Senate campaign.
1	Bush	Bush	PROPN	NNP	Number=Sing	2	nsubj	2:nsubj	_
2	asked	ask	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
3	for	for	ADP	IN	_	4	case	4:case	_
4	permission	permission	NOUN	NN	Number=Sing	2	obl	2:obl:for	_
5	to	to	PART	TO	_	6	mark	6:mark	_
6	go	go	VERB	VB	VerbForm=Inf	4	acl	4:acl:to	_
7	to	to	ADP	IN	_	8	case	8:case	_
8	Alabama	Alabama	PROPN	NNP	Number=Sing	6	obl	6:obl:to	_
9	to	to	PART	TO	_	10	mark	10:mark	_
10	work	work	VERB	VB	VerbForm=Inf	6	advcl	6:advcl:to	_
11	on	on	ADP	IN	_	14	case	14:case	_
12	a	a	DET	DT	Definite=Ind|PronType=Art	14	det	14:det	_
13	Senate	Senate	PROPN	NNP	Number=Sing	14	compound	14:compound	_
14	campaign	campaign	NOUN	NN	Number=Sing	10	obl	10:obl:on	SpaceAfter=No
15	!!!!!	!	PUNCT	.	_	2	punct	2:punct	_

# sent_id = weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0049
# text = His superior officers said OK.
1	His	his	PRON	PRP$	Case=Gen|Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs	3	nmod:poss	3:nmod:poss	_
2	superior	superior	ADJ	JJ	Degree=Pos	3	amod	3:amod	_
3	officers	officer	NOUN	NNS	Number=Plur	4	nsubj	4:nsubj	_
4	said	say	VERB	VBD	Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin	0	root	0:root	_
5	OK	ok	INTJ	UH	_	4	obj	4:obj	SpaceAfter=No
6	?????	?	PUNCT	.	_	4	punct	4:punct	_
"""


def test_punct_simplification():
    """
    Test a punctuation simplification that should make it so unexpected
    question/exclamation marks types are processed into ? and !
    """
    sample = CoNLL.conll2doc(input_str=EWT_PUNCT_SAMPLE)

    args = parser.parse_args(args=["--batch_size", "1000", "--shorthand", "en_test", "--augment_nopunct", "0"])
    data = DataLoader(sample, 5000, args, None)

    batches = [batch for batch in data]
    assert batches[0][-1] == [['Bush', 'asked', 'for', 'permission', 'to', 'go', 'to', 'Alabama', 'to', 'work', 'on', 'a', 'Senate', 'campaign', '!'],
                              ['His', 'superior', 'officers', 'said', 'OK', '?']]


# ---------------------------------------------------------------------------
# Dataset / DataLoader split
# ---------------------------------------------------------------------------
#
# DataLoader used to preprocess and tensorize every sentence exactly once,
# at construction time. It is now a thin batch-retrieval wrapper around a
# separate Dataset class (mirroring stanza.models.pos.data.Dataset), which
# fetches each sentence fresh from Dataset.__getitem__ every time a batch
# is materialized. The tests below check that this split preserves the
# previous observable behavior (data_orig_idx, set_batch_size, reshuffle)
# and that the new mechanism -- fresh per-item fetch on every access --
# actually holds, since that is the property a future per-item augmentation
# (comparable to the POS tagger's punctuation-drop augmentation) would
# depend on.

THREE_SENTENCE_SAMPLE = """
# sent_id = a
# text = Short one.
1	Short	short	ADJ	JJ	_	2	amod	2:amod	_
2	one	one	NOUN	NN	_	0	root	0:root	_
3	.	.	PUNCT	.	_	2	punct	2:punct	_

# sent_id = b
# text = This is a much longer sentence with many more words in it.
1	This	this	PRON	DT	_	2	nsubj	2:nsubj	_
2	is	be	AUX	VBZ	_	0	root	0:root	_
3	a	a	DET	DT	_	7	det	7:det	_
4	much	much	ADV	RB	_	5	advmod	5:advmod	_
5	longer	long	ADJ	JJR	_	7	amod	7:amod	_
6	sentence	sentence	NOUN	NN	_	7	compound	7:compound	_
7	with	with	ADP	IN	_	2	obl	2:obl	_
8	many	many	ADJ	JJ	_	9	amod	9:amod	_
9	more	more	ADJ	JJR	_	10	amod	10:amod	_
10	words	word	NOUN	NNS	_	7	obl	7:obl	_
11	in	in	ADP	IN	_	12	case	12:case	_
12	it	it	PRON	PRP	_	10	obl	10:obl	_
13	.	.	PUNCT	.	_	2	punct	2:punct	_

# sent_id = c
# text = Medium length one here.
1	Medium	medium	ADJ	JJ	_	3	amod	3:amod	_
2	length	length	NOUN	NN	_	3	compound	3:compound	_
3	one	one	NOUN	NN	_	0	root	0:root	_
4	here	here	ADV	RB	_	3	advmod	3:advmod	_
5	.	.	PUNCT	.	_	3	punct	3:punct	_
"""


def test_dataset_getitem_called_fresh_across_epochs():
    """
    Dataset.__getitem__ must be invoked fresh every time a batch is
    materialized -- not cached once and reused -- so that a future
    per-item augmentation added there would be re-rolled every epoch,
    the same mechanism the POS tagger's Dataset already relies on.

    Pulls more batches than exist in a single epoch via InfiniteBatch
    (which reshuffles on exhaustion), and checks Dataset.__getitem__ was
    called more times than there are sentences in the dataset -- i.e.
    it was genuinely re-invoked across multiple epochs, not served from
    a cache computed once.
    """
    sample = CoNLL.conll2doc(input_str=THREE_SENTENCE_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "8", "--shorthand", "en_test"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)

    call_count = [0]
    orig_getitem = Dataset.__getitem__
    def counting_getitem(self, key):
        call_count[0] += 1
        return orig_getitem(self, key)
    Dataset.__getitem__ = counting_getitem
    try:
        infinite_batch = InfiniteBatch(train_batch)
        for _ in range(10):
            infinite_batch.next_batch()
    finally:
        Dataset.__getitem__ = orig_getitem

    num_sentences = len(train_batch.dataset)
    assert call_count[0] > num_sentences, (
        "expected Dataset.__getitem__ to be called across multiple epochs, "
        "not just once per sentence"
    )


def test_data_orig_idx_unsorts_correctly():
    """
    eval_batch.data_orig_idx, produced by the new DepparseBatchSampler,
    must still correctly unsort predictions back to the original document
    order -- this is the exact mechanism stanza.models.depparse.utils.
    predict_dataset relies on.
    """
    from stanza.models.common import utils

    sample = CoNLL.conll2doc(input_str=THREE_SENTENCE_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "5", "--shorthand", "en_test"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    vocab = train_batch.vocab
    eval_batch = DataLoader(sample, args['batch_size'], args, None, vocab=vocab, evaluation=True, sort_during_eval=True)

    all_texts = []
    for batch in eval_batch:
        all_texts.extend(batch[-1])

    unsorted_texts = utils.unsort(all_texts, eval_batch.data_orig_idx)
    assert unsorted_texts == [
        ['Short', 'one', '.'],
        ['This', 'is', 'a', 'much', 'longer', 'sentence', 'with', 'many', 'more', 'words', 'in', 'it', '.'],
        ['Medium', 'length', 'one', 'here', '.'],
    ]


def test_data_orig_idx_none_when_not_sorted():
    """data_orig_idx should be None for train mode and for eval without sort_during_eval."""
    sample = CoNLL.conll2doc(input_str=THREE_SENTENCE_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "5", "--shorthand", "en_test"])

    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    assert train_batch.data_orig_idx is None

    vocab = train_batch.vocab
    eval_batch = DataLoader(sample, args['batch_size'], args, None, vocab=vocab, evaluation=True, sort_during_eval=False)
    assert eval_batch.data_orig_idx is None


def test_set_batch_size_and_reshuffle():
    """
    set_batch_size followed by reshuffle (as done in parser.py when
    switching to a second optimizer with a different batch size) must
    actually take effect on the next round of batches.
    """
    sample = CoNLL.conll2doc(input_str=THREE_SENTENCE_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "100", "--shorthand", "en_test"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    assert len(train_batch) == 1  # all 3 sentences fit in one batch at size 100

    train_batch.set_batch_size(1)
    train_batch.reshuffle()
    assert train_batch.batch_size == 1
    assert len(train_batch) == 3  # each sentence now forced into its own batch


def test_min_length_to_batch_separately():
    """A sentence longer than min_length_to_batch_separately gets its own batch."""
    sample = CoNLL.conll2doc(input_str=THREE_SENTENCE_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "100", "--shorthand", "en_test"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False,
                              min_length_to_batch_separately=8)
    # the 13-word sentence exceeds 8 and should be isolated into its own batch,
    # while the other two (short) sentences share a batch together
    assert len(train_batch) == 2


def test_reversed_sentences():
    """The 'reversed' arg should reverse word order within each sentence."""
    sample = CoNLL.conll2doc(input_str=THREE_SENTENCE_SAMPLE)
    args = dict(parser.parse_args(args=["--batch_size", "100", "--shorthand", "en_test"]))
    args['reversed'] = True
    args['augment_nopunct'] = 0
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)

    all_texts = []
    for batch in train_batch:
        all_texts.extend(batch[-1])
    assert ['.', 'one', 'Short'] in all_texts


# ---------------------------------------------------------------------------
# to_int
# ---------------------------------------------------------------------------

def test_to_int_valid():
    assert to_int("5") == 5

def test_to_int_invalid_raises_by_default():
    with pytest.raises(ValueError):
        to_int("_")

def test_to_int_invalid_ignored():
    assert to_int("_", ignore_error=True) == 0


# ---------------------------------------------------------------------------
# Dataset.preprocess: ROOT-token prepending
# ---------------------------------------------------------------------------
#
# words, chars, upos, xpos, feats, pretrain, and lemma are each prepended
# with a ROOT-token placeholder (so the model has an explicit ROOT input
# position); head, deprel, and text are NOT prepended, since they are
# per-real-word outputs.  These tests pin that structural distinction
# down directly on Dataset.__getitem__'s raw (un-tensorized) output.

TWO_WORD_SAMPLE = """
# sent_id = a
# text = Cats sleep.
1	Cats	cat	NOUN	NNS	Number=Plur	2	nsubj	2:nsubj	_
2	sleep	sleep	VERB	VBZ	Mood=Ind|Tense=Pres	0	root	0:root	_
3	.	.	PUNCT	.	_	2	punct	2:punct	_
"""

def _build_train_loader(sample_str, batch_size=100, extra_args=None):
    sample = CoNLL.conll2doc(input_str=sample_str)
    args = dict(parser.parse_args(args=["--batch_size", str(batch_size), "--shorthand", "en_test"]))
    if extra_args:
        args.update(extra_args)
    return DataLoader(sample, args['batch_size'], args, None, evaluation=False)

def test_preprocess_root_prepended_fields():
    from stanza.models.common.vocab import ROOT_ID

    train_batch = _build_train_loader(TWO_WORD_SAMPLE)
    rec = train_batch.dataset.data[0]
    word, char, upos, xpos, feats, pretrain, lemma, head, deprel, text = rec

    num_words = 3  # Cats, sleep, .
    # ROOT-prepended fields: length == num_words + 1, and start with ROOT_ID
    for field, name in [(word, 'word'), (upos, 'upos'), (xpos, 'xpos'),
                         (pretrain, 'pretrain'), (lemma, 'lemma')]:
        assert len(field) == num_words + 1, f"{name} should be ROOT-prepended"
        assert field[0] == ROOT_ID, f"{name}[0] should be ROOT_ID"
    assert len(char) == num_words + 1 and char[0] == [ROOT_ID]
    assert len(feats) == num_words + 1

    # NOT ROOT-prepended: length == num_words exactly
    assert len(head) == num_words
    assert len(deprel) == num_words
    assert len(text) == num_words
    assert text == ['Cats', 'sleep', '.']

def test_preprocess_head_values():
    """head[i] should be the 1-indexed position of word i's head, 0 for the root word."""
    train_batch = _build_train_loader(TWO_WORD_SAMPLE)
    rec = train_batch.dataset.data[0]
    head = rec[7]
    # Cats -> sleep (position 2), sleep -> root (0), . -> sleep (position 2)
    assert head == [2, 0, 2]

def test_preprocess_no_pretrain_gives_pad_id():
    """With no pretrain vocab, the pretrain field should be all PAD_ID (except ROOT)."""
    from stanza.models.common.vocab import ROOT_ID, PAD_ID

    train_batch = _build_train_loader(TWO_WORD_SAMPLE)
    rec = train_batch.dataset.data[0]
    pretrain_field = rec[5]
    assert pretrain_field[0] == ROOT_ID
    assert all(x == PAD_ID for x in pretrain_field[1:])


# ---------------------------------------------------------------------------
# collate(): tensor shapes and values
# ---------------------------------------------------------------------------

TWO_SENTENCE_DIFFERENT_LENGTHS = """
# sent_id = a
# text = Cats sleep.
1	Cats	cat	NOUN	NNS	Number=Plur	2	nsubj	2:nsubj	_
2	sleep	sleep	VERB	VBZ	Mood=Ind|Tense=Pres	0	root	0:root	_
3	.	.	PUNCT	.	_	2	punct	2:punct	_

# sent_id = b
# text = Dogs run fast.
1	Dogs	dog	NOUN	NNS	Number=Plur	2	nsubj	2:nsubj	_
2	run	run	VERB	VBZ	Mood=Ind|Tense=Pres	0	root	0:root	_
3	fast	fast	ADV	RB	_	2	advmod	2:advmod	_
4	.	.	PUNCT	.	_	2	punct	2:punct	_
"""

def test_collate_tensor_shapes():
    """
    Built in eval mode with sort_during_eval=True, since that is the only
    configuration where batch ordering (and therefore orig_idx) is fully
    deterministic. In train mode, data_to_batches applies its own random
    sort direction and collate()'s internal sort_all applies a second,
    independent one, so orig_idx can legitimately come out as either
    [0, 1] or [1, 0] from one build to the next -- that is not a bug,
    just not something a test should pin down.
    """
    from stanza.models.common.vocab import PAD_ID

    sample = CoNLL.conll2doc(input_str=TWO_SENTENCE_DIFFERENT_LENGTHS)
    args = parser.parse_args(args=["--batch_size", "100", "--shorthand", "en_test"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    vocab = train_batch.vocab
    eval_batch = DataLoader(sample, args['batch_size'], args, None, vocab=vocab,
                             evaluation=True, sort_during_eval=True)
    assert len(eval_batch) == 1
    batch = eval_batch[0]
    (words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats,
     pretrained, lemma, head, deprel, orig_idx, word_orig_idx,
     sentlens, word_lens, text) = batch

    # 2 sentences, longer one (4 words + ROOT = 5) sets the padded width;
    # words/upos/xpos/pretrained/lemma all include the ROOT position
    assert tuple(words.shape) == (2, 5)
    assert tuple(upos.shape) == (2, 5)
    assert tuple(xpos.shape) == (2, 5)
    assert tuple(pretrained.shape) == (2, 5)
    assert tuple(lemma.shape) == (2, 5)

    # head/deprel exclude ROOT, so the padded width is one less
    assert tuple(head.shape) == (2, 4)
    assert tuple(deprel.shape) == (2, 4)

    # batches are sorted longest-first: "Dogs run fast ." (4 words) before
    # "Cats sleep ." (3 words), so the second row is padded
    assert sentlens == [5, 4]
    assert text[0] == ['Dogs', 'run', 'fast', '.']
    assert text[1] == ['Cats', 'sleep', '.']

    # the shorter sentence's tensors should carry PAD_ID past its real length
    assert words[1, 4].item() == PAD_ID
    assert words_mask[1, 4].item() is True
    assert words_mask[0, 4].item() is False

    # sort_during_eval means DepparseBatchSampler has already sorted
    # index_list descending by length before collate() ever sees it, so
    # collate()'s own internal sort_all (also descending) finds nothing
    # left to reorder -- orig_idx here is relative to collate()'s own
    # input order (index_list), not document order, so it comes out as
    # the identity permutation. Document-order unsorting is a *separate*
    # concept, carried on eval_batch.data_orig_idx (verified independently
    # in test_data_orig_idx_unsorts_correctly), not on this per-batch value.
    assert list(orig_idx) == [0, 1]
    assert eval_batch.data_orig_idx == [1, 0]

def test_collate_head_values_exclude_root_offset():
    """head values in the collated tensor should match the raw preprocess() head list."""
    train_batch = _build_train_loader(TWO_SENTENCE_DIFFERENT_LENGTHS, extra_args={'augment_nopunct': 0})
    batch = train_batch[0]
    head = batch[9]
    text = batch[-1]
    # first row is "Dogs run fast .": Dogs->run(2), run->root(0), fast->run(2), .->run(2)
    dogs_row_idx = text.index(['Dogs', 'run', 'fast', '.'])
    assert list(head[dogs_row_idx]) == [2, 0, 2, 2]


# ---------------------------------------------------------------------------
# Vocab handling
# ---------------------------------------------------------------------------

def test_vocab_reused_when_provided():
    """Passing vocab= explicitly should reuse that exact object, not rebuild one."""
    train_batch = _build_train_loader(TWO_WORD_SAMPLE)
    vocab = train_batch.vocab

    sample = CoNLL.conll2doc(input_str=TWO_WORD_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "100", "--shorthand", "en_test"])
    second_batch = DataLoader(sample, args['batch_size'], args, None, vocab=vocab, evaluation=True)
    assert second_batch.vocab is vocab

def test_init_vocab_requires_train_mode():
    """Building a vocab from scratch (vocab=None) in eval mode should raise."""
    sample = CoNLL.conll2doc(input_str=TWO_WORD_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "100", "--shorthand", "en_test"])
    with pytest.raises(AssertionError):
        DataLoader(sample, args['batch_size'], args, None, vocab=None, evaluation=True)


# ---------------------------------------------------------------------------
# sample_train
# ---------------------------------------------------------------------------

TEN_SENTENCE_SAMPLE = "\n".join(
    "# sent_id = s{i}\n# text = word{i}.\n1\tword{i}\tword{i}\tNOUN\tNN\t_\t0\troot\t0:root\t_\n2\t.\t.\tPUNCT\t.\t_\t1\tpunct\t1:punct\t_\n".format(i=i)
    for i in range(10)
)

def test_sample_train_subsets_training_data():
    train_batch = _build_train_loader(TEN_SENTENCE_SAMPLE, extra_args={'sample_train': 0.5})
    assert len(train_batch.dataset) == 5

def test_sample_train_does_not_affect_eval():
    train_batch = _build_train_loader(TEN_SENTENCE_SAMPLE)
    vocab = train_batch.vocab
    sample = CoNLL.conll2doc(input_str=TEN_SENTENCE_SAMPLE)
    args = dict(parser.parse_args(args=["--batch_size", "100", "--shorthand", "en_test"]))
    args['sample_train'] = 0.5
    eval_batch = DataLoader(sample, args['batch_size'], args, None, vocab=vocab, evaluation=True)
    assert len(eval_batch.dataset) == 10


# ---------------------------------------------------------------------------
# DepparseBatchSampler, in isolation
# ---------------------------------------------------------------------------
#
# DepparseBatchSampler only needs an object exposing .lengths, so it can be
# tested without building a real Dataset (vocab, doc, preprocessing, etc).

class _FakeLengthsDataset:
    def __init__(self, lengths):
        self.lengths = lengths

def test_batch_sampler_matches_data_to_batches():
    """DepparseBatchSampler's batching should agree with data_to_batches on the same lengths."""
    fake = _FakeLengthsDataset([1, 2, 3])
    sampler = DepparseBatchSampler(fake, batch_size=5, eval_mode=True,
                                    sort_during_eval=True, min_length_to_batch_separately=None)
    assert len(sampler) == 2
    # sorted descending by length: idx2(len3), idx1(len2), idx0(len1);
    # budget 5 fits idx2+idx1 (3+2=5), idx0 alone in the second batch
    assert sampler.batches == [[2, 1], [0]]
    assert sampler.data_orig_idx == [2, 1, 0]
    # __iter__ should yield the same batches
    assert list(sampler) == [[2, 1], [0]]

def test_batch_sampler_reshuffle_changes_grouping():
    """reshuffle() should recompute batches (train mode reshuffles + reorders every call)."""
    fake = _FakeLengthsDataset([1, 2, 3, 4, 5])
    sampler = DepparseBatchSampler(fake, batch_size=3, eval_mode=False,
                                    sort_during_eval=False, min_length_to_batch_separately=None)
    first = sampler.batches
    seen_different = False
    for _ in range(20):
        sampler.reshuffle()
        if sampler.batches != first:
            seen_different = True
            break
    assert seen_different, "reshuffle() never produced a different batch grouping across 20 tries"
    # every index should always be accounted for exactly once, regardless of grouping
    all_indices = sorted(idx for batch in sampler.batches for idx in batch)
    assert all_indices == [0, 1, 2, 3, 4]

def test_batch_sampler_eval_mode_stable_without_reshuffle():
    """Eval-mode batches should not change just from being iterated repeatedly."""
    fake = _FakeLengthsDataset([1, 2, 3])
    sampler = DepparseBatchSampler(fake, batch_size=5, eval_mode=True,
                                    sort_during_eval=True, min_length_to_batch_separately=None)
    first = list(sampler)
    second = list(sampler)
    assert first == second


# ---------------------------------------------------------------------------
# InfiniteBatch
# ---------------------------------------------------------------------------

class _FakeBatchSet:
    """Minimal stand-in for a DataLoader: supports iteration, reshuffle, set_batch_size."""
    def __init__(self, name, n):
        self.name = name
        self.batch_size = None
        self._order = list(range(n))

    def __iter__(self):
        return iter(["{}{}".format(self.name, i) for i in self._order])

    def reshuffle(self):
        random.shuffle(self._order)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

def test_infinite_batch_single_source_cycles():
    a = _FakeBatchSet("A", 3)
    ib = InfiniteBatch(a)
    seen = [ib.next_batch() for _ in range(9)]
    # with one source and no weights, every draw comes from it; over 9 draws
    # (3 epochs of a 3-item source) every item should appear at least once
    assert set(x[0] for x in seen) == {'A'}

def test_infinite_batch_weighted_mixing_skews_toward_higher_weight():
    random.seed(12345)
    a = _FakeBatchSet("A", 3)
    b = _FakeBatchSet("B", 3)
    ib = InfiniteBatch(a, b, weights=[1.0, 9.0])
    counts = Counter()
    for _ in range(2000):
        counts[ib.next_batch()[0]] += 1
    # expected ~10% A / 90% B; allow a generous tolerance to avoid flakiness
    assert counts['B'] > counts['A'] * 3, (
        f"expected source B (weight 9) to be drawn far more often than A (weight 1), got {counts}"
    )

def test_infinite_batch_set_batch_size_propagates_to_all_sources():
    a = _FakeBatchSet("A", 3)
    b = _FakeBatchSet("B", 3)
    ib = InfiniteBatch(a, b)
    ib.set_batch_size(7)
    assert a.batch_size == 7
    assert b.batch_size == 7

def test_infinite_batch_reshuffle_resets_iterators():
    a = _FakeBatchSet("A", 3)
    ib = InfiniteBatch(a)
    # exhaust part of the current iterator
    ib.next_batch()
    ib.reshuffle()
    # after an explicit reshuffle, a fresh full cycle should be available
    # again without error
    seen = [ib.next_batch() for _ in range(3)]
    assert len(seen) == 3


# ---------------------------------------------------------------------------
# DataLoader.__getitem__ error handling
# ---------------------------------------------------------------------------

def test_dataloader_getitem_rejects_non_int_key():
    train_batch = _build_train_loader(THREE_SENTENCE_SAMPLE)
    with pytest.raises(TypeError):
        train_batch["0"]

def test_dataloader_getitem_rejects_out_of_range_key():
    train_batch = _build_train_loader(THREE_SENTENCE_SAMPLE, batch_size=100)
    assert len(train_batch) == 1
    with pytest.raises(IndexError):
        train_batch[1]
    with pytest.raises(IndexError):
        train_batch[-1]


# ---------------------------------------------------------------------------
# torch DataLoader-backed eval iteration
# ---------------------------------------------------------------------------
#
# Eval-mode DataLoader.__iter__ is now backed by a real
# torch.utils.data.DataLoader (built from the same sampler and collate
# function used everywhere else), matching the mechanism
# stanza.models.pos.data.Dataset.to_loader already uses. This never
# applies any augmentation -- it is purely an alternate plumbing path for
# the same deterministic eval batches, potentially allowing num_workers>0
# to prefetch/collate in a separate process. Train mode is unaffected:
# it keeps the previous manual loop, since InfiniteBatch's
# reshuffle-on-exhaustion and weighted multi-source mixing aren't
# something an ordinary torch DataLoader supports directly.

def _build_eval_loader(sample_str, batch_size=5, sort_during_eval=True):
    sample = CoNLL.conll2doc(input_str=sample_str)
    args = parser.parse_args(args=["--batch_size", str(batch_size), "--shorthand", "en_test"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    vocab = train_batch.vocab
    return DataLoader(sample, args['batch_size'], args, None, vocab=vocab,
                       evaluation=True, sort_during_eval=sort_during_eval)

def test_eval_iter_delegates_to_dataset_to_loader():
    """
    DataLoader.__iter__ must delegate to Dataset.to_loader() in eval mode,
    not duplicate the TorchDataLoader construction inline -- to_loader()
    is meant to be the single place that wrapping happens, used both here
    and by any standalone caller. Patching TorchDataLoader itself would
    not distinguish an inline call from a delegated one, since both paths
    would reference the same name; patching Dataset.to_loader directly
    does.
    """
    eval_batch = _build_eval_loader(THREE_SENTENCE_SAMPLE)

    call_count = [0]
    real_to_loader = Dataset.to_loader
    def counting_to_loader(self, *args, **kwargs):
        call_count[0] += 1
        return real_to_loader(self, *args, **kwargs)

    Dataset.to_loader = counting_to_loader
    try:
        list(eval_batch)
    finally:
        Dataset.to_loader = real_to_loader
    assert call_count[0] == 1, "expected eval-mode iteration to call Dataset.to_loader() exactly once"

    train_batch = _build_train_loader(THREE_SENTENCE_SAMPLE)
    call_count[0] = 0
    Dataset.to_loader = counting_to_loader
    try:
        list(train_batch)
    finally:
        Dataset.to_loader = real_to_loader
    assert call_count[0] == 0, "train-mode iteration should not call Dataset.to_loader() at all"

def test_eval_iter_matches_direct_getitem_indexing():
    """
    Iterating an eval DataLoader (now torch-DataLoader-backed) must
    produce exactly the same batches, in the same order, as indexing it
    directly (which still uses the original manual per-batch fetch).
    """
    import torch as _torch
    eval_batch = _build_eval_loader(TWO_SENTENCE_DIFFERENT_LENGTHS)

    direct_batches = [eval_batch[i] for i in range(len(eval_batch))]
    iter_batches = list(eval_batch)

    assert len(direct_batches) == len(iter_batches)
    for direct, itd in zip(direct_batches, iter_batches):
        for d_val, i_val in zip(direct, itd):
            if _torch.is_tensor(d_val):
                assert _torch.equal(d_val, i_val)
            else:
                assert d_val == i_val

def test_train_iter_still_uses_manual_loop():
    """
    Train-mode DataLoader.__iter__ must NOT go through the torch
    DataLoader path -- only eval mode should. Confirmed by checking that
    repeated iteration without an explicit reshuffle() keeps returning the
    same batch grouping (the manual loop just re-reads self.sampler.batches
    each time; a torch DataLoader with a fresh RandomSampler would not
    give this guarantee, though DepparseBatchSampler itself doesn't
    reshuffle on its own either way -- the key check is that .eval is
    consulted and behaves identically to before this change).
    """
    train_batch = _build_train_loader(THREE_SENTENCE_SAMPLE, batch_size=100, extra_args={'augment_nopunct': 0})
    assert train_batch.eval is False
    first_pass = [batch[-1] for batch in train_batch]
    second_pass = [batch[-1] for batch in train_batch]
    assert first_pass == second_pass

def test_dataset_to_loader_standalone():
    """
    Dataset.to_loader() should work independently of the outer DataLoader
    wrapper, mirroring stanza.models.pos.data.Dataset.to_loader, and
    return a real torch.utils.data.DataLoader.
    """
    import torch
    sample = CoNLL.conll2doc(input_str=THREE_SENTENCE_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "5", "--shorthand", "en_test"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    vocab = train_batch.vocab

    dataset = Dataset(sample, args, None, vocab=vocab, evaluation=True)
    loader = dataset.to_loader(batch_size=5, sort_during_eval=True)
    assert isinstance(loader, torch.utils.data.DataLoader)

    batches = list(loader)
    all_texts = [sent for batch in batches for sent in batch[-1]]
    assert sorted(all_texts) == sorted([
        ['Short', 'one', '.'],
        ['This', 'is', 'a', 'much', 'longer', 'sentence', 'with', 'many', 'more', 'words', 'in', 'it', '.'],
        ['Medium', 'length', 'one', 'here', '.'],
    ])

def test_dataset_to_loader_reuses_provided_sampler():
    """
    Passing sampler= should reuse that exact sampler's batches rather
    than recomputing a fresh one from batch_size -- this is how
    DataLoader.__iter__ avoids redundant batch computation.
    """
    sample = CoNLL.conll2doc(input_str=THREE_SENTENCE_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "5", "--shorthand", "en_test"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    vocab = train_batch.vocab

    dataset = Dataset(sample, args, None, vocab=vocab, evaluation=True)
    sampler = DepparseBatchSampler(dataset, batch_size=5, eval_mode=True,
                                    sort_during_eval=True, min_length_to_batch_separately=None)
    loader = dataset.to_loader(sampler=sampler)
    assert len(list(loader)) == len(sampler.batches)

def test_dataset_to_loader_requires_batch_size_or_sampler():
    sample = CoNLL.conll2doc(input_str=THREE_SENTENCE_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "5", "--shorthand", "en_test"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    vocab = train_batch.vocab
    dataset = Dataset(sample, args, None, vocab=vocab, evaluation=True)
    with pytest.raises(ValueError):
        dataset.to_loader()


# ---------------------------------------------------------------------------
# Dynamic nopunct augmentation
# ---------------------------------------------------------------------------
#
# Some UD treebanks have every training sentence end in PUNCT, which
# teaches the parser to always expect a final punctuation mark and
# misparse sentences that lack one. This used to be compensated for by
# permanently duplicating a fraction of sentences (with their final PUNCT
# removed) into the training set -- bloating it and skewing its
# length/content distribution. It is now applied dynamically, per
# sentence, inside Dataset.__getitem__, at the same target ratio computed
# from the corpus (via the same get_augment_ratio machinery as before).

NOPUNCT_ELIGIBLE_SAMPLE = """
# sent_id = a
# text = Cats sleep.
1	Cats	cat	NOUN	NNS	Number=Plur	2	nsubj	2:nsubj	_
2	sleep	sleep	VERB	VBZ	Mood=Ind|Tense=Pres	0	root	0:root	_
3	.	.	PUNCT	.	_	2	punct	2:punct	_
"""

# 'weird' depends on the final PUNCT (head=3), so removing it would leave
# a dangling head reference -- must never be considered eligible
NOPUNCT_INELIGIBLE_DEPENDENCY_SAMPLE = """
# sent_id = a
# text = Cats weird .
1	Cats	cat	NOUN	NNS	_	3	dep	3:dep	_
2	weird	weird	ADJ	JJ	_	3	dep	3:dep	_
3	.	.	PUNCT	.	_	0	root	0:root	_
"""

NOPUNCT_NOT_PUNCT_ENDING_SAMPLE = """
# sent_id = a
# text = Cats sleep
1	Cats	cat	NOUN	NNS	_	2	nsubj	2:nsubj	_
2	sleep	sleep	VERB	VBZ	_	0	root	0:root	_
"""

NOPUNCT_SINGLE_WORD_SAMPLE = """
# sent_id = a
# text = .
1	.	.	PUNCT	.	_	0	root	0:root	_
"""

TEN_PUNCT_ENDING_SENTENCES = "\n".join(
    "# sent_id = s{i}\n# text = word{i} .\n1\tword{i}\tword{i}\tNOUN\tNN\t_\t0\troot\t0:root\t_\n2\t.\t.\tPUNCT\t.\t_\t1\tpunct\t1:punct\t_\n".format(i=i)
    for i in range(10)
)


def test_record_ends_with_punct():
    train_batch = _build_train_loader(NOPUNCT_ELIGIBLE_SAMPLE, extra_args={'augment_nopunct': 1.0})
    record = train_batch.dataset.data[0]
    assert record_ends_with_punct(record, train_batch.dataset.punct_id) is True

def test_record_ends_with_punct_false_when_not_punct():
    train_batch = _build_train_loader(NOPUNCT_NOT_PUNCT_ENDING_SAMPLE, extra_args={'augment_nopunct': 1.0})
    record = train_batch.dataset.data[0]
    assert record_ends_with_punct(record, train_batch.dataset.punct_id) is False

def test_can_augment_nopunct_eligible():
    train_batch = _build_train_loader(NOPUNCT_ELIGIBLE_SAMPLE, extra_args={'augment_nopunct': 1.0})
    record = train_batch.dataset.data[0]
    assert record_can_augment_nopunct(record, train_batch.dataset.punct_id) is True

def test_can_augment_nopunct_false_when_word_depends_on_final_punct():
    train_batch = _build_train_loader(NOPUNCT_INELIGIBLE_DEPENDENCY_SAMPLE, extra_args={'augment_nopunct': 1.0})
    record = train_batch.dataset.data[0]
    assert record_can_augment_nopunct(record, train_batch.dataset.punct_id) is False

def test_can_augment_nopunct_false_when_not_punct_ending():
    train_batch = _build_train_loader(NOPUNCT_NOT_PUNCT_ENDING_SAMPLE, extra_args={'augment_nopunct': 1.0})
    record = train_batch.dataset.data[0]
    assert record_can_augment_nopunct(record, train_batch.dataset.punct_id) is False

def test_can_augment_nopunct_false_for_single_word_sentence():
    """A lone PUNCT token: len(sentence) > 1 guard (mirroring the original augment_punct)."""
    train_batch = _build_train_loader(NOPUNCT_SINGLE_WORD_SAMPLE, extra_args={'augment_nopunct': 1.0})
    record = train_batch.dataset.data[0]
    assert record_can_augment_nopunct(record, train_batch.dataset.punct_id) is False


def test_augment_nopunct_ratio_explicit():
    train_batch = _build_train_loader(NOPUNCT_ELIGIBLE_SAMPLE, extra_args={'augment_nopunct': 0.37})
    assert train_batch.dataset.augment_nopunct_ratio == 0.37

def test_augment_nopunct_ratio_zero_disables():
    train_batch = _build_train_loader(NOPUNCT_ELIGIBLE_SAMPLE, extra_args={'augment_nopunct': 0})
    assert train_batch.dataset.augment_nopunct_ratio == 0.0

def test_augment_nopunct_ratio_auto_matches_get_augment_ratio():
    """
    Default (augment_nopunct=None) should auto-compute the same ratio
    get_augment_ratio would give directly: 10 sentences, all ending in
    PUNCT and all eligible, desired_ratio=0.1 -> 10*0.1 / 10 = 0.1.
    """
    train_batch = _build_train_loader(TEN_PUNCT_ENDING_SENTENCES)
    assert train_batch.dataset.augment_nopunct_ratio == pytest.approx(0.1)

def test_augment_nopunct_ratio_zero_when_no_punct_in_corpus():
    """If the vocab never saw a PUNCT tag at all, the ratio must be 0, not accidentally match UNK."""
    train_batch = _build_train_loader(NOPUNCT_NOT_PUNCT_ENDING_SAMPLE, extra_args={'augment_nopunct': 1.0})
    assert train_batch.dataset.punct_id is None
    assert train_batch.dataset.augment_nopunct_ratio == 0.0

def test_augment_nopunct_ratio_zero_in_eval_mode():
    """Eval mode must never augment, regardless of what augment_nopunct is set to."""
    sample = CoNLL.conll2doc(input_str=NOPUNCT_ELIGIBLE_SAMPLE)
    args = parser.parse_args(args=["--batch_size", "100", "--shorthand", "en_test", "--augment_nopunct", "1.0"])
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    vocab = train_batch.vocab
    eval_batch = DataLoader(sample, args['batch_size'], args, None, vocab=vocab, evaluation=True)
    assert eval_batch.dataset.augment_nopunct_ratio == 0.0


def test_getitem_always_truncates_eligible_sentence_at_ratio_one():
    train_batch = _build_train_loader(NOPUNCT_ELIGIBLE_SAMPLE, extra_args={'augment_nopunct': 1.0})
    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched[9] == ['Cats', 'sleep']
        assert fetched[7] == [2, 0]

def test_getitem_truncates_all_ten_fields_consistently():
    """Every field (ROOT-prepended or not) should lose exactly its last entry."""
    train_batch = _build_train_loader(NOPUNCT_ELIGIBLE_SAMPLE, extra_args={'augment_nopunct': 1.0})
    original = train_batch.dataset.data[0]
    augmented = train_batch.dataset[0]
    assert len(augmented) == len(original) == 10
    for orig_field, aug_field in zip(original, augmented):
        assert aug_field == orig_field[:-1]

def test_getitem_never_truncates_ineligible_sentence_even_at_ratio_one():
    train_batch = _build_train_loader(NOPUNCT_INELIGIBLE_DEPENDENCY_SAMPLE, extra_args={'augment_nopunct': 1.0})
    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched[9] == ['Cats', 'weird', '.']

def test_getitem_never_truncates_when_ratio_zero():
    train_batch = _build_train_loader(NOPUNCT_ELIGIBLE_SAMPLE, extra_args={'augment_nopunct': 0})
    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched[9] == ['Cats', 'sleep', '.']

def test_getitem_dynamic_mix_across_repeated_fetches():
    """
    The whole point of moving this into __getitem__: the SAME sentence
    should come back augmented some of the time and un-augmented other
    times, rather than a fixed decision made once when the dataset was
    built (as the old duplicate-into-the-corpus approach effectively did).
    """
    train_batch = _build_train_loader(NOPUNCT_ELIGIBLE_SAMPLE, extra_args={'augment_nopunct': 0.5})
    counts = Counter()
    for _ in range(2000):
        fetched = train_batch.dataset[0]
        counts['augmented' if len(fetched[9]) == 2 else 'original'] += 1
    # expect roughly a 50/50 split; generous tolerance to avoid flakiness
    assert 700 < counts['augmented'] < 1300, f"expected roughly half augmented, got {counts}"
    assert 700 < counts['original'] < 1300, f"expected roughly half original, got {counts}"


# ---------------------------------------------------------------------------
# Dynamic leading-inverted-punct drop (¿ / ¡)
# ---------------------------------------------------------------------------
#
# Some UD treebanks (Spanish, Catalan) have every training sentence begin
# with an inverted question or exclamation mark, which the model never
# learns to do without. This mirrors augment_initial_punct in
# prepare_tokenizer_treebank.py (¿ only there), applied dynamically per
# sentence inside Dataset.__getitem__ instead of by duplicating sentences
# at dataset-preparation time, and extended to ¡ as well. Unlike the
# nopunct augmentation (which only ever drops the last word), this must
# also renumber every remaining word's head position down by one, since
# removing the FIRST word shifts every later position back.

LEADING_QUESTION_SAMPLE = """
# sent_id = a
# text = ¿Cómo estás?
1	¿	¿	PUNCT	_	_	3	punct	3:punct	_
2	Cómo	cómo	PRON	_	_	3	advmod	3:advmod	_
3	estás	estar	VERB	_	_	0	root	0:root	_
4	?	?	PUNCT	_	_	3	punct	3:punct	_
"""

LEADING_QUESTION_NATURAL_SAMPLE = """
# sent_id = a
# text = Cómo estás?
1	Cómo	cómo	PRON	_	_	2	advmod	2:advmod	_
2	estás	estar	VERB	_	_	0	root	0:root	_
3	?	?	PUNCT	_	_	2	punct	2:punct	_
"""

LEADING_EXCLAMATION_SAMPLE = """
# sent_id = a
# text = ¡Qué bien!
1	¡	¡	PUNCT	_	_	3	punct	3:punct	_
2	Qué	qué	DET	_	_	3	det	3:det	_
3	bien	bien	ADV	_	_	0	root	0:root	_
4	!	!	PUNCT	_	_	3	punct	3:punct	_
"""

# something (the ADJ/NOUN) depends directly on the leading ¿ (head=1) --
# must never be eligible, since removing it would leave a dangling head
LEADING_PUNCT_DEPENDENT_SAMPLE = """
# sent_id = a
# text = ¿weird thing?
1	¿	¿	PUNCT	_	_	0	root	0:root	_
2	weird	weird	ADJ	_	_	1	dep	1:dep	_
3	thing	thing	NOUN	_	_	1	dep	1:dep	_
4	?	?	PUNCT	_	_	1	punct	1:punct	_
"""

# ¿ appears twice -- must never be touched
LEADING_PUNCT_TWICE_SAMPLE = """
# sent_id = a
# text = ¿Cómo ¿estás?
1	¿	¿	PUNCT	_	_	3	punct	3:punct	_
2	Cómo	cómo	PRON	_	_	3	advmod	3:advmod	_
3	estás	estar	VERB	_	_	0	root	0:root	_
4	¿	¿	PUNCT	_	_	3	punct	3:punct	_
5	?	?	PUNCT	_	_	3	punct	3:punct	_
"""

LEADING_PUNCT_NONE_SAMPLE = """
# sent_id = a
# text = Cómo estás?
1	Cómo	cómo	PRON	_	_	2	advmod	2:advmod	_
2	estás	estar	VERB	_	_	0	root	0:root	_
3	?	?	PUNCT	_	_	2	punct	2:punct	_
"""

# a leading ¿ plus a DIFFERENT mark (¡) elsewhere -- must never be touched,
# even though neither mark is individually repeated
LEADING_MIXED_MARKS_SAMPLE = """
# sent_id = a
# text = ¿Dijo "¡hola!"?
1	¿	¿	PUNCT	_	_	2	punct	2:punct	_
2	Dijo	decir	VERB	_	_	0	root	0:root	_
3	"	"	PUNCT	_	_	5	punct	5:punct	_
4	¡	¡	PUNCT	_	_	5	punct	5:punct	_
5	hola	hola	INTJ	_	_	2	obj	2:obj	_
6	!	!	PUNCT	_	_	5	punct	5:punct	_
7	"	"	PUNCT	_	_	5	punct	5:punct	_
8	?	?	PUNCT	_	_	2	punct	2:punct	_
"""



def _build_es_train_loader(sample_str, extra_args=None):
    sample = CoNLL.conll2doc(input_str=sample_str)
    args = dict(parser.parse_args(args=["--batch_size", "100", "--shorthand", "es_test"]))
    args['augment_nopunct'] = 0
    if extra_args:
        args.update(extra_args)
    return DataLoader(sample, args['batch_size'], args, None, evaluation=False)


def test_record_starts_with_mark_question():
    train_batch = _build_es_train_loader(LEADING_QUESTION_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    record = train_batch.dataset.data[0]
    assert record_starts_with_mark(record) is True

def test_record_starts_with_mark_exclamation():
    train_batch = _build_es_train_loader(LEADING_EXCLAMATION_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    record = train_batch.dataset.data[0]
    assert record_starts_with_mark(record) is True

def test_record_starts_with_mark_false_when_absent():
    train_batch = _build_es_train_loader(LEADING_PUNCT_NONE_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    record = train_batch.dataset.data[0]
    assert record_starts_with_mark(record) is False

def test_record_can_drop_initial_mark_false_when_word_depends_on_it():
    train_batch = _build_es_train_loader(LEADING_PUNCT_DEPENDENT_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    record = train_batch.dataset.data[0]
    assert record_starts_with_mark(record) is True
    assert record_can_drop_initial_mark(record) is False

def test_record_can_drop_initial_mark_false_when_mark_appears_twice():
    train_batch = _build_es_train_loader(LEADING_PUNCT_TWICE_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    record = train_batch.dataset.data[0]
    assert record_can_drop_initial_mark(record) is False


def test_drop_initial_punct_eligible():
    train_batch = _build_es_train_loader(LEADING_QUESTION_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    assert train_batch.dataset.drop_initial_punct_eligible is True

def test_drop_initial_punct_ineligible_when_absent():
    train_batch = _build_es_train_loader(LEADING_PUNCT_NONE_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    assert train_batch.dataset.drop_initial_punct_eligible is False
    assert train_batch.dataset.drop_initial_punct_ratio == 0.0

def test_drop_initial_punct_ratio_zero_disables():
    train_batch = _build_es_train_loader(LEADING_QUESTION_SAMPLE, extra_args={'drop_initial_punct_prob': 0})
    assert train_batch.dataset.drop_initial_punct_ratio == 0.0

def test_drop_initial_punct_ratio_explicit():
    train_batch = _build_es_train_loader(LEADING_QUESTION_SAMPLE, extra_args={'drop_initial_punct_prob': 0.37})
    assert train_batch.dataset.drop_initial_punct_ratio == 0.37

def test_drop_initial_punct_disabled_in_eval_mode():
    """Eval mode must never drop the leading mark, regardless of drop_initial_punct_prob."""
    sample = CoNLL.conll2doc(input_str=LEADING_QUESTION_SAMPLE)
    args = dict(parser.parse_args(args=["--batch_size", "100", "--shorthand", "es_test"]))
    args['augment_nopunct'] = 0
    args['drop_initial_punct_prob'] = 1.0
    train_batch = DataLoader(sample, args['batch_size'], args, None, evaluation=False)
    eval_batch = DataLoader(sample, args['batch_size'], args, None, vocab=train_batch.vocab, evaluation=True)
    assert eval_batch.dataset.drop_initial_punct_eligible is False
    assert eval_batch.dataset.drop_initial_punct_ratio == 0.0


def test_getitem_always_drops_leading_question_mark():
    train_batch = _build_es_train_loader(LEADING_QUESTION_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched[9] == ['Cómo', 'estás', '?']

def test_getitem_always_drops_leading_exclamation_mark():
    train_batch = _build_es_train_loader(LEADING_EXCLAMATION_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched[9] == ['Qué', 'bien', '!']

def test_getitem_renumbers_head_correctly():
    """
    Dropping the leading mark must renumber every remaining word's head
    down by one -- verified against a naturally-written sentence (with no
    leading ¿ to begin with) built from the same vocab, so the augmented
    result should exactly match the gold record in every field.
    """
    train_batch = _build_es_train_loader(LEADING_QUESTION_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    vocab = train_batch.vocab

    natural_sample = CoNLL.conll2doc(input_str=LEADING_QUESTION_NATURAL_SAMPLE)
    natural_args = dict(parser.parse_args(args=["--batch_size", "100", "--shorthand", "es_test"]))
    natural_args['augment_nopunct'] = 0
    natural_batch = DataLoader(natural_sample, natural_args['batch_size'], natural_args, None,
                                vocab=vocab, evaluation=True)
    gold_record = natural_batch.dataset.data[0]

    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched == gold_record

def test_getitem_never_drops_when_word_depends_on_leading_mark():
    train_batch = _build_es_train_loader(LEADING_PUNCT_DEPENDENT_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched[9] == ['¿', 'weird', 'thing', '?']

def test_getitem_never_drops_when_mark_appears_twice():
    train_batch = _build_es_train_loader(LEADING_PUNCT_TWICE_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched[9] == ['¿', 'Cómo', 'estás', '¿', '?']

def test_getitem_never_drops_when_marks_are_mixed():
    """
    A leading ¿ plus a DIFFERENT mark (¡) elsewhere in the sentence must
    also be blocked, not just a repeat of the SAME leading mark.
    '¿Dijo "¡hola!"?' has one ¿ and one ¡ -- neither mark is individually
    repeated, but there are still two candidate marks.
    """
    train_batch = _build_es_train_loader(LEADING_MIXED_MARKS_SAMPLE, extra_args={'drop_initial_punct_prob': 1.0})
    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched[9][0] == '¿'

def test_getitem_never_drops_when_ratio_zero():
    train_batch = _build_es_train_loader(LEADING_QUESTION_SAMPLE, extra_args={'drop_initial_punct_prob': 0})
    for _ in range(20):
        fetched = train_batch.dataset[0]
        assert fetched[9] == ['¿', 'Cómo', 'estás', '?']

def test_getitem_dynamic_mix_for_leading_punct():
    """Same sentence should come back with and without the leading mark across repeated fetches."""
    train_batch = _build_es_train_loader(LEADING_QUESTION_SAMPLE, extra_args={'drop_initial_punct_prob': 0.5})
    counts = Counter()
    for _ in range(2000):
        fetched = train_batch.dataset[0]
        counts['dropped' if fetched[9][0] != '¿' else 'kept'] += 1
    assert 700 < counts['dropped'] < 1300, f"expected roughly half dropped, got {counts}"
    assert 700 < counts['kept'] < 1300, f"expected roughly half kept, got {counts}"


if __name__ == '__main__':
    test_data_to_batches()

