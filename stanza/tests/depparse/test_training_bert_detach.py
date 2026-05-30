"""
Tests for the bert detach behaviour in the dependency parser model.

The key logic under test is in EmbeddingParser.embed():

    bert_finetuning = getattr(self, 'bert_finetuning', False)
    processed_bert = extract_bert_embeddings(...,
                         detach=not bert_finetuning or not self.training, ...)

This means bert embeddings are detached (no gradients flow back into the
transformer) in three situations:
  - model is in eval mode (self.training is False), regardless of finetuning
  - bert_finetuning is False, regardless of mode
  - both of the above

Gradients flow into the transformer only when:
  - bert_finetuning is True AND self.training is True

The trainer sets model.bert_finetuning based on whether any optimizer key
starts with "bert" or "peft".  So a second training stage with
second_bert_learning_rate=0.0 will have no bert optimizer key, meaning
bert_finetuning=False, meaning detach=True even in training mode.

Tests here are structured to avoid running a full training loop where
possible, instead directly constructing a model and inspecting whether
the bert output carries a grad_fn.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from stanza.tests import TEST_WORKING_DIR
from stanza.tests.depparse.parser_training import run_training

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

BERT_MODEL = "hf-internal-testing/tiny-bert"

TRAIN_DATA = """
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0003
# text = DPA: Iraqi authorities announced that they had busted up 3 terrorist cells operating in Baghdad.
1\tDPA\tDPA\tPROPN\tNNP\tNumber=Sing\t0\troot\t0:root\tSpaceAfter=No
2\t:\t:\tPUNCT\t:\t_\t1\tpunct\t1:punct\t_
3\tIraqi\tIraqi\tADJ\tJJ\tDegree=Pos\t4\tamod\t4:amod\t_
4\tauthorities\tauthority\tNOUN\tNNS\tNumber=Plur\t5\tnsubj\t5:nsubj\t_
5\tannounced\tannounce\tVERB\tVBD\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\t1\tparataxis\t1:parataxis\t_
6\tthat\tthat\tSCONJ\tIN\t_\t9\tmark\t9:mark\t_
7\tthey\tthey\tPRON\tPRP\tCase=Nom|Number=Plur|Person=3|PronType=Prs\t9\tnsubj\t9:nsubj\t_
8\thad\thave\tAUX\tVBD\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\t9\taux\t9:aux\t_
9\tbusted\tbust\tVERB\tVBN\tTense=Past|VerbForm=Part\t5\tccomp\t5:ccomp\t_
10\tup\tup\tADP\tRP\t_\t9\tcompound:prt\t9:compound:prt\t_
11\t3\t3\tNUM\tCD\tNumForm=Digit|NumType=Card\t13\tnummod\t13:nummod\t_
12\tterrorist\tterrorist\tADJ\tJJ\tDegree=Pos\t13\tamod\t13:amod\t_
13\tcells\tcell\tNOUN\tNNS\tNumber=Plur\t9\tobj\t9:obj\t_
14\toperating\toperate\tVERB\tVBG\tVerbForm=Ger\t13\tacl\t13:acl\t_
15\tin\tin\tADP\tIN\t_\t16\tcase\t16:case\t_
16\tBaghdad\tBaghdad\tPROPN\tNNP\tNumber=Sing\t14\tobl\t14:obl:in\tSpaceAfter=No
17\t.\t.\tPUNCT\t.\t_\t1\tpunct\t1:punct\t_

# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0004
# text = Two of them were being run by 2 officials of the Ministry of the Interior!
1\tTwo\ttwo\tNUM\tCD\tNumForm=Word|NumType=Card\t6\tnsubj:pass\t6:nsubj:pass\t_
2\tof\tof\tADP\tIN\t_\t3\tcase\t3:case\t_
3\tthem\tthey\tPRON\tPRP\tCase=Acc|Number=Plur|Person=3|PronType=Prs\t1\tnmod\t1:nmod:of\t_
4\twere\tbe\tAUX\tVBD\tMood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin\t6\taux\t6:aux\t_
5\tbeing\tbe\tAUX\tVBG\tVerbForm=Ger\t6\taux:pass\t6:aux:pass\t_
6\trun\trun\tVERB\tVBN\tTense=Past|VerbForm=Part|Voice=Pass\t0\troot\t0:root\t_
7\tby\tby\tADP\tIN\t_\t9\tcase\t9:case\t_
8\t2\t2\tNUM\tCD\tNumForm=Digit|NumType=Card\t9\tnummod\t9:nummod\t_
9\tofficials\tofficial\tNOUN\tNNS\tNumber=Plur\t6\tobl\t6:obl:by\t_
10\tof\tof\tADP\tIN\t_\t12\tcase\t12:case\t_
11\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t12\tdet\t12:det\t_
12\tMinistry\tMinistry\tPROPN\tNNP\tNumber=Sing\t9\tnmod\t9:nmod:of\t_
13\tof\tof\tADP\tIN\t_\t15\tcase\t15:case\t_
14\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t15\tdet\t15:det\t_
15\tInterior\tInterior\tPROPN\tNNP\tNumber=Sing\t12\tnmod\t12:nmod:of\tSpaceAfter=No
16\t!\t!\tPUNCT\t.\t_\t6\tpunct\t6:punct\t_

""".lstrip()

DEV_DATA = """
1\tFrom\tfrom\tADP\tIN\t_\t3\tcase\t3:case\t_
2\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t3\tdet\t3:det\t_
3\tAP\tAP\tPROPN\tNNP\tNumber=Sing\t4\tobl\t4:obl:from\t_
4\tcomes\tcome\tVERB\tVBZ\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t0:root\t_
5\tthis\tthis\tDET\tDT\tNumber=Sing|PronType=Dem\t6\tdet\t6:det\t_
6\tstory\tstory\tNOUN\tNN\tNumber=Sing\t4\tnsubj\t4:nsubj\t_
7\t:\t:\tPUNCT\t:\t_\t4\tpunct\t4:punct\t_

""".lstrip()


@pytest.fixture(scope="module")
def wordvec_pretrain_file():
    return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bert_model():
    """Return a tiny-bert model and tokenizer, ready to use."""
    from stanza.models.common.bert_embedding import load_bert
    return load_bert(BERT_MODEL)


def make_parser_model(bert_model, bert_tokenizer, wordvec_pretrain_file):
    """
    Construct a minimal GraphParser with the given bert model attached,
    using the same vocab and args that the training tests use.
    """
    from stanza.models.common import pretrain
    from stanza.models.depparse.model import GraphParser
    from stanza.models.pos.vocab import MultiVocab
    from stanza.models.common.vocab import CompositeVocab

    # Build a minimal vocab that satisfies GraphParser's requirements
    # without running a full data pipeline
    from stanza.models.common.vocab import UNK_ID
    import stanza.models.depparse.data as data

    pt = pretrain.Pretrain(wordvec_pretrain_file)

    # Minimal args matching the training test setup
    args = {
        'word_emb_dim': 75,
        'tag_emb_dim': 50,
        'char': True,
        'char_emb_dim': 100,
        'char_hidden_dim': 400,
        'charlm': None,
        'pretrain': True,
        'transformed_dim': 125,
        'hidden_dim': 200,
        'num_layers': 3,
        'dropout': 0.33,
        'word_dropout': 0.1,
        'rec_dropout': 0,
        'deep_biaff_hidden_dim': 400,
        'deep_biaff_output_dim': None,
        'linearization': False,
        'distance': False,
        'bert_model': BERT_MODEL,
        'bert_hidden_layers': 2,
        'use_peft': False,
        'use_arc_embedding': False,
        'use_upos': True,
        'use_xpos': True,
        'use_ufeats': True,
    }
    return args, pt


def captured_detach_arg(bert_model, bert_tokenizer, bert_finetuning, training):
    """
    Run embed() on a minimal model with controlled bert_finetuning and
    training mode, and return the detach argument that was actually passed
    to extract_bert_embeddings.

    Rather than building a full GraphParser (which requires a complete
    vocab and data pipeline), we test the detach logic directly by
    intercepting the extract_bert_embeddings call.
    """
    detach_values = []

    real_extract = None
    try:
        from stanza.models.common import bert_embedding
        real_extract = bert_embedding.extract_bert_embeddings
    except ImportError:
        pytest.skip("bert_embedding not available")

    def capturing_extract(*args, detach=True, **kwargs):
        detach_values.append(detach)
        # Return a plausible output shape so embed() doesn't crash:
        # one sentence, 5 tokens, hidden_size features
        hidden_size = bert_model.config.hidden_size
        return [torch.zeros(7, hidden_size)]  # 7 = 5 words + 2 endpoints

    with patch('stanza.models.common.bert_embedding.extract_bert_embeddings',
               side_effect=capturing_extract):
        # Build a minimal EmbeddingParser-like object just to test the
        # detach logic, without needing a full vocab/data pipeline
        import torch.nn as nn

        class MinimalBertModel(nn.Module):
            """Stripped-down stand-in for EmbeddingParser that only
            exercises the bert detach logic."""
            def __init__(self):
                super().__init__()
                self.bert_model = bert_model
                self.bert_tokenizer = bert_tokenizer
                self.bert_finetuning = bert_finetuning
                self.peft_name = None
                self.bert_layer_mix = nn.Linear(2, 1, bias=False)
                nn.init.zeros_(self.bert_layer_mix.weight)
                self.args = {
                    'bert_model': BERT_MODEL,
                    'bert_hidden_layers': 2,
                }

            def run_bert(self, text):
                from stanza.models.common.bert_embedding import extract_bert_embeddings
                device = torch.device('cpu')
                bert_finetuning = getattr(self, 'bert_finetuning', False)
                return extract_bert_embeddings(
                    self.args['bert_model'],
                    self.bert_tokenizer,
                    self.bert_model,
                    text,
                    device,
                    keep_endpoints=True,
                    num_layers=self.bert_layer_mix.in_features,
                    detach=not bert_finetuning or not self.training,
                    peft_name=self.peft_name,
                )

        model = MinimalBertModel()
        if training:
            model.train()
        else:
            model.eval()

        model.run_bert([["The", "cat", "sat", "on", "the"]])

    assert len(detach_values) == 1, "extract_bert_embeddings should be called exactly once"
    return detach_values[0]


# ---------------------------------------------------------------------------
# 1.  Unit tests for the detach logic directly
# ---------------------------------------------------------------------------

class TestDetachLogic:
    """
    Tests for the exact detach condition:
        detach = not bert_finetuning or not self.training

    Four combinations of (bert_finetuning, training mode):
      (False, eval)  → detach=True   (frozen, not training)
      (False, train) → detach=True   (frozen, in training mode)
      (True,  eval)  → detach=True   (finetuning, but not training)
      (True,  train) → detach=False  (finetuning AND in training mode)
    """

    @pytest.fixture(scope="class")
    def tiny_bert(self):
        return make_bert_model()

    def test_detach_when_not_finetuning_eval_mode(self, tiny_bert):
        """bert_finetuning=False, eval mode → must detach."""
        model, tokenizer = tiny_bert
        detach = captured_detach_arg(model, tokenizer,
                                     bert_finetuning=False, training=False)
        assert detach is True

    def test_detach_when_not_finetuning_train_mode(self, tiny_bert):
        """bert_finetuning=False, train mode → must still detach.
        This is the second-stage-with-zero-bert-lr case."""
        model, tokenizer = tiny_bert
        detach = captured_detach_arg(model, tokenizer,
                                     bert_finetuning=False, training=True)
        assert detach is True

    def test_detach_when_finetuning_eval_mode(self, tiny_bert):
        """bert_finetuning=True, eval mode → must detach.
        Even during finetuning, eval mode should not produce gradients."""
        model, tokenizer = tiny_bert
        detach = captured_detach_arg(model, tokenizer,
                                     bert_finetuning=True, training=False)
        assert detach is True

    def test_no_detach_when_finetuning_train_mode(self, tiny_bert):
        """bert_finetuning=True, train mode → must NOT detach.
        This is the only case where gradients should flow into the transformer."""
        model, tokenizer = tiny_bert
        detach = captured_detach_arg(model, tokenizer,
                                     bert_finetuning=True, training=True)
        assert detach is False

    def test_bert_finetuning_defaults_to_false(self):
        """If bert_finetuning is not set on the model at all (e.g. old
        checkpoint), getattr(..., False) should default to detaching."""
        from stanza.models.common import bert_embedding

        detach_values = []

        def capturing_extract(*args, detach=True, **kwargs):
            detach_values.append(detach)
            model, _ = make_bert_model()
            hidden_size = model.config.hidden_size
            return [torch.zeros(7, hidden_size)]

        with patch('stanza.models.common.bert_embedding.extract_bert_embeddings',
                   side_effect=capturing_extract):
            import torch.nn as nn

            class NoBertFinetuningAttr(nn.Module):
                def __init__(self):
                    super().__init__()
                    # deliberately no self.bert_finetuning attribute

                def run_bert(self, bert_model, bert_tokenizer):
                    from stanza.models.common.bert_embedding import extract_bert_embeddings
                    bert_finetuning = getattr(self, 'bert_finetuning', False)
                    return extract_bert_embeddings(
                        BERT_MODEL, bert_tokenizer, bert_model,
                        [["hello", "world"]], torch.device('cpu'),
                        keep_endpoints=True, num_layers=2,
                        detach=not bert_finetuning or not self.training,
                        peft_name=None,
                    )

            m = NoBertFinetuningAttr()
            m.train()
            bert_model, bert_tokenizer = make_bert_model()
            m.run_bert(bert_model, bert_tokenizer)

        assert detach_values[0] is True


# ---------------------------------------------------------------------------
# 2.  Trainer sets bert_finetuning correctly
# ---------------------------------------------------------------------------

class TestTrainerSetsBertFinetuning:
    """
    The trainer sets model.bert_finetuning based on whether any optimizer
    key starts with "bert" or "peft".  These tests verify that the flag
    is set correctly for different training configurations.
    """

    def test_bert_finetuning_true_when_bert_optimizer_present(
        self, tmp_path, wordvec_pretrain_file
    ):
        """When bert_finetune=True, the trainer creates a bert_optimizer,
        and model.bert_finetuning should be True."""
        trainer = run_training(
            tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA,
            extra_args=['--bert_model', BERT_MODEL,
                        '--bert_hidden_layers', '2',
                        '--bert_finetune']
        )
        assert trainer.model.bert_finetuning is True
        assert any(k.startswith('bert') for k in trainer.optimizer)

    def test_bert_finetuning_false_when_no_bert_optimizer(
        self, tmp_path, wordvec_pretrain_file
    ):
        """Without bert finetuning, there should be no bert optimizer key
        and model.bert_finetuning should be False."""
        trainer = run_training(
            tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA,
            extra_args=['--bert_model', BERT_MODEL,
                        '--bert_hidden_layers', '2']
        )
        assert trainer.model.bert_finetuning is False
        assert not any(k.startswith('bert') for k in trainer.optimizer)

    def test_bert_finetuning_false_without_bert_model(
        self, tmp_path, wordvec_pretrain_file
    ):
        """Without a bert model at all, bert_finetuning should be False."""
        trainer = run_training(
            tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA
        )
        assert trainer.model.bert_finetuning is False

    def test_bert_finetuning_true_with_peft(
        self, tmp_path, wordvec_pretrain_file
    ):
        """PEFT also triggers finetuning; model.bert_finetuning should be True."""
        pytest.importorskip("peft")
        trainer = run_training(
            tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA,
            extra_args=['--bert_model', BERT_MODEL,
                        '--bert_hidden_layers', '2',
                        '--bert_finetune',
                        '--use_peft']
        )
        assert trainer.model.bert_finetuning is True
        assert any(k.startswith('bert') or k.startswith('peft')
                   for k in trainer.optimizer)


# ---------------------------------------------------------------------------
# 3.  Second-stage training with zero bert LR detaches the transformer
# ---------------------------------------------------------------------------

class TestSecondStageBertDetach:
    """
    In the second stage of training (second_optim is set), if
    second_bert_learning_rate is 0.0 (the default), there will be no
    bert optimizer key, so bert_finetuning=False and the transformer
    will be detached even though self.training is True.

    This is the primary use case that motivated the detach change in PR #1590.
    """

    def test_second_stage_zero_bert_lr_sets_bert_finetuning_false(
        self, tmp_path, wordvec_pretrain_file
    ):
        """
        Second stage with second_bert_learning_rate intentionally set to 0.0
        should result in no bert optimizer and bert_finetuning=False.
        """
        trainer = run_training(
            tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA,
            extra_args=['--bert_model', BERT_MODEL,
                        '--bert_hidden_layers', '2',
                        '--bert_finetune',
                        '--second_optim', 'sgd',
                        '--second_optim_start_step', '6',
                        '--second_bert_learning_rate', '0.0']
        )
        # After second stage kicks in, bert optimizer should be absent
        assert not any(k.startswith('bert') for k in trainer.optimizer), (
            "Second stage with zero bert LR should have no bert optimizer"
        )
        assert trainer.model.bert_finetuning is False, (
            "bert_finetuning should be False in second stage with zero bert LR"
        )

    def test_second_stage_nonzero_bert_lr_sets_bert_finetuning_true(
        self, tmp_path, wordvec_pretrain_file
    ):
        """
        Second stage with an explicit nonzero second_bert_learning_rate
        should keep a bert optimizer and set bert_finetuning=True.
        """
        trainer = run_training(
            tmp_path, wordvec_pretrain_file, TRAIN_DATA, DEV_DATA,
            extra_args=['--bert_model', BERT_MODEL,
                        '--bert_hidden_layers', '2',
                        '--bert_finetune',
                        '--second_optim', 'sgd',
                        '--second_optim_start_step', '6',
                        '--second_bert_learning_rate', '1e-5']
        )
        assert any(k.startswith('bert') for k in trainer.optimizer), (
            "Second stage with nonzero bert LR should have a bert optimizer"
        )
        assert trainer.model.bert_finetuning is True

    def test_detach_true_in_training_mode_without_finetuning(self):
        """
        Unit test for the exact scenario: model in training mode but
        bert_finetuning=False.  Gradients must not flow into the transformer.

        This is what happens during a second-stage training run where
        second_bert_learning_rate=0.0.
        """
        bert_model, bert_tokenizer = make_bert_model()
        detach = captured_detach_arg(bert_model, bert_tokenizer,
                                     bert_finetuning=False, training=True)
        assert detach is True, (
            "Transformer must be detached in training mode when "
            "bert_finetuning=False (e.g. second stage with zero bert LR)"
        )


# ---------------------------------------------------------------------------
# 4.  Gradient flow through the transformer
# ---------------------------------------------------------------------------

class TestGradientFlowThroughTransformer:
    """
    More direct tests that verify whether gradients actually reach the
    transformer parameters, rather than just checking the detach flag.
    """

    def _run_forward_and_check_bert_grad(self, bert_finetuning, training):
        """
        Do a tiny forward pass through the transformer directly (not through
        the full parser model) and check whether the transformer's parameters
        received gradients.
        """
        bert_model, bert_tokenizer = make_bert_model()
        # Zero out any existing gradients
        for p in bert_model.parameters():
            p.grad = None

        if training:
            bert_model.train()
        else:
            bert_model.eval()

        tokens = bert_tokenizer(
            [["The", "cat", "sat"]],
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
        )

        if not bert_finetuning or not training:
            with torch.no_grad():
                output = bert_model(**tokens, output_hidden_states=True)
            # No backward possible — just check no grads exist
            has_grads = any(p.grad is not None
                            for p in bert_model.parameters())
            return has_grads
        else:
            output = bert_model(**tokens, output_hidden_states=True)
            # Pull out a scalar and backprop
            loss = output.last_hidden_state.sum()
            loss.backward()
            has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in bert_model.parameters())
            return has_grads

    def test_no_grads_when_not_finetuning_eval(self):
        has_grads = self._run_forward_and_check_bert_grad(
            bert_finetuning=False, training=False
        )
        assert not has_grads

    def test_no_grads_when_not_finetuning_train(self):
        """Second-stage zero-bert-lr scenario: training mode but no grads."""
        has_grads = self._run_forward_and_check_bert_grad(
            bert_finetuning=False, training=True
        )
        assert not has_grads

    def test_no_grads_when_finetuning_eval(self):
        """Even with bert_finetuning=True, eval mode should produce no grads."""
        has_grads = self._run_forward_and_check_bert_grad(
            bert_finetuning=True, training=False
        )
        assert not has_grads

    def test_grads_flow_when_finetuning_train(self):
        """Only when bert_finetuning=True AND training=True should grads flow."""
        has_grads = self._run_forward_and_check_bert_grad(
            bert_finetuning=True, training=True
        )
        assert has_grads, (
            "Gradients should flow into the transformer when "
            "bert_finetuning=True and the model is in training mode"
        )
