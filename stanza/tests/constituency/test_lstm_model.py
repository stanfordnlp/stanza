import os

import pytest
import torch

from stanza.models.common import pretrain
from stanza.models.common.utils import set_random_seed
from stanza.models.constituency import parse_transitions
from stanza.tests import *
from stanza.tests.constituency import test_parse_transitions
from stanza.tests.constituency.test_trainer import build_trainer

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

@pytest.fixture(scope="module")
def pretrain_file():
    return f'{TEST_WORKING_DIR}/in/tiny_emb.pt'

def build_model(pretrain_file, *args):
    # By default, we turn off multistage, since that can turn off various other structures in the initial training
    args = ['--no_multistage', '--pattn_num_layers', '4', '--pattn_d_model', '256', '--hidden_size', '128', '--use_lattn'] + list(args)
    trainer = build_trainer(pretrain_file, *args)
    return trainer.model

@pytest.fixture(scope="module")
def unary_model(pretrain_file):
    return build_model(pretrain_file, "--transition_scheme", "TOP_DOWN_UNARY")

def test_initial_state(unary_model):
    test_parse_transitions.test_initial_state(unary_model)

def test_shift(pretrain_file):
    # TODO: might be good to include some tests specifically for shift
    # in the context of a model with unaries
    model = build_model(pretrain_file)
    test_parse_transitions.test_shift(model)

def test_unary(unary_model):
    test_parse_transitions.test_unary(unary_model)

def test_unary_requires_root(unary_model):
    test_parse_transitions.test_unary_requires_root(unary_model)

def test_open(unary_model):
    test_parse_transitions.test_open(unary_model)

def test_compound_open(pretrain_file):
    model = build_model(pretrain_file, '--transition_scheme', "TOP_DOWN_COMPOUND")
    test_parse_transitions.test_compound_open(model)

def test_in_order_open(pretrain_file):
    model = build_model(pretrain_file, '--transition_scheme', "IN_ORDER")
    test_parse_transitions.test_in_order_open(model)

def test_close(unary_model):
    test_parse_transitions.test_close(unary_model)

def run_forward_checks(model, num_states=1):
    """
    Run a couple small transitions and a forward pass on the given model

    Results are not checked in any way.  This function allows for
    testing that building models with various options results in a
    functional model.
    """
    states = test_parse_transitions.build_initial_state(model, num_states)
    model(states)

    shift = parse_transitions.Shift()
    shifts = [shift for _ in range(num_states)]
    states = parse_transitions.bulk_apply(model, states, shifts)
    model(states)

    open_transition = parse_transitions.OpenConstituent("NP")
    open_transitions = [open_transition for _ in range(num_states)]
    assert open_transition.is_legal(states[0], model)
    states = parse_transitions.bulk_apply(model, states, open_transitions)
    assert states[0].num_opens == 1
    model(states)

    states = parse_transitions.bulk_apply(model, states, shifts)
    model(states)
    states = parse_transitions.bulk_apply(model, states, shifts)
    model(states)
    assert states[0].num_opens == 1
    # now should have "mox", "opal" on the constituents

    close_transition = parse_transitions.CloseConstituent()
    close_transitions = [close_transition for _ in range(num_states)]
    assert close_transition.is_legal(states[0], model)
    states = parse_transitions.bulk_apply(model, states, close_transitions)
    assert states[0].num_opens == 0

    model(states)

def test_unary_forward(unary_model):
    """
    Checks that the forward pass doesn't crash when run after various operations

    Doesn't check the forward pass for making reasonable answers
    """
    run_forward_checks(unary_model)

def test_lstm_forward(pretrain_file):
    model = build_model(pretrain_file)
    run_forward_checks(model, num_states=1)
    run_forward_checks(model, num_states=2)

def test_lstm_layers(pretrain_file):
    model = build_model(pretrain_file, '--num_lstm_layers', '1')
    run_forward_checks(model)
    model = build_model(pretrain_file, '--num_lstm_layers', '2')
    run_forward_checks(model)
    model = build_model(pretrain_file, '--num_lstm_layers', '3')
    run_forward_checks(model)

def test_multiple_output_forward(pretrain_file):
    """
    Test a couple different sizes of output layers
    """
    model = build_model(pretrain_file, '--num_output_layers', '1', '--num_lstm_layers', '2')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--num_output_layers', '2', '--num_lstm_layers', '2')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--num_output_layers', '3', '--num_lstm_layers', '2')
    run_forward_checks(model)

def test_no_tag_embedding_forward(pretrain_file):
    """
    Test that the model continues to work if the tag embedding is turned on or off
    """
    model = build_model(pretrain_file, '--tag_embedding_dim', '20')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--tag_embedding_dim', '0')
    run_forward_checks(model)

def test_forward_combined_dummy(pretrain_file):
    """
    Tests combined dummy and open node embeddings
    """
    model = build_model(pretrain_file, '--combined_dummy_embedding')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--no_combined_dummy_embedding')
    run_forward_checks(model)

def test_nonlinearity_init(pretrain_file):
    """
    Tests that different initialization methods of the nonlinearities result in valid tensors
    """
    model = build_model(pretrain_file, '--nonlinearity', 'relu')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--nonlinearity', 'tanh')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--nonlinearity', 'silu')
    run_forward_checks(model)

def test_forward_charlm(pretrain_file):
    """
    Tests loading and running a charlm

    Note that this doesn't test the results of the charlm itself,
    just that the model is shaped correctly
    """
    forward_charlm_path = os.path.join(TEST_MODELS_DIR, "en", "forward_charlm", "1billion.pt")
    backward_charlm_path = os.path.join(TEST_MODELS_DIR, "en", "backward_charlm", "1billion.pt")
    assert os.path.exists(forward_charlm_path), "Need to download en test models (or update path to the forward charlm)"
    assert os.path.exists(backward_charlm_path), "Need to download en test models (or update path to the backward charlm)"

    model = build_model(pretrain_file, '--charlm_forward_file', forward_charlm_path, '--charlm_backward_file', backward_charlm_path, '--sentence_boundary_vectors', 'none')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--charlm_forward_file', forward_charlm_path, '--charlm_backward_file', backward_charlm_path, '--sentence_boundary_vectors', 'words')
    run_forward_checks(model)

def test_forward_bert(pretrain_file):
    """
    Test on a tiny Bert, which hopefully does not take up too much disk space or memory
    """
    bert_model = "hf-internal-testing/tiny-bert"

    model = build_model(pretrain_file, '--bert_model', bert_model)
    run_forward_checks(model)


def test_forward_xlnet(pretrain_file):
    """
    Test on a tiny xlnet, which hopefully does not take up too much disk space or memory
    """
    bert_model = "hf-internal-testing/tiny-random-xlnet"

    model = build_model(pretrain_file, '--bert_model', bert_model)
    run_forward_checks(model)


def test_forward_sentence_boundaries(pretrain_file):
    """
    Test start & stop boundary vectors
    """
    model = build_model(pretrain_file, '--sentence_boundary_vectors', 'everything')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--sentence_boundary_vectors', 'words')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--sentence_boundary_vectors', 'none')
    run_forward_checks(model)

def test_forward_constituency_composition(pretrain_file):
    """
    Test different constituency composition functions
    """
    model = build_model(pretrain_file, '--constituency_composition', 'bilstm')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'max')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'key')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'untied_key')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'untied_max')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'bilstm_max')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'tree_lstm')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'tree_lstm_cx')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'bigram')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'attn')
    run_forward_checks(model, num_states=2)

def test_forward_key_position(pretrain_file):
    """
    Test KEY and UNTIED_KEY either with or without reduce_position
    """
    model = build_model(pretrain_file, '--constituency_composition', 'untied_key', '--reduce_position', '0')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'untied_key', '--reduce_position', '32')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'key', '--reduce_position', '0')
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'key', '--reduce_position', '32')
    run_forward_checks(model, num_states=2)


def test_forward_attn_hidden_size(pretrain_file):
    """
    Test that when attn is used with hidden sizes not evenly divisible by reduce_heads, the model reconfigures the hidden_size
    """
    model = build_model(pretrain_file, '--constituency_composition', 'attn', '--hidden_size', '129')
    assert model.hidden_size >= 129
    assert model.hidden_size % model.reduce_heads == 0
    run_forward_checks(model, num_states=2)

    model = build_model(pretrain_file, '--constituency_composition', 'attn', '--hidden_size', '129', '--reduce_heads', '10')
    assert model.hidden_size == 130
    assert model.reduce_heads == 10

def test_forward_partitioned_attention(pretrain_file):
    """
    Test with & without partitioned attention layers
    """
    model = build_model(pretrain_file, '--pattn_num_heads', '8', '--pattn_num_layers', '8')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--pattn_num_heads', '0', '--pattn_num_layers', '0')
    run_forward_checks(model)

def test_forward_labeled_attention(pretrain_file):
    """
    Test with & without labeled attention layers
    """
    model = build_model(pretrain_file, '--lattn_d_proj', '64', '--lattn_d_l', '16')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--lattn_d_proj', '0', '--lattn_d_l', '0')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--lattn_d_proj', '64', '--lattn_d_l', '16', '--lattn_combined_input')
    run_forward_checks(model)

def test_lattn_partitioned(pretrain_file):
    model = build_model(pretrain_file, '--lattn_d_proj', '64', '--lattn_d_l', '16', '--lattn_partitioned')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--lattn_d_proj', '64', '--lattn_d_l', '16', '--no_lattn_partitioned')
    run_forward_checks(model)


def test_lattn_projection(pretrain_file):
    """
    Test with & without labeled attention layers
    """
    with pytest.raises(ValueError):
        # this is too small
        model = build_model(pretrain_file, '--pattn_d_model', '1024', '--lattn_d_proj', '64', '--lattn_d_l', '16', '--lattn_d_input_proj', '256', '--lattn_partitioned')
        run_forward_checks(model)

    model = build_model(pretrain_file, '--pattn_d_model', '1024', '--lattn_d_proj', '64', '--lattn_d_l', '16', '--no_lattn_partitioned', '--lattn_d_input_proj', '256')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--lattn_d_proj', '64', '--lattn_d_l', '16', '--lattn_d_input_proj', '768')
    run_forward_checks(model)

    # check that it works if we turn off the projection,
    # in case having it on beccomes the default
    model = build_model(pretrain_file, '--lattn_d_proj', '64', '--lattn_d_l', '16', '--lattn_d_input_proj', '0')
    run_forward_checks(model)

def test_forward_timing_choices(pretrain_file):
    """
    Test different timing / position encodings
    """
    model = build_model(pretrain_file, '--pattn_num_heads', '4', '--pattn_num_layers', '4', '--pattn_timing', 'sin')
    run_forward_checks(model)

    model = build_model(pretrain_file, '--pattn_num_heads', '4', '--pattn_num_layers', '4', '--pattn_timing', 'learned')
    run_forward_checks(model)

def test_transition_stack(pretrain_file):
    """
    Test different transition stack types: lstm & attention
    """
    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--transition_stack', 'attn', '--transition_heads', '1')
    run_forward_checks(model)

    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--transition_stack', 'attn', '--transition_heads', '4')
    run_forward_checks(model)

    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--transition_stack', 'lstm')
    run_forward_checks(model)

def test_constituent_stack(pretrain_file):
    """
    Test different constituent stack types: lstm & attention
    """
    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--constituent_stack', 'attn', '--constituent_heads', '1')
    run_forward_checks(model)

    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--constituent_stack', 'attn', '--constituent_heads', '4')
    run_forward_checks(model)

    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--constituent_stack', 'lstm')
    run_forward_checks(model)

def test_different_transition_sizes(pretrain_file):
    """
    If the transition hidden size and embedding size are different, the model should still work
    """
    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--transition_embedding_dim', '10', '--transition_hidden_size', '10',
                        '--sentence_boundary_vectors', 'everything')
    run_forward_checks(model)

    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--transition_embedding_dim', '20', '--transition_hidden_size', '10',
                        '--sentence_boundary_vectors', 'everything')
    run_forward_checks(model)

    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--transition_embedding_dim', '10', '--transition_hidden_size', '20',
                        '--sentence_boundary_vectors', 'everything')
    run_forward_checks(model)

    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--transition_embedding_dim', '10', '--transition_hidden_size', '10',
                        '--sentence_boundary_vectors', 'none')
    run_forward_checks(model)

    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--transition_embedding_dim', '20', '--transition_hidden_size', '10',
                        '--sentence_boundary_vectors', 'none')
    run_forward_checks(model)

    model = build_model(pretrain_file,
                        '--pattn_num_layers', '0', '--lattn_d_proj', '0',
                        '--transition_embedding_dim', '10', '--transition_hidden_size', '20',
                        '--sentence_boundary_vectors', 'none')
    run_forward_checks(model)


def test_lstm_tree_forward(pretrain_file):
    """
    Test the LSTM_TREE forward pass
    """
    model = build_model(pretrain_file, '--num_tree_lstm_layers', '1', '--constituency_composition', 'tree_lstm')
    run_forward_checks(model)
    model = build_model(pretrain_file, '--num_tree_lstm_layers', '2', '--constituency_composition', 'tree_lstm')
    run_forward_checks(model)
    model = build_model(pretrain_file, '--num_tree_lstm_layers', '3', '--constituency_composition', 'tree_lstm')
    run_forward_checks(model)

def test_lstm_tree_cx_forward(pretrain_file):
    """
    Test the LSTM_TREE_CX forward pass
    """
    model = build_model(pretrain_file, '--num_tree_lstm_layers', '1', '--constituency_composition', 'tree_lstm_cx')
    run_forward_checks(model)
    model = build_model(pretrain_file, '--num_tree_lstm_layers', '2', '--constituency_composition', 'tree_lstm_cx')
    run_forward_checks(model)
    model = build_model(pretrain_file, '--num_tree_lstm_layers', '3', '--constituency_composition', 'tree_lstm_cx')
    run_forward_checks(model)

def test_maxout(pretrain_file):
    """
    Test with and without maxout layers for output
    """
    model = build_model(pretrain_file, '--maxout_k', '0')
    run_forward_checks(model)
    # check the output size & implicitly check the type
    # to check for a particularly silly bug
    assert model.output_layers[-1].weight.shape[0] == len(model.transitions)

    model = build_model(pretrain_file, '--maxout_k', '2')
    run_forward_checks(model)
    assert model.output_layers[-1].linear.weight.shape[0] == len(model.transitions) * 2

    model = build_model(pretrain_file, '--maxout_k', '3')
    run_forward_checks(model)
    assert model.output_layers[-1].linear.weight.shape[0] == len(model.transitions) * 3

def check_structure_test(pretrain_file, args1, args2):
    """
    Test that the "copy" method copies the parameters from one model to another

    Also check that the copied models produce the same results
    """
    set_random_seed(1000)
    other = build_model(pretrain_file, *args1)
    other.eval()

    set_random_seed(1001)
    model = build_model(pretrain_file, *args2)
    model.eval()

    assert not torch.allclose(model.delta_embedding.weight, other.delta_embedding.weight)
    assert not torch.allclose(model.output_layers[0].weight, other.output_layers[0].weight)

    model.copy_with_new_structure(other)

    assert torch.allclose(model.delta_embedding.weight, other.delta_embedding.weight)
    assert torch.allclose(model.output_layers[0].weight, other.output_layers[0].weight)
    # the norms will be the same, as the non-zero values are all the same
    assert torch.allclose(torch.linalg.norm(model.word_lstm.weight_ih_l0), torch.linalg.norm(other.word_lstm.weight_ih_l0))

    # now, check that applying one transition to an initial state
    # results in the same values in the output states for both models
    # as the pattn layer inputs are 0, the output values should be equal
    shift = [parse_transitions.Shift()]
    model_states = test_parse_transitions.build_initial_state(model, 1)
    model_states = parse_transitions.bulk_apply(model, model_states, shift)

    other_states = test_parse_transitions.build_initial_state(other, 1)
    other_states = parse_transitions.bulk_apply(other, other_states, shift)

    for i, j in zip(other_states[0].word_queue, model_states[0].word_queue):
        assert torch.allclose(i.hx, j.hx, atol=1e-07)
    for i, j in zip(other_states[0].transitions, model_states[0].transitions):
        assert torch.allclose(i.lstm_hx, j.lstm_hx)
        assert torch.allclose(i.lstm_cx, j.lstm_cx)
    for i, j in zip(other_states[0].constituents, model_states[0].constituents):
        assert (i.value is None) == (j.value is None)
        if i.value is not None:
            assert torch.allclose(i.value.tree_hx, j.value.tree_hx, atol=1e-07)
        assert torch.allclose(i.lstm_hx, j.lstm_hx)
        assert torch.allclose(i.lstm_cx, j.lstm_cx)

def test_copy_with_new_structure_same(pretrain_file):
    """
    Test that copying the structure with no changes works as expected
    """
    check_structure_test(pretrain_file,
                         ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10'],
                         ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10'])

def test_copy_with_new_structure_untied(pretrain_file):
    """
    Test that copying the structure with no changes works as expected
    """
    check_structure_test(pretrain_file,
                         ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10', '--constituency_composition', 'MAX'],
                         ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10', '--constituency_composition', 'UNTIED_MAX'])

def test_copy_with_new_structure_pattn(pretrain_file):
    check_structure_test(pretrain_file,
                         ['--pattn_num_layers', '0', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10'],
                         ['--pattn_num_layers', '1', '--lattn_d_proj', '0', '--hidden_size', '20', '--delta_embedding_dim', '10', '--pattn_d_model', '20', '--pattn_num_heads', '2'])

def test_copy_with_new_structure_both(pretrain_file):
    check_structure_test(pretrain_file,
                         ['--pattn_num_layers', '0', '--lattn_d_proj',  '0', '--hidden_size', '20', '--delta_embedding_dim', '10'],
                         ['--pattn_num_layers', '1', '--lattn_d_proj', '32', '--hidden_size', '20', '--delta_embedding_dim', '10', '--pattn_d_model', '20', '--pattn_num_heads', '2'])

def test_copy_with_new_structure_lattn(pretrain_file):
    check_structure_test(pretrain_file,
                         ['--pattn_num_layers', '1', '--lattn_d_proj',  '0', '--hidden_size', '20', '--delta_embedding_dim', '10', '--pattn_d_model', '20', '--pattn_num_heads', '2'],
                         ['--pattn_num_layers', '1', '--lattn_d_proj', '32', '--hidden_size', '20', '--delta_embedding_dim', '10', '--pattn_d_model', '20', '--pattn_num_heads', '2'])

def test_parse_tagged_words(pretrain_file):
    """
    Small test which doesn't check results, just execution
    """
    model = build_model(pretrain_file)

    sentence = [("I", "PRP"), ("am", "VBZ"), ("Luffa", "NNP")]

    # we don't expect a useful tree out of a random model
    # so we don't check the result
    # just check that it works without crashing
    result = model.parse_tagged_words([sentence], 10)
    assert len(result) == 1
    pts = [x for x in result[0].yield_preterminals()]

    for word, pt in zip(sentence, pts):
        assert pt.children[0].label == word[0]
        assert pt.label == word[1]
