from collections import Counter, namedtuple
import copy
import logging
import os
import random
import re

import torch
from torch import nn

#from stanza.models.common import pretrain

from stanza.models.common import utils
from stanza.models.common.foundation_cache import FoundationCache, NoTransformerFoundationCache
from stanza.models.common.large_margin_loss import LargeMarginInSoftmaxLoss
from stanza.models.common.utils import sort_with_indices, unsort
from stanza.models.constituency import parse_transitions
from stanza.models.constituency import transition_sequence
from stanza.models.constituency import tree_reader
from stanza.models.constituency.in_order_compound_oracle import InOrderCompoundOracle
from stanza.models.constituency.in_order_oracle import InOrderOracle
from stanza.models.constituency.lstm_model import LSTMModel
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.top_down_oracle import TopDownOracle
from stanza.models.constituency.trainer import Trainer
from stanza.models.constituency.utils import retag_trees, build_optimizer, build_scheduler, verify_transitions, get_open_nodes, check_constituents, check_root_labels, remove_duplicate_trees, remove_singleton_trees
from stanza.server.parser_eval import EvaluateParser, ParseResult
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

tlogger = logging.getLogger('stanza.constituency.trainer')

TrainItem = namedtuple("TrainItem", ['tree', 'gold_sequence', 'preterminals'])

class EpochStats(namedtuple("EpochStats", ['epoch_loss', 'transitions_correct', 'transitions_incorrect', 'repairs_used', 'fake_transitions_used', 'nans'])):
    def __add__(self, other):
        transitions_correct = self.transitions_correct + other.transitions_correct
        transitions_incorrect = self.transitions_incorrect + other.transitions_incorrect
        repairs_used = self.repairs_used + other.repairs_used
        fake_transitions_used = self.fake_transitions_used + other.fake_transitions_used
        epoch_loss = self.epoch_loss + other.epoch_loss
        nans = self.nans + other.nans
        return EpochStats(epoch_loss, transitions_correct, transitions_incorrect, repairs_used, fake_transitions_used, nans)

def evaluate(args, model_file, retag_pipeline):
    """
    Loads the given model file and tests the eval_file treebank.

    May retag the trees using retag_pipeline
    Uses a subprocess to run the Java EvalB code
    """
    # we create the Evaluator here because otherwise the transformers
    # library constantly complains about forking the process
    # note that this won't help in the event of training multiple
    # models in the same run, although since that would take hours
    # or days, that's not a very common problem
    if args['num_generate'] > 0:
        kbest = args['num_generate'] + 1
    else:
        kbest = None

    with EvaluateParser(kbest=kbest) as evaluator:
        foundation_cache = retag_pipeline[0].foundation_cache if retag_pipeline else FoundationCache()
        load_args = {
            'wordvec_pretrain_file': args['wordvec_pretrain_file'],
            'charlm_forward_file': args['charlm_forward_file'],
            'charlm_backward_file': args['charlm_backward_file'],
            'device': args['device'],
        }
        trainer = Trainer.load(model_file, args=load_args, foundation_cache=foundation_cache)

        if args['log_shapes']:
            trainer.log_shapes()

        treebank = tree_reader.read_treebank(args['eval_file'])
        tlogger.info("Read %d trees for evaluation", len(treebank))

        retagged_treebank = treebank
        if retag_pipeline is not None:
            retag_method = trainer.model.retag_method
            retag_xpos = retag_method == 'xpos'
            tlogger.info("Retagging trees using the %s tags from the %s package...", retag_method, args['retag_package'])
            retagged_treebank = retag_trees(treebank, retag_pipeline, retag_xpos)
            tlogger.info("Retagging finished")

        if args['log_norms']:
            trainer.log_norms()
        f1, kbestF1, _ = run_dev_set(trainer.model, retagged_treebank, treebank, args, evaluator)
        tlogger.info("F1 score on %s: %f", args['eval_file'], f1)
        if kbestF1 is not None:
            tlogger.info("KBest F1 score on %s: %f", args['eval_file'], kbestF1)

def remove_optimizer(args, model_save_file, model_load_file):
    """
    A utility method to remove the optimizer from a save file

    Will make the save file a lot smaller
    """
    # TODO: kind of overkill to load in the pretrain rather than
    # change the load/save to work without it, but probably this
    # functionality isn't used that often anyway
    load_args = {
        'wordvec_pretrain_file': args['wordvec_pretrain_file'],
        'charlm_forward_file': args['charlm_forward_file'],
        'charlm_backward_file': args['charlm_backward_file'],
        'device': args['device'],
    }
    trainer = Trainer.load(model_load_file, args=load_args, load_optimizer=False)
    trainer.save(model_save_file)

def add_grad_clipping(trainer, grad_clipping):
    """
    Adds a torch.clamp hook on each parameter if grad_clipping is not None
    """
    if grad_clipping is not None:
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -grad_clipping, grad_clipping))

def build_trainer(args, train_trees, dev_trees, silver_trees, foundation_cache, model_load_file):
    """
    Builds a Trainer (with model) and the train_sequences and transitions for the given trees.
    """
    train_constituents = Tree.get_unique_constituent_labels(train_trees)
    tlogger.info("Unique constituents in training set: %s", train_constituents)
    if args['check_valid_states']:
        check_constituents(train_constituents, dev_trees, "dev", fail=args['strict_check_constituents'])
        check_constituents(train_constituents, silver_trees, "silver", fail=args['strict_check_constituents'])
    constituent_counts = Tree.get_constituent_counts(train_trees)
    tlogger.info("Constituent node counts: %s", constituent_counts)

    tags = Tree.get_unique_tags(train_trees)
    if None in tags:
        raise RuntimeError("Fatal problem: the tagger put None on some of the nodes!")
    tlogger.info("Unique tags in training set: %s", tags)
    # no need to fail for missing tags between train/dev set
    # the model has an unknown tag embedding
    for tag in Tree.get_unique_tags(dev_trees):
        if tag not in tags:
            tlogger.info("Found tag in dev set which does not exist in train set: %s  Continuing...", tag)

    unary_limit = max(max(t.count_unary_depth() for t in train_trees),
                      max(t.count_unary_depth() for t in dev_trees)) + 1
    if silver_trees:
        unary_limit = max(unary_limit, max(t.count_unary_depth() for t in silver_trees))
    tlogger.info("Unary limit: %d", unary_limit)
    train_sequences, train_transitions = transition_sequence.convert_trees_to_sequences(train_trees, "training", args['transition_scheme'], args['reversed'])
    dev_sequences, dev_transitions = transition_sequence.convert_trees_to_sequences(dev_trees, "dev", args['transition_scheme'], args['reversed'])
    silver_sequences, silver_transitions = transition_sequence.convert_trees_to_sequences(silver_trees, "silver", args['transition_scheme'], args['reversed'])

    tlogger.info("Total unique transitions in train set: %d", len(train_transitions))
    tlogger.info("Unique transitions in training set:\n  %s", "\n  ".join(map(str, train_transitions)))
    expanded_train_transitions = set(train_transitions + [x for trans in train_transitions for x in trans.components()])
    if args['check_valid_states']:
        parse_transitions.check_transitions(expanded_train_transitions, dev_transitions, "dev")
        # theoretically could just train based on the items in the silver dataset
        parse_transitions.check_transitions(expanded_train_transitions, silver_transitions, "silver")

    root_labels = Tree.get_root_labels(train_trees)
    check_root_labels(root_labels, dev_trees, "dev")
    check_root_labels(root_labels, silver_trees, "silver")
    tlogger.info("Root labels in treebank: %s", root_labels)

    verify_transitions(train_trees, train_sequences, args['transition_scheme'], unary_limit, args['reversed'], "train", root_labels)
    verify_transitions(dev_trees, dev_sequences, args['transition_scheme'], unary_limit, args['reversed'], "dev", root_labels)

    # we don't check against the words in the dev set as it is
    # expected there will be some UNK words
    words = Tree.get_unique_words(train_trees)
    rare_words = Tree.get_rare_words(train_trees, args['rare_word_threshold'])
    # rare/unknown silver words will just get UNK if they are not already known
    if silver_trees and args['use_silver_words']:
        tlogger.info("Getting silver words to add to the delta embedding")
        silver_words = Tree.get_common_words(tqdm(silver_trees, postfix='Silver words'), len(words))
        words = sorted(set(words + silver_words))

    # also, it's not actually an error if there is a pattern of
    # compound unary or compound open nodes which doesn't exist in the
    # train set.  it just means we probably won't ever get that right
    open_nodes = get_open_nodes(train_trees, args['transition_scheme'])
    tlogger.info("Using the following open nodes:\n  %s", "\n  ".join(map(str, open_nodes)))

    # at this point we have:
    # pretrain
    # train_trees, dev_trees
    # lists of transitions, internal nodes, and root states the parser needs to be aware of

    trainer = Trainer.build_trainer(args, train_transitions, train_constituents, tags, words, rare_words, root_labels, open_nodes, unary_limit, foundation_cache, model_load_file)

    trainer.log_num_words_known(words)
    # grad clipping is not saved with the rest of the model,
    # so even in the case of a model we saved,
    # we now have to add the grad clipping
    add_grad_clipping(trainer, args['grad_clipping'])

    return trainer, train_sequences, silver_sequences, train_transitions

def train(args, model_load_file, retag_pipeline):
    """
    Build a model, train it using the requested train & dev files
    """
    utils.log_training_args(args, tlogger)

    # we create the Evaluator here because otherwise the transformers
    # library constantly complains about forking the process
    # note that this won't help in the event of training multiple
    # models in the same run, although since that would take hours
    # or days, that's not a very common problem
    if args['num_generate'] > 0:
        kbest = args['num_generate'] + 1
    else:
        kbest = None

    if args['wandb']:
        global wandb
        import wandb
        wandb_name = args['wandb_name'] if args['wandb_name'] else "%s_constituency" % args['shorthand']
        wandb.init(name=wandb_name, config=args)
        wandb.run.define_metric('dev_score', summary='max')

    with EvaluateParser(kbest=kbest) as evaluator:
        utils.ensure_dir(args['save_dir'])

        train_trees = tree_reader.read_treebank(args['train_file'])
        tlogger.info("Read %d trees for the training set", len(train_trees))
        if args['train_remove_duplicates']:
            train_trees = remove_duplicate_trees(train_trees, "train")
        train_trees = remove_singleton_trees(train_trees)

        dev_trees = tree_reader.read_treebank(args['eval_file'])
        tlogger.info("Read %d trees for the dev set", len(dev_trees))
        dev_trees = remove_duplicate_trees(dev_trees, "dev")

        silver_trees = []
        if args['silver_file']:
            silver_trees = tree_reader.read_treebank(args['silver_file'])
            tlogger.info("Read %d trees for the silver training set", len(silver_trees))
            if args['silver_remove_duplicates']:
                silver_trees = remove_duplicate_trees(silver_trees, "silver")

        if retag_pipeline is not None:
            tlogger.info("Retagging trees using the %s tags from the %s package...", args['retag_method'], args['retag_package'])
            train_trees = retag_trees(train_trees, retag_pipeline, args['retag_xpos'])
            dev_trees = retag_trees(dev_trees, retag_pipeline, args['retag_xpos'])
            silver_trees = retag_trees(silver_trees, retag_pipeline, args['retag_xpos'])
            tlogger.info("Retagging finished")

        foundation_cache = retag_pipeline[0].foundation_cache if retag_pipeline else FoundationCache()
        trainer, train_sequences, silver_sequences, train_transitions = build_trainer(args, train_trees, dev_trees, silver_trees, foundation_cache, model_load_file)

        if args['log_shapes']:
            trainer.log_shapes()
        trainer = iterate_training(args, trainer, train_trees, train_sequences, train_transitions, dev_trees, silver_trees, silver_sequences, foundation_cache, evaluator)

    if args['wandb']:
        wandb.finish()

    return trainer

def compose_train_data(trees, sequences):
    preterminal_lists = [[Tree(label=preterminal.label, children=Tree(label=preterminal.children[0].label))
                          for preterminal in tree.yield_preterminals()]
                         for tree in trees]
    data = [TrainItem(*x) for x in zip(trees, sequences, preterminal_lists)]
    return data

def next_epoch_data(leftover_training_data, train_data, epoch_size):
    """
    Return the next epoch_size trees from the training data, starting
    with leftover data from the previous epoch if there is any

    The training loop generally operates on a fixed number of trees,
    rather than going through all the trees in the training set
    exactly once, and keeping the leftover training data via this
    function ensures that each tree in the training set is touched
    once before beginning to iterate again.
    """
    if not train_data:
        return [], []

    epoch_data = leftover_training_data
    while len(epoch_data) < epoch_size:
        random.shuffle(train_data)
        epoch_data.extend(train_data)
    leftover_training_data = epoch_data[epoch_size:]
    epoch_data = epoch_data[:epoch_size]

    return leftover_training_data, epoch_data

def update_bert_learning_rate(args, optimizer, epochs_trained):
    """
    Update the learning rate for the bert finetuning, if applicable
    """
    # would be nice to have a parameter group specific scheduler
    # however, there is an issue with the optimizer we had the most success with, madgrad
    # when the learning rate is 0 for a group, it still learns by some
    # small amount because of the eps parameter
    # in fact, that is enough to make the learning for the bert in the
    # second half broken
    for base_param_group in optimizer.param_groups:
        if base_param_group['param_group_name'] == 'base':
            break
    else:
        raise AssertionError("There should always be a base parameter group")
    for param_group in optimizer.param_groups:
        if param_group['param_group_name'] == 'bert':
            # Occasionally a model goes haywire and forgets how to use the transformer
            # So far we have only seen this happen with Electra on the non-NML version of PTB
            # We tried fixing that with an increasing transformer learning rate, but that
            # didn't fully resolve the problem
            # Switching to starting the finetuning after a few epochs seems to help a lot, though
            old_lr = param_group['lr']
            if args['bert_finetune_begin_epoch'] is not None and epochs_trained < args['bert_finetune_begin_epoch']:
                param_group['lr'] = 0.0
            elif args['bert_finetune_end_epoch'] is not None and epochs_trained >= args['bert_finetune_end_epoch']:
                param_group['lr'] = 0.0
            elif args['multistage'] and epochs_trained < args['epochs'] // 2:
                param_group['lr'] = base_param_group['lr'] * args['stage1_bert_learning_rate']
            else:
                param_group['lr'] = base_param_group['lr'] * args['bert_learning_rate']
            if param_group['lr'] != old_lr:
                tlogger.info("Setting %s finetuning rate from %f to %f", param_group['param_group_name'], old_lr, param_group['lr'])

def iterate_training(args, trainer, train_trees, train_sequences, transitions, dev_trees, silver_trees, silver_sequences, foundation_cache, evaluator):
    """
    Given an initialized model, a processed dataset, and a secondary dev dataset, train the model

    The training is iterated in the following loop:
      extract a batch of trees of the same length from the training set
      convert those trees into initial parsing states
      repeat until trees are done:
        batch predict the model's interpretation of the current states
        add the errors to the list of things to backprop
        advance the parsing state for each of the trees
    """
    # Somewhat unusual, but possibly related to the extreme variability in length of trees
    # Various experiments generally show about 0.5 F1 loss on various
    # datasets when using 'mean' instead of 'sum' for reduction
    # (Remember to adjust the weight decay when rerunning that experiment)
    if args['loss'] == 'cross':
        tlogger.info("Building CrossEntropyLoss(sum)")
        process_outputs = lambda x: x
        model_loss_function = nn.CrossEntropyLoss(reduction='sum')
    elif args['loss'] == 'focal':
        try:
            from focal_loss.focal_loss import FocalLoss
        except ImportError:
            raise ImportError("focal_loss not installed.  Must `pip install focal_loss_torch` to use the --loss=focal feature")
        tlogger.info("Building FocalLoss, gamma=%f", args['loss_focal_gamma'])
        process_outputs = lambda x: torch.softmax(x, dim=1)
        model_loss_function = FocalLoss(reduction='sum', gamma=args['loss_focal_gamma'])
    elif args['loss'] == 'large_margin':
        tlogger.info("Building LargeMarginInSoftmaxLoss(sum)")
        process_outputs = lambda x: x
        model_loss_function = LargeMarginInSoftmaxLoss(reduction='sum')
    else:
        raise ValueError("Unexpected loss term: %s" % args['loss'])

    device = trainer.device
    model_loss_function.to(device)
    transition_tensors = {x: torch.tensor(y, requires_grad=False, device=device).unsqueeze(0)
                          for (y, x) in enumerate(trainer.transitions)}
    trainer.train()

    train_data = compose_train_data(train_trees, train_sequences)
    silver_data = compose_train_data(silver_trees, silver_sequences)

    if not args['epoch_size']:
        args['epoch_size'] = len(train_data)
    if silver_data and not args['silver_epoch_size']:
        args['silver_epoch_size'] = args['epoch_size']

    if args['multistage']:
        multistage_splits = {}
        # if we're halfway, only do pattn.  save lattn for next time
        multistage_splits[args['epochs'] // 2] = (args['pattn_num_layers'], False)
        if LSTMModel.uses_lattn(args):
            multistage_splits[args['epochs'] * 3 // 4] = (args['pattn_num_layers'], True)

    oracle = None
    if args['transition_scheme'] is TransitionScheme.IN_ORDER:
        oracle = InOrderOracle(trainer.root_labels, args['oracle_level'], args['additional_oracle_levels'], args['deactivated_oracle_levels'])
    elif args['transition_scheme'] is TransitionScheme.IN_ORDER_COMPOUND:
        oracle = InOrderCompoundOracle(trainer.root_labels, args['oracle_level'], args['additional_oracle_levels'], args['deactivated_oracle_levels'])
    elif args['transition_scheme'] is TransitionScheme.TOP_DOWN:
        oracle = TopDownOracle(trainer.root_labels, args['oracle_level'], args['additional_oracle_levels'], args['deactivated_oracle_levels'])

    leftover_training_data = []
    leftover_silver_data = []
    if trainer.best_epoch > 0:
        tlogger.info("Restarting trainer with a model trained for %d epochs.  Best epoch %d, f1 %f", trainer.epochs_trained, trainer.best_epoch, trainer.best_f1)

    # if we're training a new model, save the initial state so it can be inspected
    if args['save_each_start'] == 0 and trainer.epochs_trained == 0:
        trainer.save(args['save_each_name'] % trainer.epochs_trained, save_optimizer=True)

    # trainer.epochs_trained+1 so that if the trainer gets saved after 1 epoch, the epochs_trained is 1
    for trainer.epochs_trained in range(trainer.epochs_trained+1, args['epochs']+1):
        trainer.train()
        tlogger.info("Starting epoch %d", trainer.epochs_trained)
        update_bert_learning_rate(args, trainer.optimizer, trainer.epochs_trained)

        if args['log_norms']:
            trainer.log_norms()
        leftover_training_data, epoch_data = next_epoch_data(leftover_training_data, train_data, args['epoch_size'])
        leftover_silver_data, epoch_silver_data = next_epoch_data(leftover_silver_data, silver_data, args['silver_epoch_size'])
        epoch_data = epoch_data + epoch_silver_data
        epoch_data.sort(key=lambda x: len(x[1]))

        epoch_stats = train_model_one_epoch(trainer.epochs_trained, trainer, transition_tensors, process_outputs, model_loss_function, epoch_data, oracle, args)

        # print statistics
        # by now we've forgotten about the original tags on the trees,
        # but it doesn't matter for hill climbing
        f1, _, _ = run_dev_set(trainer.model, dev_trees, dev_trees, args, evaluator)
        if f1 > trainer.best_f1 or (trainer.best_epoch == 0 and trainer.best_f1 == 0.0):
            # best_epoch == 0 to force a save of an initial model
            # useful for tests which expect something, even when a
            # very simple model didn't learn anything
            tlogger.info("New best dev score: %.5f > %.5f", f1, trainer.best_f1)
            trainer.best_f1 = f1
            trainer.best_epoch = trainer.epochs_trained
            trainer.save(args['save_name'], save_optimizer=False)
        if epoch_stats.nans > 0:
            tlogger.warning("Had to ignore %d batches with NaN", epoch_stats.nans)
        stats_log_lines = [
            "Epoch %d finished" % trainer.epochs_trained,
            "Transitions correct: %s" % epoch_stats.transitions_correct,
            "Transitions incorrect: %s" % epoch_stats.transitions_incorrect,
            "Total loss for epoch: %.5f" % epoch_stats.epoch_loss,
            "Dev score      (%5d): %8f" % (trainer.epochs_trained, f1),
            "Best dev score (%5d): %8f" % (trainer.best_epoch, trainer.best_f1)
        ]
        tlogger.info("\n  ".join(stats_log_lines))

        old_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.scheduler.step(f1)
        new_lr = trainer.optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            tlogger.info("Updating learning rate from %f to %f", old_lr, new_lr)

        if args['wandb']:
            wandb.log({'epoch_loss': epoch_stats.epoch_loss, 'dev_score': f1}, step=trainer.epochs_trained)
            if args['wandb_norm_regex']:
                watch_regex = re.compile(args['wandb_norm_regex'])
                for n, p in trainer.model.named_parameters():
                    if watch_regex.search(n):
                        wandb.log({n: torch.linalg.norm(p)})

        if args['early_dropout'] > 0 and trainer.epochs_trained >= args['early_dropout']:
            if any(x > 0.0 for x in (trainer.model.word_dropout.p, trainer.model.predict_dropout.p, trainer.model.lstm_input_dropout.p)):
                tlogger.info("Setting dropout to 0.0 at epoch %d", trainer.epochs_trained)
            trainer.model.word_dropout.p = 0
            trainer.model.predict_dropout.p = 0
            trainer.model.lstm_input_dropout.p = 0

        # recreate the optimizer and alter the model as needed if we hit a new multistage split
        if args['multistage'] and trainer.epochs_trained in multistage_splits:
            # we may be loading a save model from an earlier epoch if the scores stopped increasing
            epochs_trained = trainer.epochs_trained
            batches_trained = trainer.batches_trained

            stage_pattn_layers, stage_uses_lattn = multistage_splits[epochs_trained]

            # when loading the model, let the saved model determine whether it has pattn or lattn
            temp_args = copy.deepcopy(trainer.model.args)
            temp_args.pop('pattn_num_layers', None)
            temp_args.pop('lattn_d_proj', None)
            # overwriting the old trainer & model will hopefully free memory
            # load a new bert, even in PEFT mode, mostly so that the bert model
            # doesn't collect a whole bunch of PEFTs
            # for one thing, two PEFTs would mean 2x the optimizer parameters,
            # messing up saving and loading the optimizer without jumping
            # through more hoops
            # loading the trainer w/o the foundation_cache should create
            # the necessary bert_model and bert_tokenizer, and then we
            # can reuse those values when building out new LSTMModel
            trainer = Trainer.load(args['save_name'], temp_args, load_optimizer=False)
            model = trainer.model
            tlogger.info("Finished stage at epoch %d.  Restarting optimizer", epochs_trained)
            tlogger.info("Previous best model was at epoch %d", trainer.epochs_trained)

            temp_args = dict(args)
            tlogger.info("Switching to a model with %d pattn layers and %slattn", stage_pattn_layers, "" if stage_uses_lattn else "NO ")
            temp_args['pattn_num_layers'] = stage_pattn_layers
            if not stage_uses_lattn:
                temp_args['lattn_d_proj'] = 0
            pt = foundation_cache.load_pretrain(args['wordvec_pretrain_file'])
            forward_charlm = foundation_cache.load_charlm(args['charlm_forward_file'])
            backward_charlm = foundation_cache.load_charlm(args['charlm_backward_file'])
            new_model = LSTMModel(pt,
                                  forward_charlm,
                                  backward_charlm,
                                  model.bert_model,
                                  model.bert_tokenizer,
                                  model.force_bert_saved,
                                  model.peft_name,
                                  model.transitions,
                                  model.constituents,
                                  model.tags,
                                  model.delta_words,
                                  model.rare_words,
                                  model.root_labels,
                                  model.constituent_opens,
                                  model.unary_limit(),
                                  temp_args)
            new_model.to(device)
            new_model.copy_with_new_structure(model)

            optimizer = build_optimizer(temp_args, new_model, False)
            scheduler = build_scheduler(temp_args, optimizer)
            trainer = Trainer(new_model, optimizer, scheduler, epochs_trained, batches_trained, trainer.best_f1, trainer.best_epoch)
            add_grad_clipping(trainer, args['grad_clipping'])

        # checkpoint needs to be saved AFTER rebuilding the optimizer
        # so that assumptions about the optimizer in the checkpoint
        # can be made based on the end of the epoch
        if args['checkpoint'] and args['checkpoint_save_name']:
            trainer.save(args['checkpoint_save_name'], save_optimizer=True)
        # same with the "each filename", actually, in case those are
        # brought back for more training or even just for testing
        if args['save_each_start'] is not None and args['save_each_start'] <= trainer.epochs_trained and trainer.epochs_trained % args['save_each_frequency'] == 0:
            trainer.save(args['save_each_name'] % trainer.epochs_trained, save_optimizer=args['save_each_optimizer'])

    return trainer

def train_model_one_epoch(epoch, trainer, transition_tensors, process_outputs, model_loss_function, epoch_data, oracle, args):
    interval_starts = list(range(0, len(epoch_data), args['train_batch_size']))
    random.shuffle(interval_starts)

    optimizer = trainer.optimizer

    epoch_stats = EpochStats(0.0, Counter(), Counter(), Counter(), 0, 0)

    for batch_idx, interval_start in enumerate(tqdm(interval_starts, postfix="Epoch %d" % epoch)):
        batch = epoch_data[interval_start:interval_start+args['train_batch_size']]
        batch_stats = train_model_one_batch(epoch, batch_idx, trainer.model, batch, transition_tensors, process_outputs, model_loss_function, oracle, args)
        trainer.batches_trained += 1

        # Early in the training, some trees will be degenerate in a
        # way that results in layers going up the tree amplifying the
        # weights until they overflow.  Generally that problem
        # resolves itself in a few iterations, so for now we just
        # ignore those batches, but report how often it happens
        if batch_stats.nans == 0:
            optimizer.step()
        optimizer.zero_grad()
        epoch_stats = epoch_stats + batch_stats


    # TODO: refactor the logging?
    total_correct = sum(v for _, v in epoch_stats.transitions_correct.items())
    total_incorrect = sum(v for _, v in epoch_stats.transitions_incorrect.items())
    tlogger.info("Transitions correct: %d\n  %s", total_correct, str(epoch_stats.transitions_correct))
    tlogger.info("Transitions incorrect: %d\n  %s", total_incorrect, str(epoch_stats.transitions_incorrect))
    if len(epoch_stats.repairs_used) > 0:
        tlogger.info("Oracle repairs:\n  %s", "\n  ".join("%s (%s): %d" % (x.name, x.value, y) for x, y in epoch_stats.repairs_used.most_common()))
    if epoch_stats.fake_transitions_used > 0:
        tlogger.info("Fake transitions used: %d", epoch_stats.fake_transitions_used)

    return epoch_stats

def train_model_one_batch(epoch, batch_idx, model, training_batch, transition_tensors, process_outputs, model_loss_function, oracle, args):
    """
    Train the model for one batch

    The model itself will be updated, and a bunch of stats are returned
    It is unclear if this refactoring is useful in any way.  Might not be

    ... although the indentation does get pretty ridiculous if this is
    merged into train_model_one_epoch and then iterate_training
    """
    # now we add the state to the trees in the batch
    # the state is built as a bulk operation
    current_batch = model.initial_state_from_preterminals([x.preterminals for x in training_batch],
                                                          [x.tree for x in training_batch],
                                                          [x.gold_sequence for x in training_batch])

    transitions_correct = Counter()
    transitions_incorrect = Counter()
    repairs_used = Counter()
    fake_transitions_used = 0

    all_errors = []
    all_answers = []

    # we iterate through the batch in the following sequence:
    # predict the logits and the applied transition for each tree in the batch
    # collect errors
    #  - we always train to the desired one-hot vector
    #    this was a noticeable improvement over training just the
    #    incorrect transitions
    # determine whether the training can continue using the "student" transition
    #   or if we need to use teacher forcing
    # update all states using either the gold or predicted transition
    # any trees which are now finished are removed from the training cycle
    while len(current_batch) > 0:
        outputs, pred_transitions, _ = model.predict(current_batch, is_legal=False)
        gold_transitions = [x.gold_sequence[x.num_transitions] for x in current_batch]
        trans_tensor = [transition_tensors[gold_transition] for gold_transition in gold_transitions]
        all_errors.append(outputs)
        all_answers.extend(trans_tensor)

        new_batch = []
        update_transitions = []
        for pred_transition, gold_transition, state in zip(pred_transitions, gold_transitions, current_batch):
            # forget teacher forcing vs scheduled sampling
            # we're going with idiot forcing
            if pred_transition == gold_transition:
                transitions_correct[gold_transition.short_name()] += 1
                if state.num_transitions + 1 < len(state.gold_sequence):
                    if oracle is not None and epoch >= args['oracle_initial_epoch'] and random.random() < args['oracle_forced_errors']:
                        # TODO: could randomly choose from the legal transitions
                        # perhaps the second best scored transition
                        fake_transition = random.choice(model.transitions)
                        if fake_transition.is_legal(state, model):
                            _, new_sequence = oracle.fix_error(fake_transition, model, state)
                            if new_sequence is not None:
                                new_batch.append(state._replace(gold_sequence=new_sequence))
                                update_transitions.append(fake_transition)
                                fake_transitions_used = fake_transitions_used + 1
                                continue
                    new_batch.append(state)
                    update_transitions.append(gold_transition)
                continue

            transitions_incorrect[gold_transition.short_name(), pred_transition.short_name()] += 1
            # if we are on the final operation, there are two choices:
            #   - the parsing mode is IN_ORDER, and the final transition
            #     is the close to end the sequence, which has no alternatives
            #   - the parsing mode is something else, in which case
            #     we have no oracle anyway
            if state.num_transitions + 1 >= len(state.gold_sequence):
                continue

            if oracle is None or epoch < args['oracle_initial_epoch'] or not pred_transition.is_legal(state, model):
                new_batch.append(state)
                update_transitions.append(gold_transition)
                continue

            repair_type, new_sequence = oracle.fix_error(pred_transition, model, state)
            # we can only reach here on an error
            assert not repair_type.is_correct
            repairs_used[repair_type] += 1
            if new_sequence is not None and random.random() < args['oracle_frequency']:
                new_batch.append(state._replace(gold_sequence=new_sequence))
                update_transitions.append(pred_transition)
            else:
                new_batch.append(state)
                update_transitions.append(gold_transition)

        if len(current_batch) > 0:
            # bulk update states - significantly faster
            current_batch = model.bulk_apply(new_batch, update_transitions, fail=True)

    errors = torch.cat(all_errors)
    answers = torch.cat(all_answers)

    errors = process_outputs(errors)
    tree_loss = model_loss_function(errors, answers)
    tree_loss.backward()
    if args['watch_regex']:
        matched = False
        tlogger.info("Watching %s   ... epoch %d batch %d", args['watch_regex'], epoch, batch_idx)
        watch_regex = re.compile(args['watch_regex'])
        for n, p in trainer.model.named_parameters():
            if watch_regex.search(n):
                matched = True
                if p.requires_grad and p.grad is not None:
                    tlogger.info("  %s norm: %f grad: %f", n, torch.linalg.norm(p), torch.linalg.norm(p.grad))
                elif p.requires_grad:
                    tlogger.info("  %s norm: %f grad required, but is None!", n, torch.linalg.norm(p))
                else:
                    tlogger.info("  %s norm: %f grad not required", n, torch.linalg.norm(p))
        if not matched:
            tlogger.info("  (none found!)")
    if torch.any(torch.isnan(tree_loss)):
        batch_loss = 0.0
        nans = 1
    else:
        batch_loss = tree_loss.item()
        nans = 0

    return EpochStats(batch_loss, transitions_correct, transitions_incorrect, repairs_used, fake_transitions_used, nans)

def run_dev_set(model, retagged_trees, original_trees, args, evaluator=None):
    """
    This reparses a treebank and executes the CoreNLP Java EvalB code.

    It only works if CoreNLP 4.3.0 or higher is in the classpath.
    """
    tlogger.info("Processing %d trees from %s", len(retagged_trees), args['eval_file'])
    model.eval()

    num_generate = args.get('num_generate', 0)
    keep_scores = num_generate > 0

    sorted_trees, original_indices = sort_with_indices(retagged_trees, key=len, reverse=True)
    tree_iterator = iter(tqdm(sorted_trees))
    treebank = model.parse_sentences_no_grad(tree_iterator, model.build_batch_from_trees, args['eval_batch_size'], model.predict, keep_scores=keep_scores)
    treebank = unsort(treebank, original_indices)
    full_results = treebank

    if num_generate > 0:
        tlogger.info("Generating %d random analyses", args['num_generate'])
        generated_treebanks = [treebank]
        for i in tqdm(range(num_generate)):
            tree_iterator = iter(tqdm(retagged_trees, leave=False, postfix="tb%03d" % i))
            generated_treebanks.append(model.parse_sentences_no_grad(tree_iterator, model.build_batch_from_trees, args['eval_batch_size'], model.weighted_choice, keep_scores=keep_scores))

        #best_treebank = [ParseResult(parses[0].gold, [max([p.predictions[0] for p in parses], key=itemgetter(1))], None, None)
        #                 for parses in zip(*generated_treebanks)]
        #generated_treebanks = [best_treebank] + generated_treebanks

        # TODO: if the model is dropping trees, this will not work
        full_results = [ParseResult(parses[0].gold, [p.predictions[0] for p in parses], None, None)
                        for parses in zip(*generated_treebanks)]

    if len(full_results) < len(retagged_trees):
        tlogger.warning("Only evaluating %d trees instead of %d", len(full_results), len(retagged_trees))
    else:
        full_results = [x._replace(gold=gold) for x, gold in zip(full_results, original_trees)]

    if args.get('mode', None) == 'predict' and args['predict_file']:
        utils.ensure_dir(args['predict_dir'], verbose=False)
        pred_file = os.path.join(args['predict_dir'], args['predict_file'] + ".pred.mrg")
        orig_file = os.path.join(args['predict_dir'], args['predict_file'] + ".orig.mrg")
        if os.path.exists(pred_file):
            tlogger.warning("Cowardly refusing to overwrite {}".format(pred_file))
        elif os.path.exists(orig_file):
            tlogger.warning("Cowardly refusing to overwrite {}".format(orig_file))
        else:
            with open(pred_file, 'w') as fout:
                for tree in full_results:
                    output_tree = tree.predictions[0].tree
                    if args['predict_output_gold_tags']:
                        output_tree = output_tree.replace_tags(tree.gold)
                    fout.write(args['predict_format'].format(output_tree))
                    fout.write("\n")

            for i in range(num_generate):
                pred_file = os.path.join(args['predict_dir'], args['predict_file'] + ".%03d.pred.mrg" % i)
                with open(pred_file, 'w') as fout:
                    for tree in generated_treebanks[-(i+1)]:
                        output_tree = tree.predictions[0].tree
                        if args['predict_output_gold_tags']:
                            output_tree = output_tree.replace_tags(tree.gold)
                        fout.write(args['predict_format'].format(output_tree))
                        fout.write("\n")

            with open(orig_file, 'w') as fout:
                for tree in full_results:
                    fout.write(args['predict_format'].format(tree.gold))
                    fout.write("\n")

    if len(full_results) == 0:
        return 0.0, 0.0
    if evaluator is None:
        if num_generate > 0:
            kbest = max(len(fr.predictions) for fr in full_results)
        else:
            kbest = None
        with EvaluateParser(kbest=kbest) as evaluator:
            response = evaluator.process(full_results)
    else:
        response = evaluator.process(full_results)

    kbestF1 = response.kbestF1 if response.HasField("kbestF1") else None
    return response.f1, kbestF1, response.treeF1
