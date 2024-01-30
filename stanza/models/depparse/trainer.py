"""
A trainer class to handle training and testing of models.
"""

import copy
import sys
import logging
import torch
from torch import nn

from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.common import utils, loss
from stanza.models.common.foundation_cache import NoTransformerFoundationCache
from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from stanza.models.depparse.model import Parser
from stanza.models.pos.vocab import MultiVocab

logger = logging.getLogger('stanza')

def unpack_batch(batch, device):
    """ Unpack a batch from the data loader. """
    inputs = [b.to(device) if b is not None else None for b in batch[:11]]
    orig_idx = batch[11]
    word_orig_idx = batch[12]
    sentlens = batch[13]
    wordlens = batch[14]
    text = batch[15]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens, text

class Trainer(BaseTrainer):
    """ A trainer for training models. """
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None,
                 device=None, foundation_cache=None, ignore_model_config=False, reset_history=False):
        self.global_step = 0
        self.last_best_step = 0
        self.dev_score_history = []

        orig_args = copy.deepcopy(args)
        # whether the training is in primary or secondary stage
        # during FT (loading weights), etc., the training is considered to be in "secondary stage"
        # during this time, we (optionally) use a different set of optimizers than that during "primary stage".
        #
        # Regardless, we use TWO SETS of optimizers; once primary converges, we switch to secondary

        if model_file is not None:
            # load everything from file
            self.load(model_file, pretrain, args, foundation_cache, device)

            if reset_history:
                self.global_step = 0
                self.last_best_step = 0
                self.dev_score_history = []
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = Parser(args, vocab, emb_matrix=pretrain.emb if pretrain is not None else None)
            self.model = self.model.to(device)
            self.__init_optim()

        if ignore_model_config:
            self.args = orig_args

        if self.args.get('wandb'):
            import wandb
            # track gradients!
            wandb.watch(self.model, log_freq=4, log="all", log_graph=True)

    def __init_optim(self):
        if (self.args.get("second_stage", False) and self.args.get('second_optim')):
            self.optimizer = utils.get_optimizer(self.args['second_optim'], self.model,
                                                 self.args['second_lr'], betas=(0.9, self.args['beta2']), eps=1e-6,
                                                 bert_learning_rate=self.args.get('second_bert_learning_rate', 0.0))
        else:
            self.optimizer = utils.get_optimizer(self.args['optim'], self.model,
                                                self.args['lr'], betas=(0.9, self.args['beta2']),
                                                eps=1e-6, bert_learning_rate=self.args.get('bert_learning_rate', 0.0))

    def update(self, batch, eval=False):
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        device = next(self.model.parameters()).device
        inputs, orig_idx, word_orig_idx, sentlens, wordlens, text = unpack_batch(batch, device)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], sentlens)] # remove attachment for the root
        deprel_seqs = [self.vocab['deprel'].unmap([preds[1][i][j+1][h] for j, h in enumerate(hs)]) for i, hs in enumerate(head_seqs)]

        pred_tokens = [[[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(sentlens[i]-1)] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

    def save(self, filename, skip_modules=True, save_optimizer=False):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
                'model': model_state,
                'vocab': self.vocab.state_dict(),
                'config': self.args,
                'global_step': self.global_step,
                'last_best_step': self.last_best_step,
                'dev_score_history': self.dev_score_history,
                }

        if save_optimizer and self.optimizer is not None:
            params['optimizer_state_dict'] = self.optimizer.state_dict()

        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except BaseException:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename, pretrain, args=None, foundation_cache=None, device=None):
        """
        Load a model from file, with preloaded pretrain embeddings. Here we allow the pretrain to be None or a dummy input,
        and the actual use of pretrain embeddings will depend on the boolean config "pretrain" in the loaded args.
        """
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        if args is not None: self.args.update(args)
        # preserve old models which were created before transformers were added
        if 'bert_model' not in self.args:
            self.args['bert_model'] = None
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        # load model
        emb_matrix = None
        if self.args['pretrain'] and pretrain is not None: # we use pretrain only if args['pretrain'] == True and pretrain is not None
            emb_matrix = pretrain.emb
        if any(x.startswith("bert_model.") for x in checkpoint['model'].keys()):
            logger.debug("Model %s has a finetuned transformer.  Not using transformer cache to make sure the finetuned version of the transformer isn't accidentally used elsewhere", filename)
            foundation_cache = NoTransformerFoundationCache(foundation_cache)
        self.model = Parser(self.args, self.vocab, emb_matrix=emb_matrix, foundation_cache=foundation_cache)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        if device is not None:
            self.model = self.model.to(device)

        self.__init_optim()
        optim_state_dict = checkpoint.get("optimizer_state_dict")
        if optim_state_dict:
            self.optimizer.load_state_dict(optim_state_dict)

        self.global_step = checkpoint.get("global_step", 0)
        self.last_best_step = checkpoint.get("last_best_step", 0)
        self.dev_score_history = checkpoint.get("dev_score_history", list())
