"""
Loads NER models & separates out the word vectors to base & delta

The model will then be resaved without the base word vector,
greatly reducing the size of the model

This may be useful for any external users of stanza who have an NER
model they wish to reuse without retraining

If you know which pretrain was used to build an NER model, you can
provide that pretrain.  Otherwise, you can give a directory of
pretrains and the script will test each one.  In the latter case,
the name of the pretrain needs to look like lang_dataset_pretrain.pt
"""

import argparse
from collections import defaultdict
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from stanza import Pipeline
from stanza.models.common.constant import lang_to_langcode
from stanza.models.common.pretrain import Pretrain, PretrainedWordVocab
from stanza.models.common.vocab import PAD_ID, VOCAB_PREFIX
from stanza.models.ner.trainer import Trainer

logger = logging.getLogger('stanza')
logger.setLevel(logging.ERROR)

DEBUG = False
EPS = 0.0001

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='saved_models/ner', help='Where to find NER models (dir or filename)')
    parser.add_argument('--output_path', type=str, default='saved_models/shrunk', help='Where to write shrunk NER models (dir)')
    parser.add_argument('--pretrain_path', type=str, default='saved_models/pretrain', help='Where to find pretrains (dir or filename)')
    args = parser.parse_args()

    # get list of NER models to shrink
    if os.path.isdir(args.input_path):
        ner_model_dir = args.input_path
        ners = os.listdir(ner_model_dir)
        if len(ners) == 0:
            raise FileNotFoundError("No ner models found in {}".format(args.input_path))
    else:
        if not os.path.isfile(args.input_path):
            raise FileNotFoundError("No ner model found at path {}".format(args.input_path))
        ner_model_dir, ners = os.path.split(args.input_path)
        ners = [ners]

    # get map from language to candidate pretrains
    if os.path.isdir(args.pretrain_path):
        pt_model_dir = args.pretrain_path
        pretrains = os.listdir(pt_model_dir)
        lang_to_pretrain = defaultdict(list)
        for pt in pretrains:
            lang_to_pretrain[pt.split("_")[0]].append(pt)
    else:
        pt_model_dir, pretrains = os.path.split(pt_model_dir)
        pretrains = [pretrains]
        lang_to_pretrain = defaultdict(lambda: pretrains)

    # shrunk models will all go in this directory
    new_dir = args.output_path
    os.makedirs(new_dir, exist_ok=True)

    final_pretrains = []
    missing_pretrains = []
    no_finetune = []

    # for each model, go through the various pretrains
    # until we find one that works or none of them work
    for ner_model in ners:
        ner_path = os.path.join(ner_model_dir, ner_model)

        expected_ending = "_nertagger.pt"
        if not ner_model.endswith(expected_ending):
            raise ValueError("Unexpected name: {}".format(ner_model))
        short_name = ner_model[:-len(expected_ending)]
        lang, package = short_name.split("_", maxsplit=1)
        print("===============================================")
        print("Processing lang %s package %s" % (lang, package))

        # this may look funny - basically, the pipeline has machinery
        # to make sure the model has everything it needs to load,
        # including downloading other pieces if needed
        pipe = Pipeline(lang, processors="tokenize,ner", tokenize_pretokenized=True, package={"ner": package}, ner_model_path=ner_path)
        ner_processor = pipe.processors['ner']
        print("Loaded NER processor: {}".format(ner_processor))
        trainer = ner_processor.trainers[0]
        vocab = trainer.model.vocab
        word_vocab = vocab['word']
        num_vectors = trainer.model.word_emb.weight.shape[0]

        # sanity check, make sure the model loaded matches the
        # language from the model's filename
        lcode = lang_to_langcode(trainer.args['lang'])
        if lang != lcode and not (lcode == 'zh' and lang == 'zh-hans'):
            raise ValueError("lang not as expected: {} vs {} ({})".format(lang, trainer.args['lang'], lcode))

        ner_pretrains = sorted(set(lang_to_pretrain[lang] + lang_to_pretrain[lcode]))
        for pt_model in ner_pretrains:
            pt_path = os.path.join(pt_model_dir, pt_model)
            print("Attempting pretrain: {}".format(pt_path))
            pt = Pretrain(filename=pt_path)
            print("  pretrain shape:               {}".format(pt.emb.shape))
            print("  embedding in ner model shape: {}".format(trainer.model.word_emb.weight.shape))
            if pt.emb.shape[1] != trainer.model.word_emb.weight.shape[1]:
                print("  DIMENSION DOES NOT MATCH.  SKIPPING")
                continue
            N = min(pt.emb.shape[0], trainer.model.word_emb.weight.shape[0])
            if pt.emb.shape[0] != trainer.model.word_emb.weight.shape[0]:
                # If the vocab was exactly the same, that's a good
                # sign this pretrain was used, just with a different size
                # In such a case, we can reuse the rest of the pretrain
                # Minor issue: some vectors which were trained will be
                # lost in the case of |pt| < |model.word_emb|
                if all(word_vocab.id2unit(x) == word_vocab.id2unit(x) for x in range(N)):
                    print("  Attempting to use pt vectors to replace ner model's vectors")
                else:
                    print("  NUM VECTORS DO NOT MATCH.  WORDS DO NOT MATCH.  SKIPPING")
                    continue
                if pt.emb.shape[0] < trainer.model.word_emb.weight.shape[0]:
                    print("  WARNING: if any vectors beyond {} were fine tuned, that fine tuning will be lost".format(N))
            delta = trainer.model.word_emb.weight[:N, :] - torch.from_numpy(pt.emb).cuda()[:N, :]
            delta = delta.detach()
            delta_norms = torch.linalg.norm(delta, dim=1).cpu().numpy()
            if np.sum(delta_norms < 0) > 0:
                raise ValueError("This should not be - a norm was less than 0!")
            num_matching = np.sum(delta_norms < EPS)
            if num_matching > N / 2:
                print("  Accepted!  %d of %d vectors match for %s" % (num_matching, N, pt_path))
                if pt.emb.shape[0] != trainer.model.word_emb.weight.shape[0]:
                    print("  Setting model vocab to match the pretrain")
                    word_vocab = pt.vocab
                    vocab['word'] = word_vocab
                    trainer.args['word_emb_dim'] = pt.emb.shape[1]
                break
            else:
                print("  %d of %d vectors matched for %s - SKIPPING" % (num_matching, N, pt_path))
                vocab_same = sum(x in pt.vocab for x in word_vocab)
                print("  %d words were in both vocabs" % vocab_same)
                # this is expensive, and in practice doesn't happen,
                # but theoretically we might have missed a mostly matching pt
                # if the vocab had been scrambled
                if DEBUG:
                    rearranged_count = 0
                    for x in word_vocab:
                        if x not in pt.vocab:
                            continue
                        x_id = word_vocab.unit2id(x)
                        x_vec = trainer.model.word_emb.weight[x_id, :]
                        pt_id = pt.vocab.unit2id(x)
                        pt_vec = pt.emb[pt_id, :]
                        if (x_vec.detach().cpu() - pt_vec).norm() < EPS:
                            rearranged_count += 1
                    print("  %d vectors were close when ignoring id ordering" % rearranged_count)
        else:
            print("COULD NOT FIND A MATCHING PT: {}".format(ner_processor))
            missing_pretrains.append(ner_model)
            continue

        # build a delta vector & embedding
        assert 'delta' not in vocab.keys()
        delta_vectors = [delta[i].cpu() for i in range(4)]
        delta_vocab = []
        for i in range(4, len(delta_norms)):
            if delta_norms[i] > 0.0:
                delta_vocab.append(word_vocab.id2unit(i))
                delta_vectors.append(delta[i].cpu())

        trainer.model.unsaved_modules.append("word_emb")
        if len(delta_vocab) == 0:
            print("No vectors were changed!  Perhaps this model was trained without finetune.")
            no_finetune.append(ner_model)
        else:
            print("%d delta vocab" % len(delta_vocab))
            print("%d vectors in the delta set" % len(delta_vectors))
            delta_vectors = np.stack(delta_vectors)
            delta_vectors = torch.from_numpy(delta_vectors)
            assert delta_vectors.shape[0] == len(delta_vocab) + len(VOCAB_PREFIX)
            print(delta_vectors.shape)

            delta_vocab = PretrainedWordVocab(delta_vocab, lang=word_vocab.lang, lower=word_vocab.lower)
            vocab['delta'] = delta_vocab
            trainer.model.delta_emb = nn.Embedding(delta_vectors.shape[0], delta_vectors.shape[1], PAD_ID)
            trainer.model.delta_emb.weight.data.copy_(delta_vectors)

        new_path = os.path.join(new_dir, ner_model)
        trainer.save(new_path)

        final_pretrains.append((ner_model, pt_model))

    print()
    if len(final_pretrains) > 0:
        print("Final pretrain mappings:")
        for i in final_pretrains:
            print(i)
    if len(missing_pretrains) > 0:
        print("MISSING EMBEDDINGS:")
        for i in missing_pretrains:
            print(i)
    if len(no_finetune) > 0:
        print("NOT FINE TUNED:")
        for i in no_finetune:
            print(i)

if __name__ == '__main__':
    main()
