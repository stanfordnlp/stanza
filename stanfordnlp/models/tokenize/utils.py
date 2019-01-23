from collections import Counter
from copy import copy
import json
import numpy as np

from stanfordnlp.models.common.utils import ud_scores, harmonic_mean

def load_mwt_dict(filename):
    if filename is not None:
        with open(filename, 'r') as f:
            mwt_dict0 = json.load(f)

        mwt_dict = dict()
        for item in mwt_dict0:
            (key, expansion), count = item

            if key not in mwt_dict or mwt_dict[key][1] < count:
                mwt_dict[key] = (expansion, count)

        return mwt_dict
    else:
        return

def print_sentence(sentence, f, mwt_dict=None):
    i = 0
    for tok, p in sentence:
        expansion = None
        if (p == 3 or p == 4) and mwt_dict is not None:
            # MWT found, (attempt to) expand it!
            if tok in mwt_dict:
                expansion = mwt_dict[tok][0]
            elif tok.lower() in mwt_dict:
                expansion = mwt_dict[tok.lower()][0]
        if expansion is not None:
            f.write("{}-{}\t{}{}\n".format(i+1, i+len(expansion), tok, "\t_" * 8))
            for etok in expansion:
                f.write("{}\t{}{}\t{}{}\n".format(i+1, etok, "\t_" * 4, i, "\t_" * 3))
                i += 1
        else:
            if len(tok) <= 0:
                continue
            f.write("{}\t{}{}\t{}{}\t{}\n".format(i+1, tok, "\t_" * 4, i, "\t_" * 2, "MWT=Yes" if p == 3 or p == 4 else "_"))
            i += 1
    f.write('\n')

def output_predictions(output_file, trainer, data_generator, vocab, mwt_dict, max_seqlen=1000):
    paragraphs = []
    for i, p in enumerate(data_generator.sentences):
        start = 0 if i == 0 else paragraphs[-1][2]
        length = sum([len(x) for x in p])
        paragraphs += [(i, start, start+length, length+1)] # para idx, start idx, end idx, length

    paragraphs = list(sorted(paragraphs, key=lambda x: x[3], reverse=True))

    all_preds = [None] * len(paragraphs)
    all_raw = [None] * len(paragraphs)

    eval_limit = max(3000, max_seqlen)

    batch_size = trainer.args['batch_size']
    batches = int((len(paragraphs) + batch_size - 1) / batch_size)

    t = 0
    for i in range(batches):
        batchparas = paragraphs[i * batch_size : (i + 1) * batch_size]
        offsets = [x[1] for x in batchparas]
        t += sum([x[3] for x in batchparas])

        batch = data_generator.next(eval_offsets=offsets)
        raw = batch[3]

        N = len(batch[3][0])
        if N <= eval_limit:
            pred = np.argmax(trainer.predict(batch), axis=2)
        else:
            idx = [0] * len(batchparas)
            Ns = [p[3] for p in batchparas]
            pred = [[] for _ in batchparas]
            while True:
                ens = [min(N - idx1, eval_limit) for idx1, N in zip(idx, Ns)]
                en = max(ens)
                batch1 = batch[0][:, :en], batch[1][:, :en], batch[2][:, :en], [x[:en] for x in batch[3]]
                pred1 = np.argmax(trainer.predict(batch1), axis=2)

                for j in range(len(batchparas)):
                    sentbreaks = np.where((pred1[j] == 2) + (pred1[j] == 4))[0]
                    if len(sentbreaks) <= 0 or idx[j] >= Ns[j] - eval_limit:
                        advance = ens[j]
                    else:
                        advance = np.max(sentbreaks) + 1

                    pred[j] += [pred1[j, :advance]]
                    idx[j] += advance

                if all([idx1 >= N for idx1, N in zip(idx, Ns)]):
                    break
                batch = data_generator.next(eval_offsets=[x+y for x, y in zip(idx, offsets)])

            pred = [np.concatenate(p, 0) for p in pred]

        for j, p in enumerate(batchparas):
            len1 = len([1 for x in raw[j] if x != '<PAD>'])
            if pred[j][len1-1] < 2:
                pred[j][len1-1] = 2
            elif pred[j][len1-1] > 2:
                pred[j][len1-1] = 4
            all_preds[p[0]] = pred[j][:len1]
            all_raw[p[0]] = raw[j]

    offset = 0
    oov_count = 0

    for j in range(len(paragraphs)):
        raw = all_raw[j]
        pred = all_preds[j]

        current_tok = ''
        current_sent = []

        for t, p in zip(raw, pred):
            if t == '<PAD>':
                break
            # hack la_ittb
            if trainer.args['shorthand'] == 'la_ittb' and t in [":", ";"]:
                p = 2
            offset += 1
            if vocab.unit2id(t) == vocab.unit2id('<UNK>'):
                oov_count += 1

            current_tok += t
            if p >= 1:
                tok = vocab.normalize_token(current_tok)
                assert '\t' not in tok, tok
                if len(tok) <= 0:
                    current_tok = ''
                    continue
                current_sent += [(tok, p)]
                current_tok = ''
                if p == 2 or p == 4:
                    print_sentence(current_sent, output_file, mwt_dict)
                    current_sent = []

        if len(current_tok):
            tok = vocab.normalize_token(current_tok)
            assert '\t' not in tok, tok
            if len(tok) > 0:
                current_sent += [(tok, 2)]

        if len(current_sent):
            print_sentence(current_sent, output_file, mwt_dict)

    return oov_count, offset, all_preds

def eval_model(args, trainer, batches, vocab, mwt_dict):
    with open(args['conll_file'], 'w') as conll_output:
        oov_count, N, all_preds = output_predictions(conll_output, trainer, batches, vocab, mwt_dict, args['max_seqlen'])

    all_preds = np.concatenate(all_preds, 0)
    labels = [y[1] for x in batches.data for y in x]
    counter = Counter(zip(all_preds, labels))

    def f1(pred, gold, mapping):
        pred = [mapping[p] for p in pred]
        gold = [mapping[g] for g in gold]

        lastp = -1; lastg = -1
        tp = 0; fp = 0; fn = 0
        for i, (p, g) in enumerate(zip(pred, gold)):
            if p == g > 0 and lastp == lastg:
                lastp = i
                lastg = i
                tp += 1
            elif p > 0 and g > 0:
                lastp = i
                lastg = i
                fp += 1
                fn += 1
            elif p > 0:
                # and g == 0
                lastp = i
                fp += 1
            elif g > 0:
                lastg = i
                fn += 1

        if tp == 0:
            return 0
        else:
            return 2 * tp / (2 * tp + fp + fn)

    f1tok = f1(all_preds, labels, {0:0, 1:1, 2:1, 3:1, 4:1})
    f1sent = f1(all_preds, labels, {0:0, 1:0, 2:1, 3:0, 4:1})
    f1mwt = f1(all_preds, labels, {0:0, 1:1, 2:1, 3:2, 4:2})
    print(args['shorthand'], f1tok, f1sent, f1mwt)
    return harmonic_mean([f1tok, f1sent, f1mwt], [1, 1, .01])

