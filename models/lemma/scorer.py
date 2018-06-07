"""
Utils and wrappers for scoring lemmatizers.
"""
from utils import conll18_ud_eval as ud_eval

def write_to_conllu(input_sents, pred_tokens, filename):
    """ Write predictions to a conllu file, only for eval purpose."""
    assert len(sum(input_sents, [])) == len(pred_tokens), \
            "Num of pred tokens does not match num of input tokens"
    idx = 0
    with open(filename, 'w') as outfile:
        for sent in input_sents:
            i = 0
            for tok in sent:
                tok = tok[0]
                lem = pred_tokens[idx]
                outfile.write('{}\t{}\t{}{}\t{}{}\n'.format(i+1, tok, lem, '\t_'*3, i, '\t_'*3))
                i += 1
                idx += 1
            outfile.write('\n')
    return

def score(system_conllu_file, gold_conllu_file):
    """ Wrapper for lemma scorer. """
    gold_ud = ud_eval.load_conllu_file(gold_conllu_file)
    system_ud = ud_eval.load_conllu_file(system_conllu_file)
    evaluation = ud_eval.evaluate(gold_ud, system_ud)
    el = evaluation["Lemmas"]
    p, r, f = el.precision, el.recall, el.f1
    return p, r, f

