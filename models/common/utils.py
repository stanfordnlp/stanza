import utils.conll18_ud_eval as ud_eval

def ud_scores(gold_conllu_file, system_conllu_file):
    gold_ud = ud_eval.load_conllu_file(gold_conllu_file)
    system_ud = ud_eval.load_conllu_file(system_conllu_file)
    evaluation = ud_eval.evaluate(gold_ud, system_ud)

    return evaluation

def harmonic_mean(a):
    if any([x == 0 for x in a]):
        return 0
    else:
        return len(a) / sum([1/x for x in a])
