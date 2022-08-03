import stanza
from stanza.server.semgrex import Semgrex

nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

doc = nlp("Banning opal removed all artifact decks from the meta.  I miss playing lantern.")
with Semgrex(classpath="$CLASSPATH") as sem:
    semgrex_results = sem.process(doc,
                                  "{pos:NN}=object <obl {}=action",
                                  "{cpos:NOUN}=thing <obj {cpos:VERB}=action")
    print("COMPLETE RESULTS")
    print(semgrex_results)

    print("Number of matches in graph 0 ('Banning opal...') for semgrex query 1 (thing <obj action): %d" % len(semgrex_results.result[0].result[1].match))
    for match_idx, match in enumerate(semgrex_results.result[0].result[1].match):
        print("Match {}:\n-----------\n{}".format(match_idx, match))

    print("graph 1 for semgrex query 0 is an empty match: len %d" % len(semgrex_results.result[1].result[0].match))
