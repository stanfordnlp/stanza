# TODO: Figure out how to load in the UD files into Stanza objects to get the features from them.

import stanza

doc = stanza.utils.CoNLL.conll2doc("/u/scr/corpora/Universal_Dependencies/Universal_Dependencies_2.12/ud-treebanks-v2.12/UD_English-GUM/en_gum-ud-test.conllu")

print(doc.sentences)

