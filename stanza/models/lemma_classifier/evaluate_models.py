# TODO: Figure out how to load in the UD files into Stanza objects to get the features from them.

import stanza
from stanza.stanza.utils import CoNLL 

print(CoNLL.conll2doc("/u/scr/corpora/Universal_Dependencies/Universal_Dependencies_2.12/ud-treebanks-v2.12/UD_English-GUM/en_gum-ud-test.conllu"))