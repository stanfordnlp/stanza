import sys

from collections import Counter

from stanza.models.constituency import parse_tree
from stanza.models.constituency import tree_reader

word_counter = Counter()
count_words = lambda x: word_counter.update(x.leaf_labels())

tree_reader.read_tree_file(sys.argv[1], tree_callback=count_words)
print(word_counter.most_common()[:100])
