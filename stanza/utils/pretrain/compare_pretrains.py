import sys
import numpy as np

from stanza.models.common.pretrain import Pretrain

pt1_filename = sys.argv[1]
pt2_filename = sys.argv[2]

pt1 = Pretrain(pt1_filename)
pt2 = Pretrain(pt2_filename)

vocab1 = pt1.vocab
vocab2 = pt2.vocab

common_words = [x for x in vocab1 if x in vocab2]
print("%d shared words, out of %d in %s and %d in %s" % (len(common_words), len(vocab1), pt1_filename, len(vocab2), pt2_filename))

eps = 0.0001
total_norm = 0.0
total_close = 0

words_different = []

for word, idx in vocab1._unit2id.items():
    if word not in vocab2:
        continue
    v1 = pt1.emb[idx]
    v2 = pt2.emb[pt2.vocab[word]]
    norm = np.linalg.norm(v1 - v2)

    if norm < eps:
        total_close += 1
    else:
        total_norm += norm
        if len(words_different) < 10:
            words_different.append("|%s|" % word)
            #print(word, idx, pt2.vocab[word])
            #print(v1)
            #print(v2)

if total_close < len(common_words):
    avg_norm = total_norm / (len(common_words) - total_close)
    print("%d vectors were close.  Average difference of the others: %f" % (total_close, avg_norm))
    print("The first few different words were:\n  %s" % "\n  ".join(words_different))
else:
    print("All %d vectors were close!" % total_close)

    for word, idx in vocab1._unit2id.items():
        if word not in vocab2:
            continue
        if pt2.vocab[word] != idx:
            break
    else:
        print("All indices are the same")
