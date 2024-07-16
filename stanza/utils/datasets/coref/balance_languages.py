"""
balance_concat.py
create a test set from a dev set which is language balanced
"""

import json
from collections import defaultdict

from random import Random

# fix random seed for reproducability
R = Random(42)

with open("./corefud_concat_v1_0_langid.train.json", 'r') as df:
    raw = json.load(df)

# calculate type of each class; then, we will select the one
# which has the LOWEST counts as the sample rate
lang_counts = defaultdict(int)
for i in raw:
    lang_counts[i["lang"]] += 1

min_lang_count = min(lang_counts.values())

# sample 20% of the smallest amount for test set
# this will look like an absurdly small number, but
# remember this is DOCUMENTS not TOKENS or UTTERANCES
# so its actually decent
# also its per language
test_set_size = int(0.1*min_lang_count)

# sampling input by language
raw_by_language = defaultdict(list)
for i in raw:
    raw_by_language[i["lang"]].append(i)
languages = list(set(raw_by_language.keys()))

train_set = []
test_set = []
for i in languages:
    length = list(range(len(raw_by_language[i])))
    choices = R.sample(length, test_set_size)

    for indx,i in enumerate(raw_by_language[i]):
        if indx in choices:
            test_set.append(i)
        else:
            train_set.append(i)

with open("./corefud_concat_v1_0_langid-bal.train.json", 'w') as df:
    json.dump(train_set, df, indent=2)

with open("./corefud_concat_v1_0_langid-bal.test.json", 'w') as df:
    json.dump(test_set, df, indent=2)



# raw_by_language["en"]


