import os
import random

import stanza
from stanza.utils.datasets.sentiment import process_utils
import stanza.utils.default_paths as default_paths

def main(in_dir, out_dir, short_name):
    nlp = stanza.Pipeline("it", processors='tokenize')

    mapping = {
        ('0', '0'): "1", # neither negative nor positive: neutral
        ('1', '0'): "2", # positive, not negative: positive
        ('0', '1'): "0", # negative, not positive: negative
        ('1', '1'): "1", # mixed?  neutral?  not sure what to do with this
    }

    test_filename = os.path.join(in_dir, "test_set_sentipolc16_gold2000.csv")
    test_snippets = process_utils.read_snippets(test_filename, (2,3), 8, "it", mapping, delimiter=",", skip_first_line=False, quotechar='"', nlp=nlp)

    train_filename = os.path.join(in_dir, "training_set_sentipolc16.csv")
    train_snippets = process_utils.read_snippets(train_filename, (2,3), 8, "it", mapping, delimiter=",", skip_first_line=True, quotechar='"', nlp=nlp)

    random.shuffle(train_snippets)
    dev_len = len(train_snippets) // 10
    dev_snippets = train_snippets[:dev_len]
    train_snippets = train_snippets[dev_len:]

    dataset = (train_snippets, dev_snippets, test_snippets)

    process_utils.write_dataset(dataset, out_dir, short_name)

if __name__ == '__main__':
    paths = default_paths.get_default_paths()
    random.seed(1234)

    in_directory = os.path.join(paths['SENTIMENT_BASE'], "italian", "sentipolc16")
    out_directory = paths['SENTIMENT_DATA_DIR']
    main(in_directory, out_directory, "it_sentipolc16")
