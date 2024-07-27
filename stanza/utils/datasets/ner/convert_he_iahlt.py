from collections import defaultdict
import os

from stanza.utils.conll import CoNLL
import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.ner.utils import write_dataset

def extract_sentences(doc):
    sentences = []
    for sentence in doc.sentences:
        current_entity = "O"
        words = []
        for word in sentence.words:
            text = word.text
            misc = word.misc
            if misc is None:
                if current_entity == 'O':
                    entity = 'O'
                else:
                    entity = "I-" + current_entity
                words.append((text, entity))
                continue
            pieces = misc.split("|")
            for piece in pieces:
                if piece.startswith("Entity="):
                    entity = piece.split("=", maxsplit=1)[1]
                    if entity.startswith("(") and entity.endswith(")"):
                        assert current_entity == 'O'
                        entity = "B-" + entity[1:-1]
                    elif entity.startswith("("):
                        assert current_entity == 'O'
                        entity = entity[1:]
                        current_entity = entity
                        entity = "B-" + entity
                    elif entity.endswith(")"):
                        entity = entity[:-1]
                        assert current_entity == entity
                        entity = "I-" + entity
                        current_entity = "O"
                    else:
                        assert current_entity == entity
                        entity = "I-" + entity
                    words.append((text, entity))
                    break
            else: # closes for loop
                if current_entity == 'O':
                    entity = 'O'
                else:
                    entity = "I-" + current_entity
                words.append((text, entity))
        sentences.append(words)

    return sentences

def convert_iahlt(udbase, output_dir, short_name):
    shards = ("train", "dev", "test")
    ud_datasets = ["UD_Hebrew-IAHLTknesset"]
    base_filenames = ["he_iahltknesset-ud-%s.conllu"]
    datasets = defaultdict(list)

    for ud_dataset, base_filename in zip(ud_datasets, base_filenames):
        ud_dataset_path = os.path.join(udbase, ud_dataset)
        for shard in shards:
            filename = os.path.join(ud_dataset_path, base_filename % shard)
            doc = CoNLL.conll2doc(filename)
            sentences = extract_sentences(doc)
            print("Read %d sentences from %s" % (len(sentences), filename))
            datasets[shard].extend(sentences)

    datasets = [datasets[x] for x in shards]
    write_dataset(datasets, output_dir, short_name)

def main():
    paths = default_paths.get_default_paths()

    udbase = paths["UDBASE_GIT"]
    output_directory = paths["NER_DATA_DIR"]
    convert_iahlt(udbase, output_directory, "he_iahlt")

if __name__ == '__main__':
    main()
