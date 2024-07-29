from collections import defaultdict
import os
import re

from stanza.utils.conll import CoNLL
import stanza.utils.default_paths as default_paths
from stanza.utils.datasets.ner.utils import write_dataset

def output_entities(sentence):
    for word in sentence.words:
        misc = word.misc
        if misc is None:
            continue

        pieces = misc.split("|")
        for piece in pieces:
            if piece.startswith("Entity="):
                entity = piece.split("=", maxsplit=1)[1]
                print("  " + entity)
                break

def extract_single_sentence(sentence):
    current_entity = []
    words = []
    for word in sentence.words:
        text = word.text
        misc = word.misc
        if misc is None:
            pieces = []
        else:
            pieces = misc.split("|")

        closes = []
        first_entity = False
        for piece in pieces:
            if piece.startswith("Entity="):
                entity = piece.split("=", maxsplit=1)[1]
                entity_pieces = re.split(r"([()])", entity)
                entity_pieces = [x for x in entity_pieces if x]   # remove blanks from re.split
                entity_idx = 0
                while entity_idx < len(entity_pieces):
                    if entity_pieces[entity_idx] == '(':
                        assert len(entity_pieces) > entity_idx + 1, "Opening an unspecified entity"
                        if len(current_entity) == 0:
                            first_entity = True
                        current_entity.append(entity_pieces[entity_idx + 1])
                        entity_idx += 2
                    elif entity_pieces[entity_idx] == ')':
                        assert entity_idx != 0, "Closing an unspecified entity"
                        closes.append(entity_pieces[entity_idx-1])
                        entity_idx += 1
                    else:
                        # the entities themselves get added or removed via the ()
                        entity_idx += 1

        if len(current_entity) == 0:
            entity = 'O'
        else:
            entity = current_entity[0]
            entity = "B-" + entity if first_entity else "I-" + entity
        words.append((text, entity))

        assert len(current_entity) >= len(closes), "Too many closes for the current open entities"
        for close_entity in closes:
            # TODO: check the close is closing the right thing
            assert close_entity == current_entity[-1], "Closed the wrong entity: %s vs %s" % (close_entity, current_entity[-1])
            current_entity = current_entity[:-1]
    return words

def extract_sentences(doc):
    sentences = []
    for sentence in doc.sentences:
        try:
            words = extract_single_sentence(sentence)
            sentences.append(words)
        except AssertionError as e:
            print("Skipping sentence %s  ... %s" % (sentence.sent_id, str(e)))
            output_entities(sentence)

    return sentences

def convert_iahlt(udbase, output_dir, short_name):
    shards = ("train", "dev", "test")
    ud_datasets = ["UD_Hebrew-IAHLTwiki", "UD_Hebrew-IAHLTknesset"]
    base_filenames = ["he_iahltwiki-ud-%s.conllu", "he_iahltknesset-ud-%s.conllu"]
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
