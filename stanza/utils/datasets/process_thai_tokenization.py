import os
import random

def write_section(output_dir, dataset_name, section, documents):
    """
    Writes a list of documents for tokenization, including a file in conll format

    The Thai datasets generally have no MWT (apparently not relevant for Thai)

    output_dir: the destination directory for the output files
    dataset_name: orchid, BEST, lst20, etc
    section: train/dev/test
    documents: a nested list of documents, paragraphs, sentences, words
      words is a list of (word, space_follows)
    """
    with open(os.path.join(output_dir, 'th_%s-ud-%s-mwt.json' % (dataset_name, section)), 'w') as fout:
        fout.write("[]\n")

    text_out = open(os.path.join(output_dir, 'th_%s.%s.txt' % (dataset_name, section)), 'w')
    label_out = open(os.path.join(output_dir, 'th_%s-ud-%s.toklabels' % (dataset_name, section)), 'w')
    for document in documents:
        for paragraph in document:
            for sentence_idx, sentence in enumerate(paragraph):
                for word_idx, word in enumerate(sentence):
                    # TODO: split with newlines to make it more readable?
                    text_out.write(word[0])
                    for i in range(len(word[0]) - 1):
                        label_out.write("0")
                    if word_idx == len(sentence) - 1:
                        label_out.write("2")
                    else:
                        label_out.write("1")
                    if word[1] and sentence_idx != len(paragraph) - 1:
                        text_out.write(' ')
                        label_out.write('0')

            text_out.write("\n\n")
            label_out.write("\n\n")

    text_out.close()
    label_out.close()

    with open(os.path.join(output_dir, 'th_%s.%s.gold.conllu' % (dataset_name, section)), 'w') as fout:
        for document in documents:
            for paragraph in document:
                for sentence in paragraph:
                    for word_idx, word in enumerate(sentence):
                        # SpaceAfter is left blank if there is space after the word
                        space = '_' if word[1] else 'SpaceAfter=No'
                        # Note the faked dependency structure: the conll reading code
                        # needs it even if it isn't being used in any way
                        fake_dep = 'root' if word_idx == 0 else 'dep'
                        fout.write('{}\t{}\t_\t_\t_\t_\t{}\t{}\t{}:{}\t{}\n'.format(word_idx+1, word[0], word_idx, fake_dep, word_idx, fake_dep, space))
                    fout.write('\n')

def write_dataset(documents, output_dir, dataset_name):
    """
    Shuffle a list of documents, write three sections
    """
    random.shuffle(documents)
    num_train = int(len(documents) * 0.8)
    num_dev = int(len(documents) * 0.1)
    os.makedirs(output_dir, exist_ok=True)
    write_section(output_dir, dataset_name, 'train', documents[:num_train])
    write_section(output_dir, dataset_name, 'dev', documents[num_train:num_train+num_dev])
    write_section(output_dir, dataset_name, 'test', documents[num_train+num_dev:])
