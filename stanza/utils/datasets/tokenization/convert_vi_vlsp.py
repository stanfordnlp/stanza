
import os

def find_spaces(sentence):
    # TODO: there are some sentences where there is only one quote,
    # and some of them should be attached to the previous word instead
    # of the next word.  Training should work this way, though
    odd_quotes = False

    spaces = []
    for word_idx, word in enumerate(sentence):
        space = True
        if word_idx < len(sentence) - 1:
            if sentence[word_idx+1] in (',', '.', '!', '?', ')', ':', ';', '”', '…', '...'):
                space = False
        if word in ('(', '“'):
            space = False
        if word == '"':
            if odd_quotes:
                # already saw one quote.  put this one at the end of the PREVIOUS word
                # note that we know there must be at least one word already
                odd_quotes = False
                spaces[word_idx-1] = False
            else:
                odd_quotes = True
                space = False
        spaces.append(space)
    return spaces

def write_file(vlsp_include_spaces, output_filename, sentences, shard):
    with open(output_filename, "w") as fout:
        for sent_idx, sentence in enumerate(sentences):
            fout.write("# sent_id = %s.%d\n" % (shard, sent_idx))
            orig_text = " ".join(sentence)
            if vlsp_include_spaces:
                fout.write("# text = %s\n" % orig_text)
            else:
                spaces = find_spaces(sentence)
                full_text = ""
                for word, space in zip(sentence, spaces):
                    # could be made more efficient, but shouldn't matter
                    full_text = full_text + word
                    if space:
                        full_text = full_text + " "
                fout.write("# text = %s\n" % full_text)
                fout.write("# orig_text = %s\n" % orig_text)
            for word_idx, word in enumerate(sentence):
                fake_dep = "root" if word_idx == 0 else "dep"
                fout.write("%d\t%s\t%s" % ((word_idx+1), word, word))
                fout.write("\t_\t_\t_")
                fout.write("\t%d\t%s" % (word_idx, fake_dep))
                fout.write("\t_\t")
                if vlsp_include_spaces or spaces[word_idx]:
                    fout.write("_")
                else:
                    fout.write("SpaceAfter=No")
                fout.write("\n")
            fout.write("\n")

def convert_file(vlsp_include_spaces, input_filename, output_filename, shard, split_filename=None, split_shard=None):
    with open(input_filename) as fin:
        lines = fin.readlines()

    sentences = []
    for line in lines:
        words = line.split()
        words = [w.replace("_", " ") for w in words]
        sentences.append(words)

    if split_filename is not None:
        # even this is a larger dev set than the train set
        split_point = int(len(sentences) * 0.95)
        write_file(vlsp_include_spaces, output_filename, sentences[:split_point], shard)
        write_file(vlsp_include_spaces, split_filename, sentences[split_point:], split_shard)
    else:
        write_file(vlsp_include_spaces, output_filename, sentences, shard)

def convert_vi_vlsp(extern_dir, tokenizer_dir, args):
    input_path = os.path.join(extern_dir, "vietnamese", "VLSP2013-WS-data")

    input_train_filename = os.path.join(input_path, "VLSP2013_WS_train_gold.txt")
    input_test_filename = os.path.join(input_path, "VLSP2013_WS_test_gold.txt")
    if not os.path.exists(input_train_filename):
        raise FileNotFoundError("Cannot find train set for VLSP at %s" % input_train_filename)
    if not os.path.exists(input_test_filename):
        raise FileNotFoundError("Cannot find test set for VLSP at %s" % input_test_filename)

    output_train_filename = os.path.join(tokenizer_dir, "vi_vlsp.train.gold.conllu")
    output_dev_filename = os.path.join(tokenizer_dir,   "vi_vlsp.dev.gold.conllu")
    output_test_filename = os.path.join(tokenizer_dir,  "vi_vlsp.test.gold.conllu")

    convert_file(args.vlsp_include_spaces, input_train_filename, output_train_filename, "train", output_dev_filename, "dev")
    convert_file(args.vlsp_include_spaces, input_test_filename, output_test_filename, "test")

