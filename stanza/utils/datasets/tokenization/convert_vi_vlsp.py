
import os

punctuation_set = (',', '.', '!', '?', ')', ':', ';', '”', '…', '...')

def find_spaces(sentence):
    # TODO: there are some sentences where there is only one quote,
    # and some of them should be attached to the previous word instead
    # of the next word.  Training should work this way, though
    odd_quotes = False

    spaces = []
    for word_idx, word in enumerate(sentence):
        space = True
        # Quote period at the end of a sentence needs to be attached
        # to the rest of the text.  Some sentences have `"... text`
        # in the middle, though, so look for that
        if word_idx < len(sentence) - 2 and sentence[word_idx+1] == '"':
            if sentence[word_idx+2] == '.':
                space = False
            elif word_idx == len(sentence) - 3 and sentence[word_idx+2] == '...':
                space = False
        if word_idx < len(sentence) - 1:
            if sentence[word_idx+1] in (',', '.', '!', '?', ')', ':', ';', '”', '…', '...','/', '%'):
                space = False
        if word in ('(', '“', '/'):
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

def add_vlsp_args(parser):
    parser.add_argument('--include_pos_data', action='store_true', default=False, help='To include or not POS training dataset for tokenization training. The path to POS dataset is expected to be in the same dir with WS path. For example, extern_dir/vietnamese/VLSP2013-POS-data')
    parser.add_argument('--vlsp_include_spaces', action='store_true', default=False, help='When processing vi_vlsp tokenization, include all of the spaces.  Otherwise, we try to turn the text back into standard text')


def write_file(vlsp_include_spaces, output_filename, sentences, shard):
    with open(output_filename, "w") as fout:
        check_headlines = False
        for sent_idx, sentence in enumerate(sentences):
            fout.write("# sent_id = %s.%d\n" % (shard, sent_idx))
            orig_text = " ".join(sentence)
            #check if the previous line is a headline (no ending mark at the end) then make this sentence a new par
            if check_headlines:
                fout.write("# newpar id =%s.%d.1\n" % (shard, sent_idx))
                check_headlines = False
            if sentence[len(sentence) - 1] not in punctuation_set:
                check_headlines = True

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

def convert_pos_dataset(file_path):
    """
    This function is to process the pos dataset
    """
    
    file = open(file_path, "r")
    document = file.readlines()
    sentences = []
    sent = []
    for line in document:
        if line == "\n" and len(sent)>1:
            if sent not in sentences:
                sentences.append(sent)
            sent = []
        elif line != "\n":
            sent.append(line.split("\t")[0].replace("_"," ").strip())
    return sentences
        
def convert_file(vlsp_include_spaces, input_filename, output_filename, shard, split_filename=None, split_shard=None, pos_data = None):
    with open(input_filename) as fin:
        lines = fin.readlines()

    sentences = []
    set_sentences = set()
    for line in lines:
        if len(line.replace("_", " ").split())>1:
            words = line.split()
            #one syllable lines are eliminated
            if len(words) == 1 and len(words[0].split("_")) == 1:
                continue
            else:
                words = [w.replace("_", " ") for w in words]
                #only add sentences that hasn't been added before
                if words not in sentences:
                    sentences.append(words)
                    set_sentences.add(' '.join(words))
                
    if split_filename is not None:
        # even this is a larger dev set than the train set
        split_point = int(len(sentences) * 0.95)
        #check pos_data that aren't overlapping with current VLSP WS dataset
        sentences_pos = [] if pos_data is None else [sent for sent in pos_data if ' '.join(sent) not in set_sentences]
        print("Added ", len(sentences_pos), " sentences from POS dataset.")
        write_file(vlsp_include_spaces, output_filename, sentences[:split_point]+sentences_pos, shard)
        write_file(vlsp_include_spaces, split_filename, sentences[split_point:], split_shard)
    else:
        write_file(vlsp_include_spaces, output_filename, sentences, shard)

def convert_vi_vlsp(extern_dir, tokenizer_dir, args):
    input_path = os.path.join(extern_dir, "vietnamese", "VLSP2013-WS-data")
    input_pos_path = os.path.join(extern_dir, "vietnamese", "VLSP2013-POS-data")
    input_train_filename = os.path.join(input_path, "VLSP2013_WS_train_gold.txt")
    input_test_filename = os.path.join(input_path, "VLSP2013_WS_test_gold.txt")
    
    input_pos_filename = os.path.join(input_pos_path, "VLSP2013_POS_train_BI_POS_Column.txt.goldSeg")
    if not os.path.exists(input_train_filename):
        raise FileNotFoundError("Cannot find train set for VLSP at %s" % input_train_filename)
    if not os.path.exists(input_test_filename):
        raise FileNotFoundError("Cannot find test set for VLSP at %s" % input_test_filename)
    pos_data = None
    if args.include_pos_data:
        if not os.path.exists(input_pos_filename):
            raise FileNotFoundError("Cannot find pos dataset for VLSP at %" % input_pos_filename)
        else:
            pos_data = convert_pos_dataset(input_pos_filename) 

    output_train_filename = os.path.join(tokenizer_dir, "vi_vlsp.train.gold.conllu")
    output_dev_filename = os.path.join(tokenizer_dir,   "vi_vlsp.dev.gold.conllu")
    output_test_filename = os.path.join(tokenizer_dir,  "vi_vlsp.test.gold.conllu")

    convert_file(args.vlsp_include_spaces, input_train_filename, output_train_filename, "train", output_dev_filename, "dev", pos_data)
    convert_file(args.vlsp_include_spaces, input_test_filename, output_test_filename, "test")

