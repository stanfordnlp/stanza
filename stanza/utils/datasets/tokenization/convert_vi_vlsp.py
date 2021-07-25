
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
    parser.add_argument('--include_pos_data', action='store_true', default=False, help='To include or not POS dataset for tokenization training.')
    parser.add_argument('--vlsp_include_spaces', action='store_true', default=False, help='When processing vi_vlsp tokenization, include all of the spaces.  Otherwise, we try to turn the text back into standard text')
    parser.add_argument('--check_overlap', type=str, default="", help='Check overlap between two dataseet, input name of file 1')
    #parser.add_argument('--check_overlap_2', type=str, default="", help='Check overlap between two dataseet, input name of file 1')
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
    file = open(file_path, "r")
    document = file.readlines()
    sentences = []
    sent = []
    for line in document:
        if line == "\n" and len(sent)>1:
            sentences.append(sent)
            sent = []
        elif line != "\n":
            sent.append(line.split("\t")[0].replace("_"," ").strip())
    return sentences
        

def convert_file(vlsp_include_spaces, input_filename, output_filename, shard, split_filename=None, split_shard=None, pos_data = []):
    with open(input_filename) as fin:
        lines = fin.readlines()

    sentences = []
    for line in lines:
        if len(line.replace("_", " ").split())>1:
            words = line.split()
            #one syllable lines are eliminated
            if len(words) == 1 and len(words[0].split("_")) == 1:
                continue
            else:
                words = [w.replace("_", " ") for w in words]
                if words not in sentences:
                    sentences.append(words)
    
    set_sent = []
    another_set = []
    for sent in sentences:
        if sent not in set_sent:
            set_sent.append(sent)
        elif sent in set_sent:
            another_set.append(sent)
    print(another_set[:5])
    #set_sent.sort()
    #set_sent = list(k for k,_ in itertools.groupby(set_sent))
    
    #split_point = int(len(sentences) * 0.95)
    #copy = sentences.copy()
    #set_sent = [sent for sent in sentences if sent not in ]
    print("There are ", len(sentences)-len(set_sent), " overlapping sentences in botb training and dev sets of VLSP WS.")
    if split_filename is not None:
        # even this is a larger dev set than the train set
        split_point = int(len(sentences) * 0.95)
        sentences_pos = [sent for sent in pos_data if sent not in sentences]
        print("Eliminated ", len(pos_data)-len(sentences_pos), " sentences from VLSP POS dataset that are overlapping with VLSSP WS train and dev sets.")
        #inter_1 = [sent for sent in pos_data if sent in sentences[split_point:]]
        #print("Dev vlsp seg dataset has ", len(inter_1),"/",len(sentences[split_point:]), " sentences in common with the file you're checking.")
        write_file(vlsp_include_spaces, output_filename, sentences[:split_point]+sentences_pos, shard)
        write_file(vlsp_include_spaces, split_filename, sentences[split_point:], split_shard)
    else:
        #inter_1 = [sent for sent in pos_data if sent in sentences]
        #print("Test vlsp seg dataset has ", len(inter_1), "/",len(sentences)," sentences in common with the file you're checking.")
        write_file(vlsp_include_spaces, output_filename, sentences, shard)

    #check overlap between training seg and file
    #inter_1 = [sent for sent in pos_data if sent in sentences[:split_point]]
    #print("Training vlsp seg dataset has ", len(inter_1), " sentences in common with the file you're checking.")
    #inter_1 = [sent for sent in pos_data if sent in sentences[split_point:]]
    #print("Dev vlsp seg dataset has ", len(inter_1), " sentences in common with the file you're checking.")
    #inter_1 = [sent for sent in pos_data if sent in sentences]
    #print("Test vlsp seg dataset has ", len(inter_1), " sentences in common with the file you're checking.")
def convert_vi_vlsp(extern_dir, tokenizer_dir, args):
    input_path = os.path.join(extern_dir, "vietnamese", "VLSP2013-WS-data")

    input_train_filename = os.path.join(input_path, "VLSP2013_WS_train_gold.txt")
    input_test_filename = os.path.join(input_path, "VLSP2013_WS_test_gold.txt")
    
    input_pos_filename = os.path.join(input_path, "VLSP2013_POS_%s_BI_POS_Column.txt.goldSeg"%args.check_overlap)
    if not os.path.exists(input_train_filename):
        raise FileNotFoundError("Cannot find train set for VLSP at %s" % input_train_filename)
    if not os.path.exists(input_test_filename):
        raise FileNotFoundError("Cannot find test set for VLSP at %s" % input_test_filename)
    pos_data = []
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

