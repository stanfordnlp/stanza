import logging
import stanza.models.classifiers.classifier_args as classifier_args

logger = logging.getLogger('stanza')

def update_text(sentence, wordvec_type):
    """
    Process a line of text (with tokenization provided as whitespace)
    into a list of strings.
    """
    # stanford sentiment dataset has a lot of random - and /
    sentence = sentence.replace("-", " ")
    sentence = sentence.replace("/", " ")
    sentence = sentence.split()
    # our current word vectors are all entirely lowercased
    sentence = [word.lower() for word in sentence]
    if wordvec_type == classifier_args.WVType.WORD2VEC:
        return sentence
    elif wordvec_type == classifier_args.WVType.GOOGLE:
        new_sentence = []
        for word in sentence:
            if word != '0' and word != '1':
                word = re.sub('[0-9]', '#', word)
            new_sentence.append(word)
        return new_sentence
    elif wordvec_type == classifier_args.WVType.FASTTEXT:
        return sentence
    elif wordvec_type == classifier_args.WVType.OTHER:
        return sentence
    else:
        raise ValueError("Unknown wordvec_type {}".format(wordvec_type))


def read_dataset(dataset, wordvec_type, min_len):
    """
    returns a list where the values of the list are
      label, [token...]
    """
    lines = []
    for filename in dataset.split(","):
        try:
            new_lines = open(filename, encoding="utf-8").readlines()
        except UnicodeDecodeError:
            logger.error("Could not read {}".format(filename))
            raise
        lines.extend(new_lines)
    lines = [x.strip() for x in lines]
    lines = [x.split(maxsplit=1) for x in lines if x]
    lines = [x for x in lines if len(x) > 1]
    # TODO: maybe do this processing later, once the model is built.
    # then move the processing into the model so we can use
    # overloading to potentially make future model types
    lines = [(x[0], update_text(x[1], wordvec_type)) for x in lines]
    if min_len:
        lines = [x for x in lines if len(x[1]) >= min_len]
    return lines

