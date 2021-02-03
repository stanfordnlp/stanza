import os

def get_default_paths():
    """
    Gets base paths for the data directories

    If DATA_ROOT is set in the environment, use that as the root
    otherwise use "./data"
    individual paths can also be set in the environment
    """
    DATA_ROOT = os.environ.get("DATA_ROOT", "data")
    defaults = {
        "TOKENIZE_DATA_DIR": DATA_ROOT + "/tokenize",
        "MWT_DATA_DIR": DATA_ROOT + "/mwt",
        "LEMMA_DATA_DIR": DATA_ROOT + "/lemma",
        "POS_DATA_DIR": DATA_ROOT + "/pos",
        "DEPPARSE_DATA_DIR": DATA_ROOT + "/depparse",
        "ETE_DATA_DIR": DATA_ROOT + "/ete",
        "NER_DATA_DIR": DATA_ROOT + "/ner",
        "CHARLM_DATA_DIR": DATA_ROOT + "/charlm",
        "SENTIMENT_DATA_DIR": DATA_ROOT + "/sentiment",

        # Set directories to store external word vector data
        "WORDVEC_DIR": "extern_data/wordvec",

        # TODO: not sure what other people actually have
        # TODO: also, could make this automatically update to the latest
        "UDBASE": "extern_data/ud2/ud-treebanks-v2.7",

        "NERBASE": "extern_data/ner",

        # there's a stanford github, stanfordnlp/handparsed-treebank,
        # with some data for different languages
        "HANDPARSED_DIR": "extern_data/handparsed-treebank",
    }

    paths = { "DATA_ROOT" : DATA_ROOT }
    for k, v in defaults.items():
        paths[k] = os.environ.get(k, v)

    return paths
