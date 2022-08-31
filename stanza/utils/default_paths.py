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
        "CONSTITUENCY_DATA_DIR": DATA_ROOT + "/constituency",

        # Set directories to store external word vector data
        "WORDVEC_DIR": "extern_data/wordvec",

        # TODO: not sure what other people actually have
        # TODO: also, could make this automatically update to the latest
        "UDBASE": "extern_data/ud2/ud-treebanks-v2.10",
        "UDBASE_GIT": "extern_data/ud2/git",

        "NERBASE": "extern_data/ner",
        "CONSTITUENCY_BASE": "extern_data/constituency",
        "SENTIMENT_BASE": "extern_data/sentiment",

        # there's a stanford github, stanfordnlp/handparsed-treebank,
        # with some data for different languages
        "HANDPARSED_DIR": "extern_data/handparsed-treebank",

        # directory with the contents of https://nlp.stanford.edu/projects/stanza/bio/
        # on the cluster, for example, /u/nlp/software/stanza/bio_ud
        "BIO_UD_DIR": "extern_data/bio",

        # data root for other general input files, such as VI_VLSP
        "EXTERN_DIR": "extern_data",
    }

    paths = { "DATA_ROOT" : DATA_ROOT }
    for k, v in defaults.items():
        paths[k] = os.environ.get(k, v)

    return paths
