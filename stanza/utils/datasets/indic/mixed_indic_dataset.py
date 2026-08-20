"""
Build a combined training dataset for a low-resource target language.

Originally written for Odia, this script now works for any target treebank
that has only a test split.  It first splits that treebank deterministically
into train/dev/test using a hash of each sentence's sent_id.  That way
sentences keep their split assignment even if the file is later reordered.

It also works with *no* target treebank at all (--no_target).  In that case
train, dev, and test are all built out of the donor languages: each donor
contributes its own train/dev/test splits, and any donor missing a dev or
test split has one carved out of its train file with the same deterministic
sent_id hashing.  This is the mode to use for a language such as Saraiki,
where there is no UD data whatsoever but the neighboring languages share a
script and a good deal of structure.

A donor only takes part in dev or test when the target language size is 0.
As soon as the target has any data of its own, dev and test are made up
entirely of target sentences and the donors contribute train data only -
their dev and test files are not even read.

Additional languages supported in MuRIL that can be mixed in as donors:
  Hindi   (UD_Hindi-HDTB)
  Urdu    (UD_Urdu-UDTB)
  Sindhi  (UD_Sindhi-Isra)
  Marathi (UD_Marathi-UFAL)
  Tamil   (UD_Tamil-TTB)

For all donor languages, xpos and morphological features are stripped so the
model learns only UPOS from them.  The target language keeps its full
annotation.  (When there is no target language, the donor dev/test sets are
stripped as well, so the model is never scored against annotation layers it
was never given.)

The purpose of mixing in additional datasets is that while the target
dataset is quite small, there is the Muril transformer from Google
which supports several related South Asian languages.  Thus we can
mix in those datasets and use only the Muril embedding (no fasttext or
charlm) in order to get crosslingual training.  This worked quite well
for Sindhi when building that dataset, and for Odia after it, and should
help for other low-resource targets as well.

https://aclanthology.org/2025.udw-1.11/

Three model types are supported via --mode:

  pos       writes {short_name}.{dev,test}.in.conllu plus a zip of one
            conllu per source for train      -> data/pos
  depparse  same layout as pos                -> data/depparse
  tokenize  the tokenizer does not read zips and needs a second processing
            step, so all of the sources are concatenated into a single
            {short_name}.{shard}.gold.conllu, after which
            common.convert_conllu_to_txt and
            common.prepare_tokenizer_treebank_labels are run to produce the
            .txt, .toklabels, and -mwt.json files  -> data/tokenize

Usage examples:

  # POS tagging dataset (default), mixing in Hindi, Urdu, and Sindhi
  python mixed_indic_dataset.py \\
      --donors hindi urdu sindhi \\
      --hindi_size 1000 --urdu_size 1000 --sindhi_size 1000

  # Dependency parsing dataset
  python mixed_indic_dataset.py --mode depparse --use_all_languages

  # All languages, all modes
  python mixed_indic_dataset.py --mode pos      --use_all_languages
  python mixed_indic_dataset.py --mode depparse --use_all_languages
  python mixed_indic_dataset.py --mode tokenize --use_all_languages

  # A different target language
  python mixed_indic_dataset.py --target_file path/to/bho_bhtb-ud-test.conllu \\
      --target_shortname bho_bhtb --donors hindi

  # A Saraiki tokenizer with no target data at all: everything, including
  # dev and test, comes from Urdu and Sindhi
  python mixed_indic_dataset.py --mode tokenize --no_target \\
      --target_shortname skr_mixed --donors urdu sindhi \\
      --urdu_size -1 --sindhi_size -1

Adapted from the Sindhi build script in UD_Sindhi-Isra by the original authors.

If needed, we can refactor / repurpose this to operate on larger
combinations of datasets.
"""

import argparse
import hashlib
import io
import os
import random
import re
import zipfile

from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL
from stanza.utils.datasets import common
from stanza.utils.default_paths import get_default_paths


# ---------------------------------------------------------------------------
# Donor language configuration
# ---------------------------------------------------------------------------

# name -> treebank directory under UDBASE.  The individual train/dev/test
# files are located with common.find_treebank_dataset_file so that donors
# which only have some of the splits still work.
DONOR_CONFIGS = {
    "hindi":   "UD_Hindi-HDTB",
    "urdu":    "UD_Urdu-UDTB",
    "sindhi":  "UD_Sindhi-Isra",
    "marathi": "UD_Marathi-UFAL",
    "tamil":   "UD_Tamil-TTB",
}

# how many train sentences to use from each donor unless told otherwise.
# None means "all of them"
DEFAULT_DONOR_SIZES = {
    "hindi":   1000,
    "urdu":    1000,
    "sindhi":  1000,
    "marathi": None,
    "tamil":   None,
}

MODES = ("tokenize", "pos", "depparse")

DATA_DIRS = {
    "tokenize": ("TOKENIZE_DATA_DIR", "data/tokenize"),
    "pos":      ("POS_DATA_DIR",      "data/pos"),
    "depparse": ("DEPPARSE_DATA_DIR", "data/depparse"),
}

SENT_ID_RE = re.compile(r"^#\s*sent_id\s*=\s*(.+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remove_xpos_and_features(doc):
    """Strip xpos and morphological features from every word in *doc*."""
    for sent in doc.sentences:
        for word in sent.words:
            word.feats = None
            word.xpos = None


def read_conllu(path, strip_xpos=True):
    """Read a single .conllu file and return a Stanza Document."""
    if not os.path.exists(path):
        raise FileNotFoundError("File not found: %s" % path)
    doc = CoNLL.conll2doc(path)
    if strip_xpos:
        remove_xpos_and_features(doc)
    return doc


def normalize_size(size):
    """
    Turn a size argument into either None (meaning "keep everything") or a count.

    Negative values are accepted on the command line as a way of asking for
    the whole dataset even when the default for that language is a cap.
    """
    if size is None or size < 0:
        return None
    return size


def random_select(doc, size, seed=1234):
    """Return *size* sentences chosen at random from *doc*."""
    sentences = list(doc.sentences)
    rng = random.Random(seed)
    rng.shuffle(sentences)
    chosen = sentences[:size]
    return Document(
        [s.to_dict() for s in chosen],
        comments=[s.comments for s in chosen],
    )


def maybe_downsample(doc, size, label, seed=1234):
    """Down-sample *doc* to *size* sentences if it is larger than that."""
    size = normalize_size(size)
    if size is not None and len(doc.sentences) > size:
        doc = random_select(doc, size, seed=seed)
        print("  down-sampled %s to %d sentences" % (label, len(doc.sentences)))
    return doc


def prefixed_comments(comments, prefix):
    """
    Return a copy of *comments* with the sent_id prefixed by *prefix*.

    When several treebanks are concatenated into one file, their sent_ids
    would otherwise collide.  A prefix keeps them unique and makes it obvious
    which donor a sentence came from when debugging the output.

    *prefix* of None leaves the comments alone.
    """
    if not prefix:
        return list(comments)
    new_comments = []
    for comment in comments:
        match = SENT_ID_RE.match(comment)
        if match:
            new_comments.append("# sent_id = %s-%s" % (prefix, match.group(1).strip()))
        else:
            new_comments.append(comment)
    return new_comments


def combine_docs(named_docs, prefix_sent_ids=True):
    """
    Concatenate several Documents into one.

    named_docs is a list of (prefix, doc) pairs.  A prefix of None means the
    sentence ids of that document are left untouched - we do that for the
    target language so that its sentences keep the ids they have in UD.
    """
    sentences = []
    comments = []
    for prefix, doc in named_docs:
        if doc is None:
            continue
        for sent in doc.sentences:
            sentences.append(sent.to_dict())
            comments.append(prefixed_comments(sent.comments,
                                              prefix if prefix_sent_ids else None))
    return Document(sentences, comments=comments)


# ---------------------------------------------------------------------------
# Deterministic split keyed on sent_id
# ---------------------------------------------------------------------------

def _sent_bucket(sent, weights=(0.8, 0.1, 0.1)):
    """
    Assign a sentence to train/dev/test deterministically using a hash of its
    sent_id comment.  Falls back to a hash of the sentence text if no sent_id
    comment is present.

    Returns 0 (train), 1 (dev), or 2 (test).
    """
    sent_id = None
    for comment in sent.comments:
        if comment.startswith("# sent_id"):
            sent_id = comment
            break
    key = sent_id if sent_id is not None else sent.text
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    # Map the hash uniformly onto [0, 1) then bucket by cumulative weights
    frac = (h % 10000) / 10000.0
    cumulative = 0.0
    for bucket, w in enumerate(weights):
        cumulative += w
        if frac < cumulative:
            return bucket
    return len(weights) - 1  # safety


def split_by_sent_id(doc, weights=(0.8, 0.1, 0.1), label="target"):
    """
    Split *doc* into (train, dev, test) Documents deterministically using
    sent_id hashing so that split membership is stable across file reorderings.

    A weight of 0.0 for a bucket means that bucket comes back empty, which is
    how we carve a single missing split off a donor's train file.
    """
    train_sents, train_comments = [], []
    dev_sents, dev_comments = [], []
    test_sents, test_comments = [], []

    for sent in doc.sentences:
        bucket = _sent_bucket(sent, weights)
        if bucket == 0:
            train_sents.append(sent.to_dict())
            train_comments.append(sent.comments)
        elif bucket == 1:
            dev_sents.append(sent.to_dict())
            dev_comments.append(sent.comments)
        else:
            test_sents.append(sent.to_dict())
            test_comments.append(sent.comments)

    train = Document(train_sents, comments=train_comments)
    dev   = Document(dev_sents,   comments=dev_comments)
    test  = Document(test_sents,  comments=test_comments)

    print("%s split: %d train / %d dev / %d test" % (
        label, len(train.sentences), len(dev.sentences), len(test.sentences)))
    return train, dev, test


# ---------------------------------------------------------------------------
# Donor loading
# ---------------------------------------------------------------------------

def load_donor(lang_name, treebank, udbase, train_size,
               dev_size=None, test_size=None, need_dev_test=False, seed=1234):
    """
    Load one donor language.

    Returns (train, dev, test).  dev and test are None unless need_dev_test,
    which is the case when there is no target language to take them from.
    If the donor has no dev and/or test of its own, the missing splits are
    carved out of its train file, so a sentence never lands in both.
    """
    train_file = common.find_treebank_dataset_file(treebank, udbase, "train", "conllu", fail=True)
    print("Reading %s train from: %s" % (lang_name, train_file))
    train = read_conllu(train_file, strip_xpos=True)
    print("  %d sentences read" % len(train.sentences))

    dev = None
    test = None
    if need_dev_test:
        dev_file  = common.find_treebank_dataset_file(treebank, udbase, "dev",  "conllu")
        test_file = common.find_treebank_dataset_file(treebank, udbase, "test", "conllu")
        if dev_file is not None:
            print("Reading %s dev from: %s" % (lang_name, dev_file))
            dev = read_conllu(dev_file, strip_xpos=True)
            print("  %d sentences read" % len(dev.sentences))
        if test_file is not None:
            print("Reading %s test from: %s" % (lang_name, test_file))
            test = read_conllu(test_file, strip_xpos=True)
            print("  %d sentences read" % len(test.sentences))

        if dev is None and test is None:
            print("  %s has neither dev nor test.  Splitting its train file" % treebank)
            train, dev, test = split_by_sent_id(train, weights=(0.8, 0.1, 0.1),
                                                label="  %s donor" % lang_name)
        elif dev is None or test is None:
            missing = "dev" if dev is None else "test"
            print("  %s has no %s.  Splitting one out of its train file" % (treebank, missing))
            train, extra, _ = split_by_sent_id(train, weights=(0.9, 0.1, 0.0),
                                               label="  %s donor" % lang_name)
            if dev is None:
                dev = extra
            else:
                test = extra

    train = maybe_downsample(train, train_size, "%s train" % lang_name, seed=seed)
    if dev is not None:
        dev = maybe_downsample(dev, dev_size, "%s dev" % lang_name, seed=seed + 1)
    if test is not None:
        test = maybe_downsample(test, test_size, "%s test" % lang_name, seed=seed + 2)

    return train, dev, test


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_tokenizer_dataset(out_dir, shortname, train_pieces, dev, test, prefix_sent_ids=True):
    """
    Write the tokenizer's train/dev/test.

    Unlike pos and depparse, the tokenizer does not read a zip of datasets,
    so everything is concatenated into one file per shard.  It also needs a
    second processing step to turn the conllu into text plus the token labels
    the model is actually trained on.
    """
    train = combine_docs(train_pieces, prefix_sent_ids=prefix_sent_ids)

    shards = (("train", train), ("dev", dev), ("test", test))
    for shard, doc in shards:
        if doc is None or len(doc.sentences) == 0:
            raise ValueError("The %s shard is empty.  The tokenizer needs all three "
                             "shards - check the donors and the size limits" % shard)

    for shard, doc in shards:
        out_path = common.tokenizer_conllu_name(out_dir, shortname, shard)
        CoNLL.write_doc2conll(doc, out_path)
        # round trip through common so that any dataset which is missing the
        # head/deprel columns gets fake dependencies filled in, which the
        # conllu tooling downstream expects
        sents = common.read_sentences_from_conllu(out_path)
        common.write_sentences_to_conllu(out_path, sents)
        print("Wrote %s -> %s (%d sentences)" % (shard, out_path, len(doc.sentences)))

    print("Converting conllu to txt")
    common.convert_conllu_to_txt(out_dir, shortname)
    print("Preparing tokenizer labels")
    common.prepare_tokenizer_treebank_labels(out_dir, shortname)
    return train


def write_zipped_dataset(out_dir, shortname, train_pieces, dev, test):
    """
    Write the pos / depparse layout: plain conllu for dev and test, and a zip
    with one conllu per source for train.
    """
    dev_path  = os.path.join(out_dir, "%s.dev.in.conllu"  % shortname)
    test_path = os.path.join(out_dir, "%s.test.in.conllu" % shortname)
    CoNLL.write_doc2conll(dev,  dev_path)
    CoNLL.write_doc2conll(test, test_path)
    print("Wrote dev  -> %s (%d sentences)" % (dev_path,  len(dev.sentences)))
    print("Wrote test -> %s (%d sentences)" % (test_path, len(test.sentences)))

    train_zip_path = os.path.join(out_dir, "%s.train.in.zip" % shortname)
    print("Writing training zip -> %s" % train_zip_path)
    with zipfile.ZipFile(train_zip_path, "w") as zout:
        for name, doc in train_pieces:
            if len(doc.sentences) == 0:
                print("  Skipping %s (empty)" % name)
                continue
            with zout.open(name, mode="w") as fout:
                with io.TextIOWrapper(fout, encoding="utf-8") as tout:
                    CoNLL.write_doc2conll(doc, tout)
            print("  Wrote %d sentences as %s" % (len(doc.sentences), name))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_argparse(udbase):
    parser = argparse.ArgumentParser(
        description="Build a combined tokenizer, POS, or dependency parsing training set "
                    "for a low-resource target language's Stanza models"
    )

    # --- Mode ---
    parser.add_argument(
        "--mode", default="pos", choices=list(MODES),
        help="Build a tokenizer, POS tagging, or dependency parsing dataset (default: pos)",
    )

    # --- Target language source ---
    parser.add_argument(
        "--target_file",
        default=os.path.join(udbase, "UD_Odia-ODTB/or_odtb-ud-test.conllu"),
        help="Path to the target-language conllu file (currently the test split only)",
    )
    parser.add_argument(
        "--odia_file", dest="target_file", default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no_target", default=False, action="store_true",
        help="Build the dataset with no target language at all.  train, dev, and test "
             "all come from the donor languages.  Requires --target_shortname.  "
             "The same thing happens if --target_file turns out to have 0 sentences",
    )
    parser.add_argument(
        "--train_weight", type=float, default=0.8,
        help="Fraction of target-language sentences for train (default 0.8)",
    )
    parser.add_argument(
        "--dev_weight", type=float, default=0.1,
        help="Fraction of target-language sentences for dev (default 0.1)",
    )
    # test gets the remainder

    # --- Donor languages ---
    parser.add_argument("--use_all_languages", default=False, action="store_true",
                        help="Include all donor languages (Hindi, Urdu, Sindhi, Marathi, Tamil). "
                             "--donors is still respected and adds to this.")
    parser.add_argument("--donors", nargs="+", default=[], metavar="LANG",
                        choices=sorted(DONOR_CONFIGS.keys()),
                        help="Donor languages to mix in. Choices: %s (default: none)" %
                             ", ".join(sorted(DONOR_CONFIGS.keys())))

    # --- Per-language size caps ---
    for lang_name in sorted(DONOR_CONFIGS.keys()):
        default_size = DEFAULT_DONOR_SIZES.get(lang_name)
        parser.add_argument("--%s_size" % lang_name, type=int, default=default_size,
                            help="Max %s train sentences to include (default: %s).  Use -1 for all" %
                                 (lang_name, "all" if default_size is None else default_size))
    parser.add_argument("--donor_dev_size", type=int, default=500,
                        help="Max dev sentences to take from each donor when there is no "
                             "target language (default 500).  Use -1 for all")
    parser.add_argument("--donor_test_size", type=int, default=500,
                        help="Max test sentences to take from each donor when there is no "
                             "target language (default 500).  Use -1 for all")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed used when down-sampling donors (default 1234)")

    # --- Output ---
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to write output files. "
             "Defaults to data/tokenize, data/pos, or data/depparse depending on --mode.",
    )
    parser.add_argument(
        "--target_shortname", default="or_odtb",
        help="Short name used in output filenames (default: or_odtb)",
    )
    parser.add_argument(
        "--dataset_name", dest="target_shortname", default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no_prefix_sent_ids", dest="prefix_sent_ids", default=True, action="store_false",
        help="Don't prefix donor sent_ids with the language name when concatenating "
             "datasets into a single file",
    )

    return parser


def main():
    paths = get_default_paths()
    udbase = paths["UDBASE"]

    parser = build_argparse(udbase)
    args = parser.parse_args()

    if args.output_dir is None:
        path_key, fallback = DATA_DIRS[args.mode]
        args.output_dir = paths.get(path_key, fallback)

    os.makedirs(args.output_dir, exist_ok=True)

    donors = [x for x in DONOR_CONFIGS if x in set(args.donors)]
    if args.use_all_languages:
        donors = list(DONOR_CONFIGS.keys())

    shortname = args.target_shortname

    if args.no_target:
        if not donors:
            parser.error("--no_target needs at least one donor language to build from.  "
                         "Use --donors or --use_all_languages")
        if shortname == parser.get_default("target_shortname"):
            parser.error("--no_target needs an explicit --target_shortname, as there is "
                         "no target treebank to name the files after")

    # ------------------------------------------------------------------
    # 1. Load and split the target-language data, if there is any.
    #    Always keep full annotation (xpos, feats, head, deprel) so the
    #    same split is usable for all three modes.
    # ------------------------------------------------------------------
    target_train = target_dev = target_test = None
    target_size = 0
    if args.no_target:
        print("No target language file")
    else:
        print("Reading target-language data from: %s" % args.target_file)
        target_doc = read_conllu(args.target_file, strip_xpos=False)
        target_size = len(target_doc.sentences)
        print("Total target-language sentences: %d" % target_size)

        if target_size > 0:
            weights = (args.train_weight, args.dev_weight,
                       1.0 - args.train_weight - args.dev_weight)
            target_train, target_dev, target_test = split_by_sent_id(
                target_doc, weights=weights, label=shortname)
            for shard, doc in (("dev", target_dev), ("test", target_test)):
                if len(doc.sentences) == 0:
                    raise ValueError(
                        "The target language has %d sentences but its %s split came out "
                        "empty.  Donor data is not used for dev or test when there is a "
                        "target language, so adjust --train_weight / --dev_weight instead"
                        % (target_size, shard))

    # The donors only contribute to dev and test when there is no target
    # language at all.  As soon as the target has any data of its own, dev and
    # test are entirely target data and the donors contribute train only.
    use_donor_dev_test = target_size == 0
    if use_donor_dev_test:
        print("Target language size is 0: train, dev, and test will all be built "
              "from the donor languages: %s" % ", ".join(donors))
    else:
        print("Target language has data, so the donors contribute train only")

    # ------------------------------------------------------------------
    # 2. Load donor-language data.
    #
    #    Always strip xpos/feats from donor languages regardless of mode.
    #    For depparse, we want the parser to generalize on UPOS and tree
    #    structure rather than language-specific xpos tagsets.
    #    The target language always retains full annotation since it is
    #    the target language.
    # ------------------------------------------------------------------
    donor_datasets = {}  # name -> (train, dev, test)

    for lang_name in donors:
        train_size = getattr(args, "%s_size" % lang_name)
        donor_datasets[lang_name] = load_donor(
            lang_name, DONOR_CONFIGS[lang_name], udbase,
            train_size=train_size,
            dev_size=args.donor_dev_size,
            test_size=args.donor_test_size,
            need_dev_test=use_donor_dev_test,
            seed=args.seed,
        )

    # ------------------------------------------------------------------
    # 3. Assemble the shards
    # ------------------------------------------------------------------
    # (name in the train zip, doc, sent_id prefix)
    train_pieces = []
    if target_train is not None:
        train_pieces.append(("%s_train.in.conllu" % shortname, target_train, None))
    for lang_name in donors:
        train_pieces.append(("%s.conllu" % lang_name, donor_datasets[lang_name][0], lang_name))

    if use_donor_dev_test:
        dev = combine_docs([(lang_name, donor_datasets[lang_name][1]) for lang_name in donors],
                           prefix_sent_ids=args.prefix_sent_ids)
        test = combine_docs([(lang_name, donor_datasets[lang_name][2]) for lang_name in donors],
                            prefix_sent_ids=args.prefix_sent_ids)
        print("Combined donor dev: %d sentences" % len(dev.sentences))
        print("Combined donor test: %d sentences" % len(test.sentences))
    else:
        # the donors were loaded with need_dev_test=False, so there is nothing
        # of theirs which could leak into dev or test here
        for lang_name in donors:
            _, donor_dev, donor_test = donor_datasets[lang_name]
            assert donor_dev is None and donor_test is None, \
                "%s should not have loaded dev/test data when there is a target language" % lang_name
        dev = target_dev
        test = target_test

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    out = args.output_dir
    print("Mode: %s  ->  writing to %s" % (args.mode, out))

    if args.mode == "tokenize":
        train = write_tokenizer_dataset(
            out, shortname,
            [(prefix, doc) for _, doc, prefix in train_pieces],
            dev, test,
            prefix_sent_ids=args.prefix_sent_ids,
        )
        print("Combined train: %d sentences" % len(train.sentences))
    else:
        write_zipped_dataset(out, shortname,
                             [(name, doc) for name, doc, _ in train_pieces],
                             dev, test)

    print("Done.")


if __name__ == "__main__":
    main()
