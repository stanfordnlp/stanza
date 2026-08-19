"""
Build a combined POS training dataset for a low-resource target language.

Originally written for Odia, this script now works for any target treebank
that has only a test split.  It first splits that treebank deterministically
into train/dev/test using a hash of each sentence's sent_id.  That way
sentences keep their split assignment even if the file is later reordered.

Additional languages supported in MuRIL that can be mixed in as donors:
  Hindi   (UD_Hindi-HDTB)
  Urdu    (UD_Urdu-UDTB)
  Sindhi  (UD_Sindhi-Isra)
  Marathi (UD_Marathi-UFAL)
  Tamil   (UD_Tamil-TTB)

For all donor languages, xpos and morphological features are stripped so the
model learns only UPOS from them.  The target language keeps its full
annotation.

The purpose of mixing in additional datasets is that while the target
dataset is quite small, there is the Muril transformer from Google
which supports several related South Asian languages.  Thus we can
mix in those datasets and use only the Muril embedding (no fasttext or
charlm) in order to get crosslingual training.  This worked quite well
for Sindhi when building that dataset, and for Odia after it, and should
help for other low-resource targets as well.

https://aclanthology.org/2025.udw-1.11/

Usage examples:

  # POS tagging dataset (default), mixing in Hindi, Urdu, and Sindhi
  python mixed_indic_dataset.py \\
      --donors hindi urdu sindhi \\
      --hindi_size 1000 --urdu_size 1000 --sindhi_size 1000

  # Dependency parsing dataset
  python mixed_indic_dataset.py --mode depparse --use_all_languages

  # All languages, both modes
  python mixed_indic_dataset.py --mode pos      --use_all_languages
  python mixed_indic_dataset.py --mode depparse --use_all_languages

  # A different target language
  python mixed_indic_dataset.py --target_file path/to/bho_bhtb-ud-test.conllu \\
      --target_shortname bho_bhtb --donors hindi

Adapted from the Sindhi build script in UD_Sindhi-Isra by the original authors.

If needed, we can refactor / repurpose this to operate on larger
combinations of datasets.
"""

import argparse
import hashlib
import io
import os
import random
import zipfile

from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL
from stanza.utils.default_paths import get_default_paths


# ---------------------------------------------------------------------------
# Donor language configuration
# ---------------------------------------------------------------------------

# name -> path to the train conllu file relative to UDBASE
DONOR_CONFIGS = {
    "hindi":   "UD_Hindi-HDTB/hi_hdtb-ud-train.conllu",
    "urdu":    "UD_Urdu-UDTB/ur_udtb-ud-train.conllu",
    "sindhi":  "UD_Sindhi-Isra/sd_isra-ud-train.conllu",
    "marathi": "UD_Marathi-UFAL/mr_ufal-ud-train.conllu",
    "tamil":   "UD_Tamil-TTB/ta_ttb-ud-train.conllu",
}


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
    """
    buckets = [[], []], [[], []]  # placeholder; rebuilt below
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
# Main
# ---------------------------------------------------------------------------

def main():
    paths = get_default_paths()
    udbase = paths["UDBASE"]

    parser = argparse.ArgumentParser(
        description="Build a combined POS or dependency parsing training set for a low-resource target language's Stanza models"
    )

    # --- Mode ---
    parser.add_argument(
        "--mode", default="pos", choices=["pos", "depparse"],
        help="Build a POS tagging dataset or a dependency parsing dataset (default: pos)",
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
    parser.add_argument("--hindi_size",   type=int, default=1000,
                        help="Max Hindi sentences to include (default 1000)")
    parser.add_argument("--urdu_size",    type=int, default=1000,
                        help="Max Urdu sentences to include (default 1000)")
    parser.add_argument("--sindhi_size",  type=int, default=1000,
                        help="Max Sindhi sentences to include (default 1000)")
    parser.add_argument("--marathi_size", type=int, default=None,
                        help="Max Marathi sentences (default: all)")
    parser.add_argument("--tamil_size",   type=int, default=None,
                        help="Max Tamil sentences (default: all)")

    # --- Output ---
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to write output files. "
             "Defaults to data/pos or data/depparse depending on --mode.",
    )
    parser.add_argument(
        "--target_shortname", default="or_odtb",
        help="Short name used in output filenames (default: or_odtb)",
    )
    parser.add_argument(
        "--dataset_name", dest="target_shortname", default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "data/pos" if args.mode == "pos" else "data/depparse"

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and split the target-language data.
    #    Always keep full annotation (xpos, feats, head, deprel) so the
    #    same split is usable for both pos and depparse modes.
    # ------------------------------------------------------------------
    print("Reading target-language data from: %s" % args.target_file)
    target_doc = read_conllu(args.target_file, strip_xpos=False)
    print("Total target-language sentences: %d" % len(target_doc.sentences))

    weights = (args.train_weight, args.dev_weight,
               1.0 - args.train_weight - args.dev_weight)
    train, dev, test = split_by_sent_id(target_doc, weights=weights, label=args.target_shortname)

    # ------------------------------------------------------------------
    # 2. Load donor-language training data.
    #
    #    Always strip xpos/feats from donor languages regardless of mode.
    #    For depparse, we want the parser to generalize on UPOS and tree
    #    structure rather than language-specific xpos tagsets.
    #    The target language always retains full annotation since it is
    #    the target language.
    # ------------------------------------------------------------------
    donor_datasets = {}  # name -> Document

    donors = set(args.donors)
    if args.use_all_languages:
        donors |= set(DONOR_CONFIGS.keys())

    size_overrides = {
        "hindi": args.hindi_size,
        "urdu": args.urdu_size,
        "sindhi": args.sindhi_size,
        "marathi": args.marathi_size,
        "tamil": args.tamil_size,
    }

    lang_configs = [
        (lang_name, lang_name in donors, size_overrides[lang_name],
         os.path.join(udbase, lang_path))
        for lang_name, lang_path in DONOR_CONFIGS.items()
    ]

    for lang_name, use_lang, size_cap, lang_path in lang_configs:
        if not use_lang:
            continue
        print("Reading %s from: %s" % (lang_name, lang_path))
        lang_doc = read_conllu(lang_path, strip_xpos=True)
        print("  %d sentences read" % len(lang_doc.sentences))
        if size_cap is not None and len(lang_doc.sentences) > size_cap:
            lang_doc = random_select(lang_doc, size_cap)
            print("  down-sampled to %d sentences" % len(lang_doc.sentences))
        donor_datasets[lang_name] = lang_doc

    # ------------------------------------------------------------------
    # 3. Write output
    # ------------------------------------------------------------------
    shortname = args.target_shortname
    out = args.output_dir

    print("Mode: %s  ->  writing to %s" % (args.mode, out))

    # dev and test are always plain conllu
    dev_path  = os.path.join(out, "%s.dev.in.conllu"  % shortname)
    test_path = os.path.join(out, "%s.test.in.conllu" % shortname)
    CoNLL.write_doc2conll(dev,  dev_path)
    CoNLL.write_doc2conll(test, test_path)
    print("Wrote dev  -> %s (%d sentences)" % (dev_path,  len(dev.sentences)))
    print("Wrote test -> %s (%d sentences)" % (test_path, len(test.sentences)))

    # train is a zip containing one conllu per source
    train_zip_path = os.path.join(out, "%s.train.in.zip" % shortname)
    train_datasets = {"%s_train.in.conllu" % shortname: train}
    for lang_name, lang_doc in donor_datasets.items():
        train_datasets["%s.conllu" % lang_name] = lang_doc

    print("Writing training zip -> %s" % train_zip_path)
    with zipfile.ZipFile(train_zip_path, "w") as zout:
        for name, doc in train_datasets.items():
            if len(doc.sentences) == 0:
                print("  Skipping %s (empty)" % name)
                continue
            with zout.open(name, mode="w") as fout:
                with io.TextIOWrapper(fout, encoding="utf-8") as tout:
                    CoNLL.write_doc2conll(doc, tout)
            print("  Wrote %d sentences as %s" % (len(doc.sentences), name))

    print("Done.")


if __name__ == "__main__":
    main()
