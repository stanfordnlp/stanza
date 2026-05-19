#!/usr/bin/env python3
"""
Run the Stanza lemmatizer on a given list of (word, UPOS) pairs.

Usage:
    python lemmatize_known_upos.py [--lang LANG] [--input FILE] [word:UPOS ...]

Examples:
    python lemmatize_known_upos.py --lang en running:VERB flies:NOUN better:ADJ
    python lemmatize_known_upos.py --lang en running_VERB flies_NOUN better_ADJ
    python lemmatize_known_upos.py --lang en --input words.txt

Input file format (one entry per line; colon, underscore, or whitespace-separated):
    running_VERB
    flies:NOUN
    better  ADJ
"""

import argparse
import sys
import stanza
from stanza.models.common.doc import Document


def build_document(word_upos_pairs):
    """
    Build a minimal Stanza Document from (word, upos) pairs.

    Each pair becomes a single-word sentence.  Grouping everything into
    one sentence also works, but one-word sentences are safer because the
    lemmatizer processes each word independently anyway.
    """
    sentences = []
    for word, upos, *_ in word_upos_pairs:
        sentences.append([{
            "id": 1,
            "text": word,
            "upos": upos,
            "lemma": "_",   # placeholder; will be overwritten
        }])
    return Document(sentences)


def lemmatize(word_upos_pairs, lang="en", model_path=None):
    """
    Return a list of (word, upos, lemma) triples.

    A Pipeline with only the lemma processor is created; Stanza will
    download the model if it is not already cached.  `lemma_pretagged=True`
    tells the processor that POS tags are already present, so it does
    not require an upstream POS processor.

    If model_path is given it is passed as `lemma_model_path`, overriding
    the default downloaded model.
    """
    pipeline_kwargs = dict(
        lang=lang,
        processors="lemma",
        package="default_accurate",
        lemma_pretagged=True,   # accept pre-supplied UPOS tags
        verbose=False,
    )
    if model_path is not None:
        pipeline_kwargs["lemma_model_path"] = model_path

    nlp = stanza.Pipeline(**pipeline_kwargs)

    doc = build_document(word_upos_pairs)

    # Call the lemma processor directly on the pre-built Document.
    lemma_proc = nlp.processors["lemma"]
    doc = lemma_proc.process(doc)

    results = []
    for sent, (word, upos, comment) in zip(doc.sentences, word_upos_pairs):
        lemma = sent.words[0].lemma or "_"
        results.append((word, upos, lemma, comment))
    return results


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

# Universal POS tags (https://universaldependencies.org/u/pos/)
_UPOS_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
}


def _split_word_upos(tok):
    """
    Split a single token into (word, upos).

    Accepted formats (in order of preference):
      word:UPOS       e.g.  running:VERB
      word_UPOS       e.g.  como_SCONJ   (underscore separator)

    The underscore split is tried only when the suffix after the last
    underscore is a recognised UPOS tag, so ordinary underscore-containing
    words (e.g. compound_noun) are not accidentally split.

    Returns None if the token cannot be split (caller should then try to
    consume the next token as the UPOS).
    """
    if ":" in tok:
        word, upos = tok.rsplit(":", 1)
        return word.strip(), upos.strip()
    if "_" in tok:
        word, upos = tok.rsplit("_", 1)
        if upos.upper() in _UPOS_TAGS:
            return word.strip(), upos.strip()
    return None


def parse_inline_pairs(tokens):
    """Parse word/UPOS pairs from command-line tokens.

    Accepted formats: 'word:UPOS', 'word_UPOS', or 'word UPOS' (two tokens).
    Returns (word, upos, None) triples (no comments on the command line).
    """
    triples = []
    it = iter(tokens)
    for tok in it:
        split = _split_word_upos(tok)
        if split is not None:
            word, upos = split
        else:
            try:
                upos = next(it)
            except StopIteration:
                sys.exit(f"Error: no UPOS tag supplied for word '{tok}'")
            word = tok.strip()
            upos = upos.strip()
        triples.append((word, upos, None))
    return triples


def parse_input_file(path):
    """Read (word, upos, comment) triples from a file.

    Each non-blank, non-comment line may use any of these formats:
      word_UPOS          # optional inline comment
      word:UPOS
      word  UPOS
    Inline comments (anything after #) are preserved and shown in output.
    """
    triples = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # split off inline comment before parsing word/upos
            comment = None
            if "#" in line:
                data, comment = line[:line.index("#")].strip(), line[line.index("#"):]
            else:
                data = line
            if not data:
                continue
            split = _split_word_upos(data)
            if split is not None:
                word, upos = split
            else:
                parts = data.split()
                if len(parts) != 2:
                    sys.exit(f"Error: line {lineno} in '{path}' is not 'word UPOS': {line!r}")
                word, upos = parts[0], parts[1]
            triples.append((word, upos, comment))
    return triples


def main():
    parser = argparse.ArgumentParser(
        description="Lemmatize (word, UPOS) pairs with the Stanza lemmatizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--input", metavar="FILE", help="Read word/UPOS pairs from a file")
    parser.add_argument(
        "--model-path", metavar="PATH",
        help="Path to a lemmatizer model file, instead of the default downloaded model",
    )
    parser.add_argument(
        "pairs",
        nargs="*",
        metavar="word:UPOS",
        help="Pairs on the command line, e.g.  running:VERB  flies_NOUN",
    )
    args = parser.parse_args()

    # both parse_input_file and parse_inline_pairs return (word, upos, comment) triples
    word_upos_pairs = []
    if args.input:
        word_upos_pairs.extend(parse_input_file(args.input))
    if args.pairs:
        word_upos_pairs.extend(parse_inline_pairs(args.pairs))

    if not word_upos_pairs:
        parser.print_help()
        sys.exit(1)

    results = lemmatize(word_upos_pairs, lang=args.lang, model_path=args.model_path)

    # Print aligned output
    col_word  = max(len(word)  for word, upos, lemma, comment in results)
    col_lemma = max(len(lemma) for word, upos, lemma, comment in results)
    has_comments = any(comment for word, upos, lemma, comment in results)
    print(f"{'WORD':<{col_word}}  {'UPOS':<10}  LEMMA")
    print("-" * (col_word + 10 + col_lemma + (20 if has_comments else 6)))
    for word, upos, lemma, comment in results:
        row = f"{word:<{col_word}}  {upos:<10}  {lemma:<{col_lemma}}"
        if has_comments:
            row += f"  {comment or ''}"
        print(row)


if __name__ == "__main__":
    main()
