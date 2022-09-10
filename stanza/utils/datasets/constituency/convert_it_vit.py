"""Converts the proprietary VIT dataset to a format suitable for stanza

There are multiple corrections in the UD version of VIT, along with
recommended splits for the MWT, along with recommended splits of
the sentences into train/dev/test

Accordingly, it is necessary to use the UD dataset as a reference

Here is a sample line of the text file we use:

#ID=sent_00002  cp-[sp-[part-negli, sn-[sa-[ag-ultimi], nt-anni]], f-[sn-[art-la, n-dinamica, spd-[partd-dei, sn-[n-polo_di_attrazione]]], ibar-[ause-è, ausep-stata, savv-[savv-[avv-sempre], avv-più], vppt-caratterizzata], compin-[spda-[partda-dall, sn-[n-emergere, spd-[pd-di, sn-[art-una, sa-[ag-crescente], n-concorrenza, f2-[rel-che, f-[ibar-[clit-si, ause-è, avv-progressivamente, vppin-spostata], compin-[spda-[partda-dalle, sn-[sa-[ag-singole], n-imprese]], sp-[part-ai, sn-[n-sistemi, sa-[coord-[ag-economici, cong-e, ag-territoriali]]]], fp-[punt-',', sv5-[vgt-determinando, compt-[sn-[art-l_, nf-esigenza, spd-[pd-di, sn-[art-una, n-riconsiderazione, spd-[partd-dei, sn-[n-rapporti, sv3-[ppre-esistenti, compin-[sp-[p-tra, sn-[n-soggetti, sa-[ag-produttivi]]], cong-e, sn-[n-ambiente, f2-[sp-[p-in, sn-[relob-cui]], f-[sn-[deit-questi], ibar-[vin-operano, punto-.]]]]]]]]]]]]]]]]]]]]]]]]

Here you can already see multiple issues when parsing:
- the first word is "negli", which is split into In_ADP gli_DET in the UD version
- also the first word is capitalized in the UD version
- comma looks like a tempting split target, but there is a ',' in this sentence
  punt-','
- not shown here is '-' which is different from the - used for denoting POS
  par-'-'

Fortunately, -[ is always an open and ] is always a close

As of April 2022, the UD version of the dataset has some minor edits
which are necessary for the proper functioning of this script.
Otherwise, the MWT won't align correctly, some typos won't be
corrected, etc.  These edits are released in UD 2.10

The data itself is available from ELRA:

http://catalog.elra.info/en-us/repository/browse/ELRA-W0040/

Internally at Stanford you can contact Chris Manning or John Bauer.

The processing goes as follows:
- read in UD and con trees
  some of the con trees have broken brackets and are discarded
  in other cases, abbreviations were turned into single tokens in UD
- extract the MWT expansions of Italian contractions,
  such as Negli -> In gli
- attempt to align the trees from the two datasets using ngrams
  some trees had the sentence splitting updated
  sentences which can't be matched are discarded
- use CoreNLP tsurgeon to update tokens in the con trees
  based on the information in the UD dataset
  - split contractions
  - rearrange clitics which are occasionally non-projective
- replace the words in the con tree with the dep tree's words
  this takes advantage of spelling & capitalization fixes

In 2022, there was an update to the dataset from Prof. Delmonte.
This update is hopefully in current ELRA distributions now.
If not, please contact ELRA to specifically ask for the updated version.
Internally to Stanford, feel free to ask Chris or John for the updates.
Look for the line below "original version with more errors"

In August 2022, Prof. Delmonte made a slight update in a zip file
`john.zip`.  If/when that gets updated to ELRA, we will update it
here.  Contact Chris or John for a copy if not updated yet, or go
back in git history to get the older version of the code which
works with the 2022 ELRA update.

Later, in September 2022, there is yet another update,
New version of VIT.zip
Unzip the contents into a folder
$CONSTITUENCY_BASE/italian/it_vit
so there should be a file
$CONSTITUENCY_BASE/italian/it_vit/VITwritten/VITconstsyntNumb
"""

from collections import defaultdict, deque
import itertools
import os
import re
import sys

from tqdm import tqdm

from stanza.models.constituency.tree_reader import read_trees, UnclosedTreeError, ExtraCloseTreeError
from stanza.server import tsurgeon
from stanza.utils.conll import CoNLL
from stanza.utils.datasets.constituency.utils import SHARDS, write_dataset
import stanza.utils.default_paths as default_paths

def read_constituency_sentences(fin):
    """
    Reads the lines from the constituency treebank and splits into ID, text

    No further processing is done on the trees yet
    """
    sentences = []
    for line in fin:
        line = line.strip()
        # WTF why doesn't strip() remove this
        line = line.replace(u'\ufeff', '')
        if not line:
            continue
        sent_id, sent_text = line.split(maxsplit=1)
        # we have seen a couple different versions of this sentence header
        # although one file is always consistent with itself, at least
        if not sent_id.startswith("#ID=sent") and not sent_id.startswith("ID#sent"):
            raise ValueError("Unexpected start of sentence: |{}|".format(sent_id))
        if not sent_text:
            raise ValueError("Empty text for |{}|".format(sent_id))
        sentences.append((sent_id, sent_text))
    return sentences

def read_constituency_file(filename):
    with open(filename, encoding='utf-8') as fin:
        return read_constituency_sentences(fin)

OPEN = "-["
CLOSE = "]"

DATE_RE = re.compile("^([0-9]{1,2})[_]([0-9]{2})$")
PERCENT_RE = re.compile(r"^([0-9]{1,2})[_]([0-9]{1,2}[%])$")
DECIMAL_RE = re.compile(r"^([0-9])[_]([0-9])$")

def raw_tree(text):
    """
    A sentence will look like this:
       #ID=sent_00001  fc-[f3-[sn-[art-le, n-infrastrutture, sc-[ccom-come, sn-[n-fattore, spd-[pd-di,
                       sn-[n-competitività]]]]]], f3-[spd-[pd-di, sn-[mw-Angela, nh-Airoldi]]], punto-.]
    Non-preterminal nodes have tags, followed by the stuff under the node, -[
    The node is closed by the ]
    """
    pieces = []
    open_pieces = text.split(OPEN)
    for open_idx, open_piece in enumerate(open_pieces):
        if open_idx > 0:
            pieces[-1] = pieces[-1] + OPEN
        open_piece = open_piece.strip()
        if not open_piece:
            raise ValueError("Unexpected empty node!")
        close_pieces = open_piece.split(CLOSE)
        for close_idx, close_piece in enumerate(close_pieces):
            if close_idx > 0:
                pieces.append(CLOSE)
            close_piece = close_piece.strip()
            if not close_piece:
                # this is okay - multiple closes at the end of a deep bracket
                continue
            word_pieces = close_piece.split(", ")
            pieces.extend([x.strip() for x in word_pieces if x.strip()])

    # at this point, pieces is a list with:
    #   tag-[     for opens
    #   tag-word  for words
    #   ]         for closes
    # this structure converts pretty well to reading using the tree reader

    PIECE_MAPPING = {
        "agn-/ter'":               "(agn ter)",
        "cong-'&'":                "(cong &)",
        "da_riempire-'...'":       "(da_riempire ...)",
        "date-1992_1993":          "(date 1992/1993)",
        "date-'31-12-95'":         "(date 31-12-95)",
        "date-'novantaquattro-95'":"(date novantaquattro-95)",
        "date-'novantaquattro-95": "(date novantaquattro-95)",
        "date-'novantaquattro-novantacinque'": "(date novantaquattro-novantacinque)",
        "dirs-':'":                "(dirs :)",
        "dirs-'\"'":               "(dirs \")",
        "mw-'&'":                  "(mw &)",
        "mw-'Presunto'":           "(mw Presunto)",
        "nh-'Alain-Gauze'":        "(nh Alain-Gauze)",
        "np-'porto_Marghera'":     "(np Porto) (np Marghera)",
        "np-'roma-l_aquila'":      "(np Roma-L'Aquila)",
        "np-'L_Aquila-Villa_Vomano'": "(np L'Aquila) (np -) (np Villa) (np Vomano)",
        "npro-'Avanti_!'":         "(npro Avanti_!)",
        "npro-'Viacom-Paramount'": "(npro Viacom-Paramount)",
        "npro-'Rhone-Poulenc'":    "(npro Rhone-Poulenc)",
        "npro-'Itar-Tass'":        "(npro Itar-Tass)",
        "par-(-)":                 "(par -)",
        "par-','":                 "(par ,)",
        "par-'<'":                 "(par <)",
        "par-'>'":                 "(par >)",
        "par-'-'":                 "(par -)",
        "par-'\"'":                "(par \")",
        "par-'('":                 "(par -LRB-)",
        "par-')'":                 "(par -RRB-)",
        "par-'&&'":                "(par &&)",
        "punt-','":                "(punt ,)",
        "punt-'-'":                "(punt -)",
        "punt-';'":                "(punt ;)",
        "punto-':'":               "(punto :)",
        "punto-';'":               "(punto ;)",
        "puntint-'!'":             "(puntint !)",
        "puntint-'?'":             "(puntint !)",
        "num-'2plus2'":            "(num 2+2)",
        "num-/bis'":               "(num bis)",
        "num-/ter'":               "(num ter)",
        "num-18_00/1_00":          "(num 18:00/1:00)",
        "num-1/500_2/000":         "(num 1.500-2.000)",
        "num-16_1":                "(num 16,1)",
        "num-0_1":                 "(num 0,1)",
        "num-0_3":                 "(num 0,3)",
        "num-2_7":                 "(num 2,7)",
        "num-455_68":              "(num 455/68)",
        "num-437_5":               "(num 437,5)",
        "num-4708_82":             "(num 4708,82)",
        "num-16EQ517_7":           "(num 16EQ517/7)",
        "num-2=184_90":            "(num 2=184/90)",
        "num-3EQ429_20":           "(num 3eq429/20)",
        "num-'1990-EQU-100'":      "(num 1990-EQU-100)",
        "num-'500-EQU-250'":       "(num 500-EQU-250)",
        "num-0_39%minus":          "(num 0,39%-)",
        "num-1_88/76":             "(num 1-88/76)",
        "num-'70/80'":             "(num 70,80)",
        "num-'18/20'":             "(num 18:20)",
        "num-295/mila'":           "(num 295mila)",
        "num-'295/mila'":          "(num 295mila)",
        "num-0/07%plus":           "(num 0,07%) (num plus)",
        "num-0/69%minus":          "(num 0,69%) (num minus)",
        "num-0_39%minus":          "(num 0,39%) (num minus)",
        "num-9_11/16":             "(num 9-11,16)",
        "num-2/184_90":            "(num 2=184/90)",
        "num-3/429_20":            "(num 3eq429/20)",
        # TODO: remove the following num conversions if possible
        # this would require editing either constituency or UD
        "num-1:28_124":            "(num 1=8/1242)",
        "num-1:28_397":            "(num 1=8/3972)",
        "num-1:28_947":            "(num 1=8/9472)",
        "num-1:29_657":            "(num 1=9/6572)",
        "num-1:29_867":            "(num 1=9/8672)",
        "num-1:29_874":            "(num 1=9/8742)",
        "num-1:30_083":            "(num 1=0/0833)",
        "num-1:30_140":            "(num 1=0/1403)",
        "num-1:30_354":            "(num 1=0/3543)",
        "num-1:30_453":            "(num 1=0/4533)",
        "num-1:30_946":            "(num 1=0/9463)",
        "num-1:31_602":            "(num 1=1/6023)",
        "num-1:31_842":            "(num 1=1/8423)",
        "num-1:32_087":            "(num 1=2/0873)",
        "num-1:32_259":            "(num 1=2/2593)",
        "num-1:33_166":            "(num 1=3/1663)",
        "num-1:34_154":            "(num 1=4/1543)",
        "num-1:34_556":            "(num 1=4/5563)",
        "num-1:35_323":            "(num 1=5/3233)",
        "num-1:36_023":            "(num 1=6/0233)",
        "num-1:36_076":            "(num 1=6/0763)",
        "num-1:36_651":            "(num 1=6/6513)",
        "n-giga_flop/s":           "(n giga_flop/s)",
        "sect-'g-1'":              "(sect g-1)",
        "sect-'h-1'":              "(sect h-1)",
        "sect-'h-2'":              "(sect h-2)",
        "sect-'h-3'":              "(sect h-3)",
        "abbr-'a-b-c'":            "(abbr a-b-c)",
        "abbr-d_o_a_":             "(abbr DOA)",
        "abbr-d_l_":               "(abbr DL)",
        "abbr-i_s_e_f_":           "(abbr ISEF)",
        "abbr-d_p_r_":             "(abbr DPR)",
        "abbr-D_P_R_":             "(abbr DPR)",
        "abbr-d_m_":               "(abbr dm)",
        "abbr-T_U_":               "(abbr TU)",
        "abbr-F_A_M_E_":           "(abbr Fame)",
        "dots-'...'":              "(dots ...)",
    }
    new_pieces = ["(ROOT "]
    for piece in pieces:
        if piece.endswith(OPEN):
            new_pieces.append("(" + piece[:-2])
        elif piece == CLOSE:
            new_pieces.append(")")
        elif piece in PIECE_MAPPING:
            new_pieces.append(PIECE_MAPPING[piece])
        else:
            # maxsplit=1 because of words like 1990-EQU-100
            tag, word = piece.split("-", maxsplit=1)
            if word.find("'") >= 0 or word.find("(") >= 0 or word.find(")") >= 0:
                raise ValueError("Unhandled weird node: {}".format(piece))
            if word.endswith("_"):
                word = word[:-1] + "'"
            date_match = DATE_RE.match(word)
            if date_match:
                # 10_30 special case sent_07072
                # 16_30 special case sent_07098
                # 21_15 special case sent_07099 and others
                word = date_match.group(1) + ":" + date_match.group(2)
            percent = PERCENT_RE.match(word)
            if percent:
                word = percent.group(1) + "," + percent.group(2)
            decimal = DECIMAL_RE.match(word)
            if decimal:
                word = decimal.group(1) + "," + decimal.group(2)
            # there are words which are multiple words mashed together
            # with _ for some reason
            # also, words which end in ' are replaced with _
            # fortunately, no words seem to have both
            # splitting like this means the tags are likely wrong,
            # but the conparser needs to retag anyway, so it shouldn't matter
            word_pieces = word.split("_")
            for word_piece in word_pieces:
                new_pieces.append("(%s %s)" % (tag, word_piece))
    new_pieces.append(")")

    text = " ".join(new_pieces)
    trees = read_trees(text)
    if len(trees) > 1:
        raise ValueError("Unexpected number of trees!")
    return trees[0]

def extract_ngrams(sentence, process_func, ngram_len=4):
    leaf_words = [x for x in process_func(sentence)]
    leaf_words = ["l'" if x == "l" else x for x in leaf_words]
    if len(leaf_words) <= ngram_len:
        return [tuple(leaf_words)]
    its = [leaf_words[i:i+len(leaf_words)-ngram_len+1] for i in range(ngram_len)]
    return [words for words in itertools.zip_longest(*its)]

def build_ngrams(sentences, process_func, id_func, ngram_len=4):
    """
    Turn the list of processed trees into a bunch of ngrams

    The returned map is from tuple to set of ids

    The idea being that this map can be used to search for trees to
    match datasets
    """
    ngram_map = defaultdict(set)
    for sentence in tqdm(sentences, postfix="Extracting ngrams"):
        sentence_id = id_func(sentence)
        ngrams = extract_ngrams(sentence, process_func, ngram_len)
        for ngram in ngrams:
            ngram_map[ngram].add(sentence_id)
    return ngram_map

# just the tokens (maybe use words?  depends on MWT in the con dataset)
DEP_PROCESS_FUNC = lambda x: [t.text.lower() for t in x.tokens]
# find the comment with "sent_id" in it, take just the id itself
DEP_ID_FUNC = lambda x: [c for c in x.comments if c.startswith("# sent_id")][0].split()[-1]

CON_PROCESS_FUNC = lambda x: [y.lower() for y in x.leaf_labels()]

def match_ngrams(sentence_ngrams, ngram_map, debug=False):
    """
    Check if there is a SINGLE matching sentence in the ngram_map for these ngrams

    If an ngram shows up in multiple sentences, that is okay, but we ignore that info
    If an ngram shows up in just one sentence, that is considered the match
    If a different ngram then shows up in a different sentence, that is a problem
    TODO: taking the intersection of all non-empty matches might be better
    """
    if debug:
        print("NGRAMS FOR DEBUG SENTENCE:")
    potential_match = None
    unknown_ngram = 0
    for ngram in sentence_ngrams:
        con_matches = ngram_map[ngram]
        if debug:
            print("{} matched {}".format(ngram, len(con_matches)))
        if len(con_matches) == 0:
            unknown_ngram += 1
            continue
        if len(con_matches) > 1:
            continue
        # get the one & only element from the set
        con_match = next(iter(con_matches))
        if debug:
            print("  {}".format(con_match))
        if potential_match is None:
            potential_match = con_match
        elif potential_match != con_match:
            return None
    if unknown_ngram > len(sentence_ngrams) / 2:
        return None
    return potential_match

def match_sentences(con_tree_map, con_vit_ngrams, dep_sentences, split_name, debug_sentence=None):
    """
    Match ngrams in the dependency sentences to the constituency sentences

    Then, to make sure the constituency sentence wasn't split into two
    in the UD dataset, this checks the ngrams in the reverse direction

    Some examples of things which don't match:
      VIT-4769 Insegnanti non vedenti, insegnanti non autosufficienti con protesi agli arti inferiori.
      this is duplicated in the original dataset, so the matching algorithm can't possibly work

      VIT-4796 I posti istituiti con attività di sostegno dei docenti che ottengono il trasferimento su classi di concorso;
      the correct con match should be sent_04829 but the brackets on that tree are broken
    """
    con_to_dep_matches = {}
    dep_ngram_map = build_ngrams(dep_sentences, DEP_PROCESS_FUNC, DEP_ID_FUNC)
    unmatched = 0
    bad_match = 0
    for sentence in dep_sentences:
        sentence_ngrams = extract_ngrams(sentence, DEP_PROCESS_FUNC)
        potential_match = match_ngrams(sentence_ngrams, con_vit_ngrams, debug_sentence is not None and DEP_ID_FUNC(sentence) == debug_sentence)
        if potential_match is None:
            if unmatched < 5:
                print("Could not match the following sentence: {} {}".format(DEP_ID_FUNC(sentence), sentence.text))
            unmatched += 1
            continue
        if potential_match not in con_tree_map:
            raise ValueError("wtf")
        con_ngrams = extract_ngrams(con_tree_map[potential_match], CON_PROCESS_FUNC)
        reverse_match = match_ngrams(con_ngrams, dep_ngram_map)
        if reverse_match is None:
            #print("Matched sentence {} to sentence {} but the reverse match failed".format(sentence.text, " ".join(con_tree_map[potential_match].leaf_labels())))
            bad_match += 1
            continue
        con_to_dep_matches[potential_match] = reverse_match
    print("Failed to match %d sentences and found %d spurious matches in the %s section" % (unmatched, bad_match, split_name))
    return con_to_dep_matches

EXCEPTIONS = ["gliene", "glielo", "gliela", "eccoci"]

def get_mwt(*dep_datasets):
    """
    Get the ADP/DET MWTs from the UD dataset

    This class of MWT are expanded in the UD but not the constituencies
    """
    mwt_map = {}
    for dataset in dep_datasets:
        for sentence in dataset.sentences:
            for token in sentence.tokens:
                if len(token.words) == 1:
                    continue
                # words such as "accorgermene" we just skip over
                # those are already expanded in the constituency dataset
                # TODO: the clitics are actually expanded weirdly, maybe need to compensate for that
                if token.words[0].upos in ('VERB', 'AUX') and all(word.upos == 'PRON' for word in token.words[1:]):
                    continue
                if token.text.lower() in EXCEPTIONS:
                    continue
                if len(token.words) != 2 or token.words[0].upos != 'ADP' or token.words[1].upos != 'DET':
                    raise ValueError("Not sure how to handle this: {}".format(token))
                expansion = (token.words[0].text, token.words[1].text)
                if token.text in mwt_map:
                    if mwt_map[token.text] != expansion:
                        raise ValueError("Inconsistent MWT: {} -> {} or {}".format(token.text, expansion, mwt_map[token.text]))
                    continue
                #print("Expanding {} to {}".format(token.text, expansion))
                mwt_map[token.text] = expansion
    return mwt_map

def update_mwts_and_special_cases(original_tree, dep_sentence, mwt_map, tsurgeon_processor):
    """
    Replace MWT structures with their UD equivalents, along with some other minor tsurgeon based edits

    original_tree: the tree as read from VIT
    dep_sentence: the UD dependency dataset version of this sentence
    """
    updated_tree = original_tree

    operations = []

    # first, remove titles or testo from the start of a sentence
    con_words = updated_tree.leaf_labels()
    if con_words[0] == "Tit'":
        operations.append(["/^Tit'$/=prune !, __", "prune prune"])
    elif con_words[0] == "TESTO":
        operations.append(["/^TESTO$/=prune !, __", "prune prune"])
    elif con_words[0] == "testo":
        operations.append(["/^testo$/ !, __ . /^:$/=prune", "prune prune"])
        operations.append(["/^testo$/=prune !, __", "prune prune"])

    if len(con_words) >= 2 and con_words[-2] == '...' and con_words[-1] == '.':
        # the most recent VIT constituency has some sentence final . after a ...
        # the UD dataset has a more typical ... ending instead
        # these lines used to say "riempire" which was rather odd
        operations.append(["/^[.][.][.]$/ . /^[.]$/=prune", "prune prune"])

    # a few constituent tags are simply errors which need to be fixed
    if original_tree.children[0].label == 'p':
        # 'p' shouldn't be at root
        operations.append(["_ROOT_ < p=p", "relabel p cp"])
    # fix one specific tree if it has an s_top in it
    operations.append(["s_top=stop < (in=in < più=piu)", "replace piu (q più)", "relabel in sq", "relabel stop sa"])
    # sect doesn't exist as a constituent.  replace it with sa
    operations.append(["sect=sect < num", "relabel sect sa"])
    # ppas as an internal node gets removed
    operations.append(["ppas=ppas < (__ < __)", "excise ppas ppas"])

    # now assemble a bunch of regex to split and otherwise manipulate
    # the MWT in the trees
    for token in dep_sentence.tokens:
        if len(token.words) == 1:
            continue
        if token.text in mwt_map:
            mwt_pieces = mwt_map[token.text]
            if len(mwt_pieces) != 2:
                raise NotImplementedError("Expected exactly 2 pieces of mwt for %s" % token.text)
            # the MWT words in the UD version will have ' when needed,
            # but the corresponding ' is skipped in the con version of VIT,
            # hence the replace("'", "")
            # however, all' has the ' included, because this is a
            # constituent treebank, not a consistent treebank
            search_regex = "/^(?i:%s(?:')?)$/" % token.text.replace("'", "")
            # tags which seem to be relevant:
            # avvl|ccom|php|part|partd|partda
            tregex = "__ !> __ <<<%d (%s=child > (__=parent $+ sn=sn))" % (token.id[0], search_regex)
            tsurgeons = ["insert (art %s) >0 sn" % mwt_pieces[1], "relabel child %s" % mwt_pieces[0]]
            operations.append([tregex] + tsurgeons)

            tregex = "__ !> __ <<<%d (%s=child > (__=parent !$+ sn !$+ (art < %s)))" % (token.id[0], search_regex, mwt_pieces[1])
            tsurgeons = ["insert (art %s) $- parent" % mwt_pieces[1], "relabel child %s" % mwt_pieces[0]]
            operations.append([tregex] + tsurgeons)
        elif len(token.words) == 2:
            #print("{} not in mwt_map".format(token.text))
            # apparently some trees like sent_00381 and sent_05070
            # have the clitic in a non-projective manner
            #   [vcl-essersi, vppin-sparato, compt-[clitdat-si
            #   intj-figurarsi, fs-[cosu-quando, f-[ibar-[clit-si
            # and before you ask, there are also clitics which are
            # simply not there at all, rather than always attached
            # in a non-projective manner
            tregex = "__=parent < (/^(?i:%s)$/=child . (__=np !< __ . (/^clit/=clit < %s)))" % (token.text, token.words[1].text)
            tsurgeon = "moveprune clit $- parent"
            operations.append([tregex, tsurgeon])

            # there are also some trees which don't have clitics
            # for example, trees should look like this:
            #   [ibar-[vsup-poteva, vcl-rivelarsi], compc-[clit-si, sn-[...]]]
            # however, at least one such example for rivelarsi instead
            # looks like this, with no corresponding clit
            #   [... vcl-rivelarsi], compc-[sn-[in-ancora]]
            # note that is the actual tag, not just me being pissed off
            # breaking down the tregex:
            # the child is the original MWT, not split
            # !. clit verifies that it is not split (and stops the tsurgeon once fixed)
            # !$+ checks that the parent of the MWT is the last element under parent
            # note that !. can leave the immediate parent to touch the clit
            # neighbor will be the place the new clit will be sticking out
            tregex = "__=parent < (/^(?i:%s)$/=child !. /^clit/) !$+ __ > (__=gp $+ __=neighbor)" % token.text
            tsurgeon = "insert (clit %s) >0 neighbor" % token.words[1].text
            operations.append([tregex, tsurgeon])

            # secondary option: while most trees are like the above,
            # with an outer bracket around the MWT and another verb,
            # some go straight into the next phrase
            #   sent_05076
            #   sv5-[vcl-adeguandosi, compin-[sp-[part-alle, ...
            tregex = "__=parent < (/^(?i:%s)$/=child !. /^clit/) $+ __" % token.text
            tsurgeon = "insert (clit %s) $- parent" % token.words[1].text
            operations.append([tregex, tsurgeon])
        else:
            pass
    if len(operations) > 0:
        updated_tree = tsurgeon_processor.process(updated_tree, *operations)[0]
    return updated_tree, operations

def update_tree(original_tree, dep_sentence, con_id, dep_id, mwt_map, tsurgeon_processor):
    """
    Update a tree using the mwt_map and tsurgeon to expand some MWTs

    Then replace the words in the con tree with the words in the dep tree
    """
    ud_words = [x.text for x in dep_sentence.words]

    updated_tree, operations = update_mwts_and_special_cases(original_tree, dep_sentence, mwt_map, tsurgeon_processor)

    # this checks number of words
    try:
        updated_tree = updated_tree.replace_words(ud_words)
    except ValueError as e:
        raise ValueError("Failed to process {} {}:\nORIGINAL TREE\n{}\nUPDATED TREE\n{}\nUPDATED LEAVES\n{}\nUD TEXT\n{}\nTsurgeons applied:\n{}\n".format(con_id, dep_id, original_tree, updated_tree, updated_tree.leaf_labels(), ud_words, "\n".join("{}".format(op) for op in operations))) from e
    return updated_tree

# train set:
#  858: missing close parens in the UD conversion
# 1169: 'che', 'poi', 'tutti', 'i', 'Paesi', 'ue', '.' -> 'per', 'tutti', 'i', 'paesi', 'Ue', '.'
# 2375: the problem is inconsistent treatment of s_p_a_
# 05052: the heuristic to fill in a missing "si" doesn't work because there's
#   already another "si" immediately after
#
# test set:
# 09764: weird punct at end
# 10058: weird punct at end
IGNORE_IDS = ["sent_00867", "sent_01169", "sent_02375", "sent_05052", "sent_09764", "sent_10058"]

def extract_updated_dataset(con_tree_map, dep_sentence_map, split_ids, mwt_map, tsurgeon_processor):
    """
    Update constituency trees using the information in the dependency treebank
    """
    trees = []
    for con_id, dep_id in tqdm(split_ids.items()):
        # skip a few trees which have non-MWT word modifications
        if con_id in IGNORE_IDS:
            continue
        original_tree = con_tree_map[con_id]
        dep_sentence = dep_sentence_map[dep_id]
        updated_tree = update_tree(original_tree, dep_sentence, con_id, dep_id, mwt_map, tsurgeon_processor)

        trees.append(updated_tree)
    return trees

def convert_it_vit(con_directory, ud_directory, output_directory, dataset_name, debug_sentence=None):
    # original version with more errors
    #con_filename = os.path.join(con_directory, "2011-12-20", "Archive", "VIT_newconstsynt.txt")
    # this is the April 2022 version
    #con_filename = os.path.join(con_directory, "VIT_newconstsynt.txt")
    # the most recent update from ELRA may look like this?
    # it's what we got, at least
    # con_filename = os.path.join(con_directory, "italian", "VITwritten", "VITconstsyntNumb")
    con_filename = os.path.join(con_directory, "italian", "it_vit", "VITwritten", "VITconstsyntNumb")
    ud_vit_train = os.path.join(ud_directory, "it_vit-ud-train.conllu")
    ud_vit_dev   = os.path.join(ud_directory, "it_vit-ud-dev.conllu")
    ud_vit_test  = os.path.join(ud_directory, "it_vit-ud-test.conllu")

    print("Reading UD train/dev/test")
    ud_train_data = CoNLL.conll2doc(input_file=ud_vit_train)
    ud_dev_data   = CoNLL.conll2doc(input_file=ud_vit_dev)
    ud_test_data  = CoNLL.conll2doc(input_file=ud_vit_test)

    ud_vit_train_map = { DEP_ID_FUNC(x) : x for x in ud_train_data.sentences }
    ud_vit_dev_map   = { DEP_ID_FUNC(x) : x for x in ud_dev_data.sentences }
    ud_vit_test_map  = { DEP_ID_FUNC(x) : x for x in ud_test_data.sentences }

    print("Getting ADP/DET expansions from UD data")
    mwt_map = get_mwt(ud_train_data, ud_dev_data, ud_test_data)

    con_sentences = read_constituency_file(con_filename)
    num_discarded = 0
    con_tree_map = {}
    for idx, sentence in enumerate(tqdm(con_sentences, postfix="Processing")):
        try:
            tree = raw_tree(sentence[1])
            if sentence[0].startswith("#ID="):
                tree_id = sentence[0].split("=")[-1]
            else:
                tree_id = sentence[0].split("#")[-1]
            # don't care about the raw text?
            con_tree_map[tree_id] = tree
        except UnclosedTreeError as e:
            num_discarded = num_discarded + 1
            print("Discarding {} because of reading error:\n  {}: {}\n  {}".format(sentence[0], type(e), e, sentence[1]))
        except ExtraCloseTreeError as e:
            num_discarded = num_discarded + 1
            print("Discarding {} because of reading error:\n  {}: {}\n  {}".format(sentence[0], type(e), e, sentence[1]))
        except ValueError as e:
            print("Discarding {} because of reading error:\n  {}: {}\n  {}".format(sentence[0], type(e), e, sentence[1]))
            num_discarded = num_discarded + 1
            #raise ValueError("Could not process line %d" % idx) from e

    print("Discarded %d trees.  Have %d trees left" % (num_discarded, len(con_tree_map)))
    if num_discarded > 0:
        raise ValueError("Oops!  We thought all of the VIT trees were properly bracketed now")
    con_vit_ngrams = build_ngrams(con_tree_map.items(), lambda x: CON_PROCESS_FUNC(x[1]), lambda x: x[0])

    # TODO: match more sentences.  some are probably missing because of MWT
    train_ids = match_sentences(con_tree_map, con_vit_ngrams, ud_train_data.sentences, "train", debug_sentence)
    dev_ids   = match_sentences(con_tree_map, con_vit_ngrams, ud_dev_data.sentences,   "dev",   debug_sentence)
    test_ids  = match_sentences(con_tree_map, con_vit_ngrams, ud_test_data.sentences,  "test",  debug_sentence)
    print("Trees: {} train {} dev {} test".format(len(train_ids), len(dev_ids), len(test_ids)))

    # the moveprune feature requires a new corenlp release after 4.4.0
    with tsurgeon.Tsurgeon(classpath="$CLASSPATH") as tsurgeon_processor:
        train_trees = extract_updated_dataset(con_tree_map, ud_vit_train_map, train_ids, mwt_map, tsurgeon_processor)
        dev_trees   = extract_updated_dataset(con_tree_map, ud_vit_dev_map,   dev_ids,   mwt_map, tsurgeon_processor)
        test_trees  = extract_updated_dataset(con_tree_map, ud_vit_test_map,  test_ids,  mwt_map, tsurgeon_processor)

    write_dataset([train_trees, dev_trees, test_trees], output_directory, dataset_name)

def main():
    paths = default_paths.get_default_paths()
    con_directory = paths["CONSTITUENCY_BASE"]
    ud_directory  = os.path.join(paths["UDBASE"], "UD_Italian-VIT")

    output_directory = paths["CONSTITUENCY_DATA_DIR"]
    dataset_name = "it_vit"

    debug_sentence = sys.argv[1] if len(sys.argv) > 1 else None

    convert_it_vit(con_directory, ud_directory, output_directory, dataset_name, debug_sentence)

if __name__ == '__main__':
    main()
