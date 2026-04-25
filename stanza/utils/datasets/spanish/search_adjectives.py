import argparse
from collections import defaultdict
import os
import re

from stanza.utils.conll import CoNLL

# caliza?  not sure if this is a special case or not

special_cases = [
    "alienígena",
    "indígena",
    "hipócrita",
    "idiota",
    "pirata",
    "extra",
    "quejica",  # whiny, apparently no quejico
    "psicópata",
    "rubioceniza", # a compound of two other ADJ
    "cosmopolita",
    "mosca", # annoyed
    "fantasma",
    "ultra",
    "nómada",
    "entusiasta",
    "boricua",
    "antidroga",
    "iconoclasta",
    "antisistema",
    "patriota",
    "okupa",    # not sure... is both M/F as NOUN, presumably also as ADJ?
    "ratonera", # rat trap... ratonero is a buzzard, so if used as ADJ, probably has to keep -a?
    "déspota",
    "biplaza",
    "toquilla",   # a type of straw, used as "straw hat"
    "masoca",     # masochist
    "hembra",     # female
    "punta",      # invariable for "peak velocity" etc?
    "multimedia",
    "basura",     # used in a phrase like "comida basura" as an ADJ
    "profamilia",
    "tierra-tierra",   # is this even a single word?  ground to ground (as in, missile)
    "supermosca",  # for example, in the context "campeón supermosca", where it is not inflected
    "antipersona", # minas antipersonas.  looked for "antipersono" and could not find anything like it
    # such as in the phrase "sangre mangosta", "mongoose blood", such
    # as you would inject into someone to make them a mongoose-powered superhero
    "mangosta",
    # in context: "el hijo honoriscausa".  not inflected in the text.
    # not in any dictionary.  described by ChatGPT as a Latin insertion.
    "honoriscausa",
    # specifically the Colombian dish "bandeja paisa"
    # this is getting in the weeds.  apparently there is also a plural form
    # "bandejas paisas" but i can't find a "paiso" attested to
    "paisa",

    # many demonyms are common gender / invariable for gender
    "persa",
    "belga",
    "francobelga",
    "croata",
    "hitita",
    "chiíta",
    "marroquí",
    "iraní",
    "iraquí",
    "israelí",
    "hindú",
    "zulú",
    "lisboeta",
    "blaugrana", # barcelona fan
    "azulgrana",
    # from San Salvador.  An example sentence in the online dictionary I found was
    #  "Juego en un equipo donostiarra"
    #  https://www.spanishdict.com/translate/donostiarra
    "donostiarra",
    "vietnamita",
    "celta",
    "maya",
    "moscovita",
    "tlaxcalteca",
    "chipriota",   # from Cyprus
    "carioca",     # from Rio de Janiero
    "inca",
    "azteca",
    "olmeca",
    "jesuíta",
    "purépecha",   # a specific Mexican minority
    "quechua",
    "quichua",
    "angora",
    "nahua",
    "etarra",   # member of ETA
    "myanma",
    "maronita", # Maronite Church in Lebanon
    "aymara",   # an Indigenous group of the Andes
    "semita",   # semite, has only -a forms
    "antisemita",
    "pastún",
    "hindú",
    # this I am not sure of
    # according to ChatGPT: It is almost certainly referring to the ancient Italic tribe: Bruttii
    "brucia",
    # also not sure of.  does "omeyo" exist?
    "omeya",
    # perj. for gay
    "marica",


    "crema",    # the color cream, used as ADJ
    "rosa",
    "naranja",
    "violeta",
    "lavanda",
    "ultravioleta",
]

def is_special_case_gender(word):
    if word in special_cases:
        return True
    if word.endswith("ista"):
        return True
    if word.endswith("ísta"):
        return True
    if word.endswith("ícola"):
        return True
    if word.endswith("cida"):
        return True
    if word.endswith("crata"):
        return True
    if word.endswith("arca"):
        return True
    if word.endswith("e"):
        return True
    return False

def print_inconsistent_lemmas(adjectives):
    for adj, candidates in adjectives.items():
        if len(candidates) > 1:
            print(adj, candidates)

def get_proposed_replacements(adjectives, known_only=True):
    replacements = {}
    for adj, candidates in adjectives.items():
        if len(candidates) == 2:
            candidates = sorted(candidates)
            if not candidates[0].endswith("a") or not candidates[1].endswith("o") or candidates[0][:-1] != candidates[1][:-1]:
                continue
            #print(adj, candidates)
            if known_only and candidates[1] not in adjectives and "%ss" % candidates[1] not in adjectives:
                continue
            if is_special_case_gender(candidates[0]):
                print("WARNING: check %s" % candidates[0])
                continue
            # now we have a male form which exists as an ADJ
            # but the feminine form was used as a lemma
            # we have high confidence this should be updated
            #print("  ", adj, candidates[1])
            replacements[adj] = candidates[1]
        elif len(candidates) == 1:
            candidate = next(iter(candidates))
            if not candidate.endswith("a") and not candidate.endswith("as"):
                continue
            if is_special_case_gender(candidate):
                continue
            if is_special_case_gender(adj.lower()) and candidate.endswith("o"):
                print("WARNING: check %s" % candidates[0])
                continue
            if candidate.endswith("a"):
                candidate = candidate[:-1] + "o"
            elif candidate.endswith("as"):
                candidate = candidate[:-2] + "o"
            if candidate in adjectives:
                replacements[adj] = candidate
    return replacements

def update_adjectives(adjectives, known_only=True):
    replacements = get_proposed_replacements(adjectives, known_only)
    print("Number of proposed replacements: %d" % len(replacements))
    for filename in filenames:
        print(filename)
        doc = CoNLL.conll2doc(filename, ignore_gapping=False)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos != 'ADJ':
                    continue
                word.lemma = replacements.get(word.text, word.lemma)
        CoNLL.write_doc2conll(doc, filename)

def search_a_ending_adjectives(adjectives):
    for adj, candidates in adjectives.items():
        if not adj.endswith("a") and not adj.endswith("as"):
            continue
        if len(candidates) != 1:
            continue
        candidate = list(candidates)[0]
        if not candidate.endswith("a"):
            if is_special_case_gender(candidate):
                print("WARNING: check %s" % candidate)
            continue
        if is_special_case_gender(candidate):
            continue
        if adj.lower() == candidate or "%ss" % candidate == adj.lower():
            print(adj)
        else:
            print("%s ... %s" % (adj, candidate))
        
FILENAMES = [
    "extern_data/ud2/git/UD_Spanish-AnCora/es_ancora-ud-train.conllu",
    "extern_data/ud2/git/UD_Spanish-AnCora/es_ancora-ud-dev.conllu",
    "extern_data/ud2/git/UD_Spanish-AnCora/es_ancora-ud-test.conllu",
    "extern_data/ud2/git/UD_Spanish-GSD/es_gsd-ud-train.conllu",
    "extern_data/ud2/git/UD_Spanish-GSD/es_gsd-ud-dev.conllu",
    "extern_data/ud2/git/UD_Spanish-GSD/es_gsd-ud-test.conllu",
    "extern_data/ud2/git/UD_Spanish-PUD/es_pud-ud-test.conllu",
]

adj_ending = re.compile("^.*[aAoO]([sS])?$")

def load_adjectives_list(filenames):
    adjectives = defaultdict(set)

    for filename in filenames:
        print(filename)

        doc = CoNLL.conll2doc(filename, ignore_gapping=False)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos != 'ADJ':
                    continue
                if word.feats and "Typo=Yes" in word.feats:
                    continue
                if word.feats and "Foreign=Yes" in word.feats:
                    continue
                #if adj_ending.match(word.text):
                adjectives[word.text].add(word.lemma)

    print(len(adjectives))
    return adjectives

def main():
    for filename in FILENAMES:
        assert os.path.exists(filename)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, choices=['pud', 'gsd', 'ancora'], help='Only use this dataset')
    args = parser.parse_args()

    if args.dataset:
        filenames = [x for x in FILENAMES if args.dataset in x]
    else:
        filenames = FILENAMES

    adjectives = load_adjectives_list(filenames)
    print_inconsistent_lemmas(adjectives)
    #update_adjectives(adjectives, known_only=False)
    #search_a_ending_adjectives(adjectives)
    #remove_gender_features()

if __name__ == '__main__':
    main()

