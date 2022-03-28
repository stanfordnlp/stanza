"""
Parses a Tiger dataset to PTB

Also handles problems specific for the Arboretum treebank.

- validation errors in the XML: 
  -- there is a "&" instead of an "&amp;" early on
  -- there are tags "<{note}>" and "<{parentes-udeladt}>" which may or may not be relevant,
     but are definitely not properly xml encoded
- trees with stranded nodes.  5 trees have links to words in a different tree.
  those trees are skipped
- trees with empty nodes.  58 trees have phrase nodes with no leaves.
  those trees are skipped
- trees with missing words.  134 trees have words in the text which aren't in the tree
  those trees are also skipped
- trees with categories not in the category directory
  for example, intj... replaced with fcl?
  most of these are replaced with what might be a sensible replacement
- trees with labels that don't have an obvious replacement
  these trees are eliminated, 4 total
- underscores in words.  those words are split into multiple words
  the tagging is not going to be ideal, but the first step of training
  a parser is usually to retag the words anyway, so this should be okay
- tree 14729 is really weirdly annotated.  skipped
- 5373 trees total have non-projective constituents.  These don't work
  with the stanza parser...  in order to work around this, we rearrange
  them when possible.
    ((X Z) Y1 Y2 ...) -> (X Y1 Y2 Z)          this rearranges 3021 trees
    ((X Z1 ...) Y1 Y2 ...) -> (X Y1 Y2 Z)     this rearranges  403 trees
    ((X Z1 ...) (tag Y1) ...) -> (X (Y1) Z)   this rearranges 1258 trees

  A couple examples of things which get rearranged
  (limited in scope and without the words to avoid breaking our license):

(vp (v-fin s4_6) (conj-c s4_8) (v-fin s4_9)) (pron-pers s4_7)
-->
(vp (v-fin s4_6) (pron-pers s4_7) (conj-c s4_8) (v-fin s4_9))

(vp (v-fin s1_2) (v-pcp2 s1_4)) (adv s1_3)
-->
(vp (v-fin s1_2) (adv s1_3) (v-pcp2 s1_4))

  This process leaves behind 691 trees.  In some cases, the
  non-projective structure is at a higher level than the attachment.
  In others, there are nested non-projectivities that are not
  rearranged by the above pattern.  A couple examples:

here, the 3-7 nonprojectivity has the 7 in a nested structure
(s
 (par
  (n s206_1)
  (pu s206_2)
  (fcl
   (fcl
    (pron-pers s206_3)
    (fcl (pron-pers s206_7) (adv s206_8) (v-fin s206_9)))
   (vp (v-fin s206_4) (v-inf s206_6))
   (pron-pers s206_5))
  (pu s206_10)))

here, 11 is attached at a higher level than 12 & 13
(s
 (fcl
  (icl
   (np
    (adv s223_1)
    (np
     (n s223_2)
     (pp
      (prp s223_3)
      (par
       (adv s223_4)
       (prop s223_5)
       (pu s223_6)
       (prop s223_7)
       (conj-c s223_8)
       (np (adv s223_9) (prop s223_10))))))
   (vp (infm s223_12) (v-inf s223_13)))
  (v-fin s223_11)
  (pu s223_14)))

even if we moved _6 between 2 and 7, we'd then have a completely flat
structure when moving 3..5 inside
(s
 (fcl
  (xx s499_1)
  (np
   (pp (pron-pers s499_2) (prp s499_7))
   (n s499_6))
  (v-fin s499_3) (adv s499_4) (adv s499_5) (pu s499_8)))

"""


from collections import namedtuple
import io
import xml.etree.ElementTree as ET

from tqdm import tqdm

from stanza.models.constituency.parse_tree import Tree
from stanza.server import tsurgeon

def read_xml_file(input_filename):
    """
    Convert an XML file into a list of trees - each <s> becomes its own object
    """
    print("Reading {}".format(input_filename))
    with open(input_filename, encoding="utf-8") as fin:
        lines = fin.readlines()

    sentences = []
    current_sentence = []
    in_sentence = False
    for line_idx, line in enumerate(lines):
        if line.startswith("<s "):
            if len(current_sentence) > 0:
                raise ValueError("Found the start of a sentence inside an existing sentence, line {}".format(line_idx))
            in_sentence = True

        if in_sentence:
            current_sentence.append(line)

        if line.startswith("</s>"):
            assert in_sentence
            current_sentence = [x.replace("<{parentes-udeladt}>", "") for x in current_sentence]
            current_sentence = [x.replace("<{note}>", "") for x in current_sentence]
            sentences.append("".join(current_sentence))
            current_sentence = []
            in_sentence = False

    assert len(current_sentence) == 0

    xml_sentences = []
    for sent_idx, text in enumerate(sentences):
        sentence = io.StringIO(text)
        try:
            tree = ET.parse(sentence)
            xml_sentences.append(tree)
        except ET.ParseError as e:
            raise ValueError("Failed to parse sentence {}".format(sent_idx))

    return xml_sentences

Word = namedtuple('Word', ['word', 'tag'])
Node = namedtuple('Node', ['label', 'children'])

class BrokenLinkError(ValueError):
    def __init__(self, error):
        super(BrokenLinkError, self).__init__(error)

def process_nodes(root_id, words, nodes, visited):
    """
    Given a root_id, a map of words, and a map of nodes, construct a Tree

    visited is a set of string ids and mutates over the course of the recursive call
    """
    if root_id in visited:
        raise ValueError("Loop in the tree!")
    visited.add(root_id)

    if root_id in words:
        word = words[root_id]
        # big brain move: put the root_id here so we can use that to
        # check the sorted order when we are done
        word_node = Tree(label=root_id)
        tag_node = Tree(label=word.tag, children=word_node)
        return tag_node
    elif root_id in nodes:
        node = nodes[root_id]
        children = [process_nodes(child, words, nodes, visited) for child in node.children]
        return Tree(label=node.label, children=children)
    else:
        raise BrokenLinkError("Unknown id! {}".format(root_id))

def check_words(tree, tsurgeon_processor):
    """
    Check that the words of a sentence are in order

    If they are not, this applies a tsurgeon to rearrange simple cases
    The tsurgeon looks at the gap between words, eg _3 to _7, and looks
    for the words between, such as _4 _5 _6.  if those words are under
    a node at the same level as the 3-7 node and does not include any
    other nodes (such as _8), that subtree is moved to between _3 and _7

    Example:

    (vp (v-fin s4_6) (conj-c s4_8) (v-fin s4_9)) (pron-pers s4_7)
    -->
    (vp (v-fin s4_6) (pron-pers s4_7) (conj-c s4_8) (v-fin s4_9))
    """
    while True:
        words = tree.leaf_labels()
        indices = [int(w.split("_", 1)[1]) for w in words]
        for word_idx, word_label in enumerate(indices):
            if word_idx != word_label - 1:
                break
        else:
            # if there are no weird indices, keep the tree
            return tree

        sorted_indices = sorted(indices)
        if indices == sorted_indices:
            raise ValueError("Skipped index!  This should already be accounted for  {}".format(tree))

        if word_idx == 0:
            return None

        prefix = words[0].split("_", 1)[0]
        prev_idx = word_idx - 1
        prev_label = indices[prev_idx]
        missing_words = ["%s_%d" % (prefix, x) for x in range(prev_label + 1, word_label)]
        missing_words = "|".join(missing_words)
        #move_tregex = "%s > (__=home > (__=parent > __=grandparent)) . (%s > (__=move > =grandparent))" % (words[word_idx], "|".join(missing_words))
        move_tregex = "%s > (__=home > (__=parent << %s $+ (__=move <<, %s <<- %s)))" % (words[word_idx], words[prev_idx], missing_words, missing_words)
        move_tsurgeon = "move move $+ home"
        modified = tsurgeon_processor.process(tree, move_tregex, move_tsurgeon)[0]
        if modified == tree:
            # this only happens if the desired fix didn't happen
            #print("Failed to process:\n  {}\n  {} {}".format(tree, prev_label, word_label))
            return None

        tree = modified

def replace_words(tree, words):
    """
    Remap the leaf words given a map of the labels we expect in the leaves
    """
    leaves = tree.leaf_labels()
    new_words = [words[w].word for w in leaves]
    new_tree = tree.replace_words(new_words)
    return new_tree

def process_tree(sentence):
    """
    Convert a single ET element representing a Tiger tree to a parse tree
    """
    sentence = sentence.getroot()
    sent_id = sentence.get("id")
    if sent_id is None:
        raise ValueError("Tree {} does not have an id".format(sent_idx))
    if len(sentence) > 1:
        raise ValueError("Longer than expected number of items in {}".format(sent_id))
    graph = sentence.find("graph")
    if not graph:
        raise ValueError("Unexpected tree structure in {} : top tag is not 'graph'".format(sent_id))

    root_id = graph.get("root")
    if not root_id:
        raise ValueError("Tree has no root id in {}".format(sent_id))

    terminals = graph.find("terminals")
    if not terminals:
        raise ValueError("No terminals in tree {}".format(sent_id))
    # some Arboretum graphs have two sets of nonterminals,
    # apparently intentionally, so we ignore that possible error
    nonterminals = graph.find("nonterminals")
    if not nonterminals:
        raise ValueError("No nonterminals in tree {}".format(sent_id))

    # read the words.  the words have ids, text, and tags which we care about
    words = {}
    for word in terminals:
        if word.tag == 'parentes-udeladt' or word.tag == 'note':
            continue
        if word.tag != "t":
            raise ValueError("Unexpected tree structure in {} : word with tag other than t".format(sent_id))
        word_id = word.get("id")
        if not word_id:
            raise ValueError("Word had no id in {}".format(sent_id))
        word_text = word.get("word")
        if not word_text:
            raise ValueError("Word had no text in {}".format(sent_id))
        word_pos = word.get("pos")
        if not word_pos:
            raise ValueError("Word had no pos in {}".format(sent_id))
        words[word_id] = Word(word_text, word_pos)

    # read the nodes.  the nodes have ids, labels, and children
    # some of the edges are labeled "secedge".  we ignore those
    nodes = {}
    for nt in nonterminals:
        if nt.tag != "nt":
            raise ValueError("Unexpected tree structure in {} : node with tag other than nt".format(sent_id))
        nt_id = nt.get("id")
        if not nt_id:
            raise ValueError("NT has no id in {}".format(sent_id))
        nt_label = nt.get("cat")
        if not nt_label:
            raise ValueError("NT has no label in {}".format(sent_id))

        children = []
        for child in nt:
            if child.tag != "edge" and child.tag != "secedge":
                raise ValueError("NT has unexpected child in {} : {}".format(sent_id, child.tag))
            if child.tag == "edge":
                child_id = child.get("idref")
                if not child_id:
                    raise ValueError("Child is missing an id in {}".format(sent_id))
                children.append(child_id)
        nodes[nt_id] = Node(nt_label, children)

    if root_id not in nodes:
        raise ValueError("Could not find root in nodes in {}".format(sent_id))

    tree = process_nodes(root_id, words, nodes, set())
    return tree, words

def word_sequence_missing_words(tree):
    """
    Check if the word sequence is missing words

    Some trees skip labels, such as
      (s (fcl (pron-pers s16817_1) (v-fin s16817_2) (prp s16817_3) (pp (prp s16817_5) (par (n s16817_6) (conj-c s16817_7) (n s16817_8))) (pu s16817_9)))
    but in these cases, the word is present in the original text and simply not attached to the tree
    """
    words = tree.leaf_labels()
    indices = [int(w.split("_")[1]) for w in words]
    indices = sorted(indices)
    for idx, label in enumerate(indices):
        if label != idx + 1:
            return True
    return False

WORD_TO_PHRASE = {
    "art": "advp",    # "en smule" is the one time this happens. it is used as an advp elsewhere
    "adj": "adjp",
    "adv": "advp",
    "conj": "cp",
    "intj": "fcl",    # not sure?  seems to match "hold kæft" when it shows up
    "n": "np",
    "num": "np",      # would prefer something like QP from PTB
    "pron": "np",     # ??
    "prop": "np",
    "prp": "pp",
    "v": "vp",
}

def split_underscores(tree):
    assert not tree.is_leaf(), "Should never reach a leaf in this code path"

    if tree.is_preterminal():
        return tree

    children = tree.children
    new_children = []
    for child in children:
        if child.is_preterminal():
            if '_' not in child.children[0].label:
                new_children.append(child)
                continue

            if child.label.split("-")[0] not in WORD_TO_PHRASE:
                raise ValueError("SPLITTING {}".format(child))
            pieces = []
            for piece in child.children[0].label.split("_"):
                # This may not be accurate, but we already retag the treebank anyway
                if len(piece) == 0:
                    raise ValueError("A word started or ended with _")
                pieces.append(Tree(child.label, Tree(piece)))
            new_children.append(Tree(WORD_TO_PHRASE[child.label.split("-")[0]], pieces))
        else:
            new_children.append(split_underscores(child))

    return Tree(tree.label, new_children)

REMAP_LABELS = {
    "adj": "adjp",
    "adv": "advp",
    "intj": "fcl",
    "n": "np",
    "num": "np",     # again, a dedicated number node would be better, but there are only a few "num" labeled
    "prp": "pp",
}


def has_weird_constituents(tree):
    """
    Eliminate a few trees with weird labels

    Eliminate p?  there are only 3 and they have varying structure underneath
    Also cl, since I have no idea how to label it and it only excludes 1 tree
    """
    labels = Tree.get_unique_constituent_labels(tree)
    if "p" in labels or "cl" in labels:
        return True
    return False

def convert_tiger_treebank(input_filename):
    sentences = read_xml_file(input_filename)

    unfixable = 0
    dangling = 0
    broken_links = 0
    missing_words = 0
    weird_constituents = 0
    trees = []

    with tsurgeon.Tsurgeon() as tsurgeon_processor:
        for sent_idx, sentence in enumerate(tqdm(sentences)):
            try:
                tree, words = process_tree(sentence)

                if not tree.all_leaves_are_preterminals():
                    dangling += 1
                    continue

                if word_sequence_missing_words(tree):
                    missing_words += 1
                    continue

                tree = check_words(tree, tsurgeon_processor)
                if tree is None:
                    unfixable += 1
                    continue

                if has_weird_constituents(tree):
                    weird_constituents += 1
                    continue

                tree = replace_words(tree, words)
                tree = split_underscores(tree)
                tree = tree.remap_constituent_labels(REMAP_LABELS)
                trees.append(tree)
            except BrokenLinkError as e:
                # the get("id") would have failed as a different error type if missing,
                # so we can safely use it directly like this
                broken_links += 1
                # print("Unable to process {} because of broken links: {}".format(sentence.getroot().get("id"), e))

    print("Found {} trees with empty nodes".format(dangling))
    print("Found {} trees with unattached words".format(missing_words))
    print("Found {} trees with confusing constituent labels".format(weird_constituents))
    print("Not able to rearrange {} nodes".format(unfixable))
    print("Unable to handle {} trees because of broken links, eg names in another tree".format(broken_links))
    print("Parsed {} trees from {}".format(len(trees), input_filename))
    return trees

def main():
    treebank = convert_tiger_treebank("extern_data/constituency/danish/W0084/arboretum.tiger/arboretum.tiger")

if __name__ == '__main__':
    main()
