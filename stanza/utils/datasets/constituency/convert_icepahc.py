"""
Currently this doesn't function

The goal is simply to demonstrate how to use tsurgeon
"""

from stanza.models.constituency.tree_reader import read_trees, read_treebank
from stanza.server import tsurgeon

TREEBANK = """
( (IP-MAT (NP-SBJ (PRO-N Það-það))
          (BEPI er-vera)
          (ADVP (ADV eiginlega-eiginlega))
          (ADJP (NEG ekki-ekki) (ADJ-N hægt-hægur))
          (IP-INF (TO að-að) (VB lýsa-lýsa))
          (NP-OB1 (N-D tilfinningu$-tilfinning) (D-D $nni-hinn))
          (IP-INF (TO að-að) (VB fá-fá))
          (IP-INF (TO að-að) (VB taka-taka))
          (NP-OB1 (N-A þátt-þáttur))
          (PP (P í-í)
              (NP (D-D þessu-þessi)))
          (, ,-,)
          (VBPI segir-segja)
          (NP-SBJ (NPR-N Sverrir-sverrir) (NPR-N Ingi-ingi))
          (. .-.)))
"""

# Output of the first tsurgeon:
#(ROOT
#  (IP-MAT
#    (NP-SBJ (PRO-N Það))
#    (BEPI er)
#    (ADVP (ADV eiginlega))
#    (ADJP (NEG ekki) (ADJ-N hægt))
#    (IP-INF (TO að) (VB lýsa))
#    (NP-OB1 (N-D tilfinningu$) (D-D $nni))
#    (IP-INF (TO að) (VB fá))
#    (IP-INF (TO að) (VB taka))
#    (NP-OB1 (N-A þátt))
#    (PP
#      (P í)
#      (NP (D-D þessu)))
#    (, ,)
#    (VBPI segir)
#    (NP-SBJ (NPR-N Sverrir) (NPR-N Ingi))
#    (. .)))

# Output of the second operation
#(ROOT
#  (IP-MAT
#    (NP-SBJ (PRO-N Það))
#    (BEPI er)
#    (ADVP (ADV eiginlega))
#    (ADJP (NEG ekki) (ADJ-N hægt))
#    (IP-INF (TO að) (VB lýsa))
#    (NP-OB1 (N-D tilfinningunni))
#    (IP-INF (TO að) (VB fá))
#    (IP-INF (TO að) (VB taka))
#    (NP-OB1 (N-A þátt))
#    (PP
#      (P í)
#      (NP (D-D þessu)))
#    (, ,)
#    (VBPI segir)
#    (NP-SBJ (NPR-N Sverrir) (NPR-N Ingi))
#    (. .)))


treebank = read_trees(TREEBANK)

with tsurgeon.Tsurgeon(classpath="$CLASSPATH") as tsurgeon_processor:
    form_tregex = "/^(.+)-.+$/#1%form=word !< __"
    form_tsurgeon = "relabel word /^.+$/%{form}/"

    noun_det_tregex = "/^N-/ < /^([^$]+)[$]$/#1%noun=noun $+ (/^D-/ < /^[$]([^$]+)$/#1%det=det)"
    noun_det_relabel = "relabel noun /^.+$/%{noun}%{det}/"
    noun_det_prune = "prune det"

    for tree in treebank:
        updated_tree = tsurgeon_processor.process(tree, (form_tregex, form_tsurgeon))[0]
        print("{:P}".format(updated_tree))
        updated_tree = tsurgeon_processor.process(updated_tree, (noun_det_tregex, noun_det_relabel, noun_det_prune))[0]
        print("{:P}".format(updated_tree))
