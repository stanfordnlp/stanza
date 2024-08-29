import argparse

from stanza.models.constituency import transition_sequence
from stanza.models.constituency import tree_reader
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency.utils import verify_transitions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default="data/constituency/en_ptb3_train.mrg", help='Input file for data loader.')
    parser.add_argument('--transition_scheme', default=TransitionScheme.IN_ORDER, type=lambda x: TransitionScheme[x.upper()],
                        help='Transition scheme to use.  {}'.format(", ".join(x.name for x in TransitionScheme)))
    parser.add_argument('--reversed', default=False, action='store_true', help='Do the transition sequence reversed')
    parser.add_argument('--iterations', default=30, type=int, help='How many times to iterate, such as if doing a cProfile')
    args = parser.parse_args()
    args = vars(args)

    train_trees = tree_reader.read_treebank(args['train_file'])
    unary_limit = max(t.count_unary_depth() for t in train_trees) + 1
    train_sequences, train_transitions = transition_sequence.convert_trees_to_sequences(train_trees, "training", args['transition_scheme'], args['reversed'])
    root_labels = Tree.get_root_labels(train_trees)
    for i in range(args['iterations']):
        verify_transitions(train_trees, train_sequences, args['transition_scheme'], unary_limit, args['reversed'], "train", root_labels)

if __name__ == '__main__':
    main()
