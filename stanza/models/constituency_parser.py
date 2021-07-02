import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--transition_embedding_dim', type=int, default=20, help="Embedding size for a transition")
    parser.add_argument('--hidden_size', type=int, default=100, help="Size of the output layers of each of the three stacks")

    args = parser.parse_args(args=args)
    args = vars(args)
    return args
