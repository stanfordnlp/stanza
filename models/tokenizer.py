from models.common import param, utils

from models.tokenize.trainer import TokenizerTrainer
from models.tokenize.utils import train, evaluate, load_mwt_dict, Env

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str, help="Input plaintext file")
    parser.add_argument('--label_file', type=str, default=None, help="Character-level label file")
    parser.add_argument('--json_file', type=str, default=None, help="JSON file with pre-chunked units")
    parser.add_argument('--mwt_json_file', type=str, default=None, help="JSON file for MWT expansions")
    parser.add_argument('--conll_file', type=str, default=None, help="CoNLL file for output")
    parser.add_argument('--vocab_file', type=str, default=None, help="Vocab file")
    parser.add_argument('--dev_txt_file', type=str, help="(Train only) Input plaintext file for the dev set")
    parser.add_argument('--dev_label_file', type=str, default=None, help="(Train only) Character-level label file for the dev set")
    parser.add_argument('--dev_json_file', type=str, default=None, help="(Train only) JSON file with pre-chunked units for the dev set")
    parser.add_argument('--dev_conll_gold', type=str, default=None, help="(Train only) CoNLL-U file for the dev set for early stopping")
    parser.add_argument('--lang', type=str, help="Language")
    parser.add_argument('--shorthand', type=str, help="UD treebank shorthand")

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--emb_dim', type=int, default=30, help="Dimension of unit embeddings")
    parser.add_argument('--hidden_dim', type=int, default=100, help="Dimension of hidden units")
    parser.add_argument('--conv_filters', type=str, default="1,5,9,,1,5,9", help="Configuration of conv filters. ,, separates layers and , separates filter sizes in the same layer.")
    parser.add_argument('--residual', action='store_true', help="Add linear residual connections")
    parser.add_argument('--hierarchical', action='store_true', help="\"Hierarchical\" RNN tokenizer")
    parser.add_argument('--no-rnn', dest='rnn', action='store_false', help="Use CNN tokenizer")
    parser.add_argument('--input_dropout', action='store_true', help="Dropout input embeddings as well")
    parser.add_argument('--aux_clf', type=float, default=0.0, help="Strength for auxiliary classifiers; default 0 (don't use auxiliary classifiers)")
    parser.add_argument('--merge_aux_clf', action='store_true', help="Merge prediction from auxiliary classifiers with final classifier output")
    parser.add_argument('--conv_res', type=str, default=None, help="Convolutional residual layers for the RNN")
    parser.add_argument('--rnn_layers', type=int, default=1, help="Layers of RNN in the tokenizer")

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm to clip to")
    parser.add_argument('--anneal', type=float, default=.9, help="Anneal the learning rate by this amount when dev performance deteriorate")
    parser.add_argument('--anneal_after', type=int, default=0, help="Anneal the learning rate no earlier than this step")
    parser.add_argument('--lr0', type=float, default=2e-3, help="Initial learning rate")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout probability")
    parser.add_argument('--unit_dropout', type=float, default=0.0, help="Unit dropout probability")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--max_seqlen', type=int, default=100, help="Maximum sequence length to consider at a time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size to use")
    parser.add_argument('--epochs', type=int, default=10, help="Total epochs to train the model for")
    parser.add_argument('--steps', type=int, default=None, help="Steps to train the model for, if unspecified use epochs")
    parser.add_argument('--report_steps', type=int, default=20, help="Update step interval to report loss")
    parser.add_argument('--shuffle_steps', type=int, default=0, help="Step interval to shuffle each paragragraph in the generator")
    parser.add_argument('--eval_steps', type=int, default=200, help="Step interval to evaluate the model on the dev set for early stopping")
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--save_dir', type=str, default='saved_models', help="Directory to save models in")
    parser.add_argument('--no_cuda', dest="cuda", action="store_false")
    parser.add_argument('--best_param', action='store_true', help='Train with best language-specific parameters.')

    args = parser.parse_args()

    args = vars(args)
    args['feat_funcs'] = ['space_before', 'capitalized', 'all_caps', 'numeric']
    args['feat_dim'] = len(args['feat_funcs'])
    args['save_name'] = "{}/{}".format(args['save_dir'], args['save_name']) if args['save_name'] is not None else '{}/{}_tokenizer.pkl'.format(args['save_dir'], args['shorthand'])

    # activate param manager and save config
    param_manager = param.ParamManager('params/mwt', args['shorthand'])
    if args['best_param']: # use best param in file, otherwise use command line params
        args = param_manager.load_to_args(args)
    utils.save_config(args, '{}/{}_config.json'.format(args['save_dir'], args['shorthand']))

    env = Env(args)
    args['vocab_size'] = len(env.vocab)

    env.trainer = TokenizerTrainer(args)
    env.param_manager = param_manager
    trainer = env.trainer

    env.mwt_dict = load_mwt_dict(args['mwt_json_file'])

    if args['mode'] == 'train':
        train(env)
    else:
        evaluate(env)
