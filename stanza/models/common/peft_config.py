"""
Set a few common flags for peft uage
"""


TRANSFORMER_LORA_RANK = {}
DEFAULT_LORA_RANK = 64

TRANSFORMER_LORA_ALPHA = {}
DEFAULT_LORA_ALPHA = 128

TRANSFORMER_LORA_DROPOUT = {}
DEFAULT_LORA_DROPOUT = 0.1

def add_peft_args(parser):
    """
    Add common default flags to an argparse
    """
    parser.add_argument('--lora_rank', type=int, default=None, help="Rank of a LoRA approximation.  Default will be %d or a model-specific parameter" % DEFAULT_LORA_RANK)
    parser.add_argument('--lora_alpha', type=int, default=None, help="Alpha of a LoRA approximation.  Default will be %d or a model-specific parameter" % DEFAULT_LORA_ALPHA)
    parser.add_argument('--lora_dropout', type=float, default=None, help="Dropout for the LoRA approximation.  Default will be %s or a model-specific parameter" % DEFAULT_LORA_DROPOUT)


def resolve_peft_args(args):
    if not hasattr(args, 'bert_model'):
        return

    if args.lora_rank is None:
        args.lora_rank = TRANSFORMER_LORA_RANK.get(args.bert_model, DEFAULT_LORA_RANK)

    if args.lora_alpha is None:
        args.lora_alpha = TRANSFORMER_LORA_ALPHA.get(args.bert_model, DEFAULT_LORA_ALPHA)

    if args.lora_dropout is None:
        args.lora_dropout = TRANSFORMER_LORA_DROPOUT.get(args.bert_model, DEFAULT_LORA_DROPOUT)
