"""
Set a few common flags for peft uage
"""


def add_peft_args(parser):
    """
    Add common default flags to an argparse
    """
    parser.add_argument('--lora_rank', type=int, default=64, help="Rank of a LoRA approximation")
    parser.add_argument('--lora_alpha', type=int, default=128, help="Alpha of a LoRA approximation")
    parser.add_argument('--lora_dropout', type=float, default=0.1, help="Dropout for the LoRA approximation")


