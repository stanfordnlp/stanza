"""
Set a few common flags for peft uage
"""


TRANSFORMER_LORA_RANK = {}
DEFAULT_LORA_RANK = 64

TRANSFORMER_LORA_ALPHA = {}
DEFAULT_LORA_ALPHA = 128

TRANSFORMER_LORA_DROPOUT = {}
DEFAULT_LORA_DROPOUT = 0.1

TRANSFORMER_LORA_TARGETS = {}
DEFAULT_LORA_TARGETS = "query,value,output.dense,intermediate.dense"

TRANSFORMER_LORA_SAVE = {}
DEFAULT_LORA_SAVE = ""

def add_peft_args(parser):
    """
    Add common default flags to an argparse
    """
    parser.add_argument('--lora_rank', type=int, default=None, help="Rank of a LoRA approximation.  Default will be %d or a model-specific parameter" % DEFAULT_LORA_RANK)
    parser.add_argument('--lora_alpha', type=int, default=None, help="Alpha of a LoRA approximation.  Default will be %d or a model-specific parameter" % DEFAULT_LORA_ALPHA)
    parser.add_argument('--lora_dropout', type=float, default=None, help="Dropout for the LoRA approximation.  Default will be %s or a model-specific parameter" % DEFAULT_LORA_DROPOUT)
    parser.add_argument('--lora_target_modules', type=str, default=None, help="Comma separated list of LoRA targets.  Default will be '%s' or a model-specific parameter" % DEFAULT_LORA_TARGETS)
    parser.add_argument('--lora_modules_to_save', type=str, default=None, help="Comma separated list of modules to save (eg, fully tune) when using LoRA.  Default will be '%s' or a model-specific parameter" % DEFAULT_LORA_SAVE)

    parser.add_argument('--use_peft', default=False, action='store_true', help="Finetune Bert using peft")

def resolve_peft_args(args, logger, check_bert_finetune=True):
    if not hasattr(args, 'bert_model'):
        return

    if args.lora_rank is None:
        args.lora_rank = TRANSFORMER_LORA_RANK.get(args.bert_model, DEFAULT_LORA_RANK)

    if args.lora_alpha is None:
        args.lora_alpha = TRANSFORMER_LORA_ALPHA.get(args.bert_model, DEFAULT_LORA_ALPHA)

    if args.lora_dropout is None:
        args.lora_dropout = TRANSFORMER_LORA_DROPOUT.get(args.bert_model, DEFAULT_LORA_DROPOUT)

    if args.lora_target_modules is None:
        args.lora_target_modules = TRANSFORMER_LORA_TARGETS.get(args.bert_model, DEFAULT_LORA_TARGETS)
    if not args.lora_target_modules.strip():
        args.lora_target_modules = []
    else:
        args.lora_target_modules = args.lora_target_modules.split(",")

    if args.lora_modules_to_save is None:
        args.lora_modules_to_save = TRANSFORMER_LORA_SAVE.get(args.bert_model, DEFAULT_LORA_SAVE)
    if not args.lora_modules_to_save.strip():
        args.lora_modules_to_save = []
    else:
        args.lora_modules_to_save = args.lora_modules_to_save.split(",")

    if check_bert_finetune and hasattr(args, 'bert_finetune'):
        if args.use_peft and not args.bert_finetune:
            logger.info("--use_peft set.  setting --bert_finetune as well")
            args.bert_finetune = True

def build_peft_wrapper(bert_model, args, logger):
    # Hide import so that the peft dependency is optional
    from peft import LoraConfig, get_peft_model
    logger.debug("Creating lora adapter with rank %d and alpha %d", args['lora_rank'], args['lora_alpha'])
    peft_config = LoraConfig(inference_mode=False,
                             r=args['lora_rank'],
                             target_modules=args['lora_target_modules'],
                             lora_alpha=args['lora_alpha'],
                             lora_dropout=args['lora_dropout'],
                             modules_to_save=args['lora_modules_to_save'],
                             bias="none")

    bert_model = get_peft_model(bert_model, peft_config)
    return bert_model
