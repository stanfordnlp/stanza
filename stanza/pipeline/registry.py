from collections import defaultdict

# these two get filled by register_processor
NAME_TO_PROCESSOR_CLASS = dict()
PIPELINE_NAMES = []

# this gets filled by register_processor_variant
PROCESSOR_VARIANTS = defaultdict(dict)
