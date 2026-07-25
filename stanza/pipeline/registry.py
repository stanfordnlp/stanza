from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stanza.pipeline.processor import Processor, ProcessorVariant

# these two get filled by register_processor
NAME_TO_PROCESSOR_CLASS: dict[str, type[Processor]] = {}
PIPELINE_NAMES: list[str] = []

# this gets filled by register_processor_variant
PROCESSOR_VARIANTS: defaultdict[
    str,
    dict[str, type[ProcessorVariant]],
] = defaultdict(dict)
