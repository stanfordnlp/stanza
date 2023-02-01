"""
A depparse processor which converts constituency trees using CoreNLP
"""

from stanza.pipeline._constants import TOKENIZE, CONSTITUENCY, DEPPARSE
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant
from stanza.server.dependency_converter import DependencyConverter

@register_processor_variant(DEPPARSE, 'converter')
class ConverterDepparse(ProcessorVariant):
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE, CONSTITUENCY])

    def __init__(self, config):
        if config['lang'] != 'en':
            raise ValueError("Constituency to dependency converter only works for English")

        # TODO: get classpath from config
        # TODO: close this when finished?
        #   a more involved approach would be to turn the Pipeline into
        #   a context with __enter__ and __exit__
        #   __exit__ would try to free all resources, although some
        #   might linger such as GPU allocations
        #   maybe it isn't worth even trying to clean things up on account of that
        self.converter = DependencyConverter(classpath="$CLASSPATH")
        self.converter.open_pipe()

    def process(self, document):
        return self.converter.process(document)
