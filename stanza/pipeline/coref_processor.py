"""
Processor that attaches coref annotations to a document
"""

from stanza.models.coref.coref_chain import CorefMention, CorefChain
from stanza.models.coref.model import CorefModel
from stanza.models.coref.predict import build_doc

from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

@register_processor(COREF)
class CorefProcessor(UDProcessor):
    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([COREF])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, pipeline, device):
        # set up model
        # currently, the model has everything packaged in it
        # (except its config)
        # TODO: separate any pretrains if possible
        # TODO: add device parameter to the load mechanism
        config_update = {'log_norms': False}
        model = CorefModel.load_model(path=config['model_path'],
                                      ignore={"bert_optimizer", "general_optimizer",
                                              "bert_scheduler", "general_scheduler"},
                                      config_update=config_update)
        if config.get('batch_size', None):
            model.config.a_scoring_batch_size = int(config['batch_size'])
        model.training = False

        self._model = model

    def process(self, document):
        sentences = document.sentences

        cased_words = []
        sent_ids = []
        word_pos = []
        for sent_idx, sentence in enumerate(sentences):
            for word_idx, word in enumerate(sentence.words):
                cased_words.append(word.text)
                sent_ids.append(sent_idx)
                word_pos.append(word_idx)

        coref_input = {
            "document_id": "wb_doc_1",
            "cased_words": cased_words,
            "sent_id": sent_ids
        }
        coref_input = build_doc(coref_input, self._model)
        results = self._model.run(coref_input)
        clusters = []
        for span_cluster in results.span_clusters:
            if len(span_cluster) == 0:
                continue
            span_cluster = sorted(span_cluster)

            # treat the longest span as the representative
            # break ties using the first one
            max_len = 0
            best_span = None
            for span_idx, span in enumerate(span_cluster):
                if span[1] - span[0] > max_len:
                    max_len = span[1] - span[0]
                    best_span = span_idx

            mentions = []
            for span in span_cluster:
                sent_id = sent_ids[span[0]]
                if sent_ids[span[1]] != sent_id:
                    raise ValueError("The coref model predicted a span that crossed two sentences!  Please send this example to us on our github")
                start_word = word_pos[span[0]]
                end_word = word_pos[span[1]]
                mentions.append(CorefMention(sent_id, start_word, end_word))
            representative = mentions[best_span]

            chain = CorefChain(mentions, representative)
            clusters.append(chain)

        document.coref = clusters
        return document
