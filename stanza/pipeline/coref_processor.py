"""
Processor that attaches coref annotations to a document
"""

from stanza.models.common.utils import misc_to_space_after
from stanza.models.coref.coref_chain import CorefMention, CorefChain

from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor

def extract_text(document, sent_id, start_word, end_word):
    sentence = document.sentences[sent_id]
    tokens = []

    # the coref model indexes the words from 0,
    # whereas the ids we are looking at on the tokens start from 1
    # here we will switch to ID space
    start_word = start_word + 1
    end_word = end_word + 1

    # For each position between start and end word:
    # If a word is part of an MWT, and the entire token
    # is inside the range, we use that Token's text for that span
    # This will let us easily handle words which are split into pieces
    # Otherwise, we only take the text of the word itself
    next_idx = start_word
    while next_idx < end_word:
        word = sentence.words[next_idx-1]
        parent_token = word.parent
        if isinstance(parent_token.id, int) or len(parent_token.id) == 1:
            tokens.append(parent_token)
            next_idx += 1
        elif parent_token.id[0] >= start_word and parent_token.id[1] < end_word:
            tokens.append(parent_token)
            next_idx = parent_token.id[1] + 1
        else:
            tokens.append(word)
            next_idx += 1

    # We use the SpaceAfter or SpacesAfter attribute on each Word or Token
    # we chose in the above loop to separate the text pieces
    text = []
    for token in tokens:
        text.append(token.text)
        text.append(misc_to_space_after(token.misc))
    # the last token space_after will be discarded
    # so that we don't have stray WS at the end of the mention text
    text = text[:-1]
    return "".join(text)


@register_processor(COREF)
class CorefProcessor(UDProcessor):
    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([COREF])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, pipeline, device):
        try:
            from stanza.models.coref.model import CorefModel
        except ImportError:
            raise ImportError("Please install the transformers and peft libraries before using coref! Try `pip install -e .[transformers]`.")

        # set up model
        # currently, the model has everything packaged in it
        # (except its config)
        # TODO: separate any pretrains if possible
        # TODO: add device parameter to the load mechanism
        config_update = {'log_norms': False,
                         'device': device}
        model = CorefModel.load_model(path=config['model_path'],
                                      ignore={"bert_optimizer", "general_optimizer",
                                              "bert_scheduler", "general_scheduler"},
                                      config_update=config_update,
                                      foundation_cache=pipeline.foundation_cache)
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
        coref_input = self._model.build_doc(coref_input)
        results = self._model.run(coref_input)
        clusters = []
        for span_cluster in results.span_clusters:
            if len(span_cluster) == 0:
                continue
            span_cluster = sorted(span_cluster)

            for span in span_cluster:
                # check there are no sentence crossings before
                # manipulating the spans, since we will expect it to
                # be this way for multiple usages of the spans
                sent_id = sent_ids[span[0]]
                if sent_ids[span[1]-1] != sent_id:
                    raise ValueError("The coref model predicted a span that crossed two sentences!  Please send this example to us on our github")

            # treat the longest span as the representative
            # break ties using the first one
            # IF there is the POS processor, and it adds upos tags
            # to the sentence, ties are broken first by maximum
            # number of UPOS and then earliest in the document
            max_len = 0
            best_span = None
            max_propn = 0
            for span_idx, span in enumerate(span_cluster):
                sent_id = sent_ids[span[0]]
                sentence = sentences[sent_id]
                start_word = word_pos[span[0]]
                # fiddle -1 / +1 so as to avoid problems with coref
                # clusters that end at exactly the end of a document
                end_word = word_pos[span[1]-1] + 1
                # very UD specific test for most number of proper nouns in a mention
                # will do nothing if POS is not active (they will all be None)
                num_propn = sum(word.pos == 'PROPN' for word in sentence.words[start_word:end_word])

                if ((span[1] - span[0] > max_len) or
                    span[1] - span[0] == max_len and num_propn > max_propn):
                    max_len = span[1] - span[0]
                    best_span = span_idx
                    max_propn = num_propn

            mentions = []
            for span in span_cluster:
                sent_id = sent_ids[span[0]]
                start_word = word_pos[span[0]]
                end_word = word_pos[span[1]-1] + 1
                mentions.append(CorefMention(sent_id, start_word, end_word))
            representative = mentions[best_span]
            representative_text = extract_text(document, representative.sentence, representative.start_word, representative.end_word)

            chain = CorefChain(len(clusters), mentions, representative_text, best_span)
            clusters.append(chain)

        document.coref = clusters
        return document
