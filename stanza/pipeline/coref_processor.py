"""
Processor that attaches coref annotations to a document
"""

from __future__ import annotations

from typing import ClassVar, Optional, Sequence, TYPE_CHECKING, TypedDict, Union

import torch

from stanza.models.common.utils import misc_to_space_after
from stanza.models.coref.coref_chain import CorefMention, CorefChain
from stanza.models.coref.config import Config
from stanza.models.coref.const import CorefResult
from stanza.models.common.doc import Document, Token, TokenEntry, Word

from stanza.pipeline._constants import COREF, TOKENIZE
from stanza.pipeline.processor import UDProcessor, register_processor

if TYPE_CHECKING:
    from stanza.models.coref.model import CorefModel
    from stanza.pipeline.core import Pipeline


class _RequiredCorefProcessorConfig(TypedDict):
    model_path: str


class _CorefProcessorConfig(_RequiredCorefProcessorConfig, total=False):
    log_norms: bool
    batch_size: Union[int, str, None]
    use_zeros: Union[bool, str]


_CorefInputValue = Union[str, list[str], list[int]]
_Device = Union[str, torch.device]
_ZeroWordId = tuple[int, int]
_ZeroNodeMap = dict[tuple[int, int], tuple[int, _ZeroWordId]]


def extract_text(
        document: Document,
        sent_id: int,
        start_word: int,
        end_word: int,
    ) -> str:
    sentence = document.sentences[sent_id]
    tokens: list[Union[Token, Word]] = []

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
        if parent_token is None:
            raise ValueError("Cannot extract coreference text from a word without a parent token")
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
    text: list[str] = []
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
    PROVIDES_DEFAULT: ClassVar[set[str]] = {COREF}
    # set of processor requirements for this processor
    REQUIRES_DEFAULT: ClassVar[set[str]] = {TOKENIZE}

    _model: CorefModel
    _use_zeros: bool

    def _set_up_model(
            self,
            config: _CorefProcessorConfig,
            pipeline: Pipeline,
            device: _Device,
        ) -> None:
        try:
            from stanza.models.coref.model import CorefModel
        except ImportError:
            raise ImportError("Please install the transformers and peft libraries before using coref! Try `pip install -e .[transformers]`.")

        # set up model
        # currently, the model has everything packaged in it
        # (except its config)
        # TODO: separate any pretrains if possible
        # TODO: add device parameter to the load mechanism
        config_update = {'log_norms': config.get('log_norms', False),
                         'device': device}
        model = CorefModel.load_model(path=config['model_path'],
                                      ignore={"bert_optimizer", "general_optimizer",
                                              "bert_scheduler", "general_scheduler"},
                                      config_update=config_update,
                                      foundation_cache=pipeline.foundation_cache)
        batch_size = config.get('batch_size')
        if batch_size:
            model_config = model.config
            if not isinstance(model_config, Config):
                raise TypeError("Loaded coreference model has an invalid configuration")
            model_config.a_scoring_batch_size = int(batch_size)
        model.training = False

        self._model = model

        # coref_use_zeros=False will turn off creating new nodes and attaching mentions to those zero nodes
        use_zeros = config.get('use_zeros', True)
        self._use_zeros = use_zeros.lower() != 'false' if isinstance(use_zeros, str) else use_zeros

    def process(self, document: Document) -> Document:
        sentences = document.sentences

        cased_words: list[str] = []
        sent_ids: list[int] = []
        word_pos: list[int] = []
        speaker: list[str] = []
        for sent_idx, sentence in enumerate(sentences):
            for word_idx, word in enumerate(sentence.words):
                cased_words.append(word.text)
                sent_ids.append(sent_idx)
                word_pos.append(word_idx)
                if sentence.speaker:
                    speaker.append(sentence.speaker)
                else:
                    speaker.append("_")

        coref_input: dict[str, _CorefInputValue] = {
            "document_id": "wb_doc_1",
            "cased_words": cased_words,
            "sent_id": sent_ids,
            "speaker": speaker,
        }
        built_coref_input = self._model.build_doc(coref_input)
        results: CorefResult = self._model.run(built_coref_input)

        
        # Handle zero anaphora - zero_scores is always predicted
        zero_nodes_created = self._handle_zero_anaphora(document, results, sent_ids, word_pos)
        
        clusters: list[CorefChain] = []
        for cluster_idx, span_cluster in enumerate(results.span_clusters):
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
            best_span: Optional[int] = None
            max_propn = 0
            for span_idx, span in enumerate(span_cluster):
                word_idx = results.word_clusters[cluster_idx][span_idx]
                is_zero = zero_nodes_created.get((cluster_idx, word_idx))
                if is_zero:
                    continue

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

            mentions: list[CorefMention] = []
            for span_idx, span in enumerate(span_cluster):
                word_idx = results.word_clusters[cluster_idx][span_idx]
                is_zero = zero_nodes_created.get((cluster_idx, word_idx))
                if is_zero:
                    (sent_id, zero_word_id) = is_zero
                    # if the word id is a tuple, it will be attached
                    # to the zero
                    mentions.append(
                        CorefMention(
                            sent_id, 
                             zero_word_id, 
                             zero_word_id
                        )
                    )
                else:
                    sent_id = sent_ids[span[0]]
                    start_word = word_pos[span[0]]
                    end_word = word_pos[span[1]-1] + 1
                    mentions.append(CorefMention(sent_id, start_word, end_word))
                
            # if we ended up with no best span, then our "representative text"
            # is just underscore
            if best_span is not None:
                representative = mentions[best_span]
                if (isinstance(representative.start_word, tuple)
                        or isinstance(representative.end_word, tuple)):
                    raise ValueError("A zero node cannot be a representative coreference mention")
                representative_text = extract_text(document, representative.sentence, representative.start_word, representative.end_word)
            else:
                representative_text = "_"

            chain = CorefChain(len(clusters), mentions, representative_text, best_span)
            clusters.append(chain)

        document.coref = clusters
        return document

    def _handle_zero_anaphora(
            self,
            document: Document,
            results: CorefResult,
            sent_ids: Sequence[int],
            word_pos: Sequence[int],
        ) -> _ZeroNodeMap:
        """Handle zero anaphora by creating zero nodes and updating coreference clusters."""
        if results.zero_scores is None or results.word_clusters is None:
            return {}
        if not self._use_zeros:
            return {}

        zero_scores = results.zero_scores.squeeze(-1) if results.zero_scores.dim() > 1 else results.zero_scores
        
        # Flatten word_clusters to get the word indices that correspond to zero_scores
        cluster_word_ids: list[int] = []
        cluster_mapping: dict[int, int] = {}
        counter = 0
        for indx, cluster in enumerate(results.word_clusters):
            for _ in range(len(cluster)):
                cluster_mapping[counter] = indx
                counter += 1
            cluster_word_ids.extend(cluster)
        
        # Find indices where zero_scores > 0
        zero_indices = (zero_scores > 0.0).nonzero()

        # this dict maps (cluster_id, word_id) to (cluster_id, start, end)
        # which overrides span_clusters
        zero_to_coref: _ZeroNodeMap = {}

        for zero_idx in zero_indices:
            zero_idx = int(zero_idx.item())
            if zero_idx >= len(cluster_word_ids):
                continue
                
            word_idx = cluster_word_ids[zero_idx]
            sent_id = sent_ids[word_idx]
            word_id = word_pos[word_idx]
            
            # Create zero node - attach BEFORE the current word
            # This means the zero node comes after word_id-1 but before word_id
            zero_word_id: _ZeroWordId = (
                word_id, 
                len(document.sentences[sent_id].empty_words)+1
            )  # attach after word_id-1, before word_id
            zero_word_entry: TokenEntry = {
                "text": "_", 
                "lemma": "_", 
                "id": zero_word_id
            }
            zero_word = Word(document.sentences[sent_id], zero_word_entry)
            document.sentences[sent_id].empty_words.append(zero_word)
            
            # Track this zero node for adding to coreference clusters
            cluster_idx = cluster_mapping[zero_idx]
            zero_to_coref[(cluster_idx, word_idx)] = (
                sent_id, zero_word_id
            )

        return zero_to_coref
