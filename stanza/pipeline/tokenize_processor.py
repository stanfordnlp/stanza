"""
Processor for performing tokenization
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
import logging
import os
import re
from typing import Final, Optional, Protocol, TYPE_CHECKING, TypedDict, Union

import torch

from stanza.models.tokenization.data import TokenizationDataset
from stanza.models.tokenization.trainer import Trainer
from stanza.models.tokenization.utils import (
    TokenizerDictionary,
    TokenizerPostprocessor,
    TokenizerTrainer,
    output_predictions,
)
from stanza.models.tokenization.vocab import Vocab
from stanza.pipeline._constants import *
from stanza.pipeline.processor import ProcessorDevice, UDProcessor, register_processor
from stanza.models.common import doc

if TYPE_CHECKING:
    from stanza.pipeline.core import Pipeline

# these imports trigger the "register_variant" decorations
from stanza.pipeline.external.jieba import JiebaTokenizer
from stanza.pipeline.external.spacy import SpacyTokenizer
from stanza.pipeline.external.sudachipy import SudachiPyTokenizer
from stanza.pipeline.external.pythainlp import PyThaiNLPTokenizer

logger = logging.getLogger('stanza')

TOKEN_TOO_LONG_REPLACEMENT: Final[str] = "<UNK>"

_SpeakerSegment = tuple[Optional[str], str, int]
_PretokenizedInput = Union[str, list[str], doc.PretokenizedText]
_TokenizerInput = Union[str, list[str], doc.PretokenizedText, doc.Document]
_TokenizedData = list[list[doc.TokenEntry]]
_OffsetUnit = Union[doc.Token, doc.Word]
_ModelPath = Union[str, os.PathLike[str]]


class _TokenizeSetupConfig(TypedDict, total=False):
    pretokenized: bool
    forward_charlm_path: Optional[str]
    model_path: _ModelPath
    postprocessor: TokenizerPostprocessor
    speaker_delim: str


class _TokenizeRuntimeConfig(_TokenizeSetupConfig, total=False):
    max_seqlen: int
    no_ssplit: bool
    num_workers: int


class _TokenizerTrainer(TokenizerTrainer, Protocol):
    @property
    def dictionary(self) -> Optional[TokenizerDictionary]:
        ...

    @property
    def vocab(self) -> Optional[Vocab]:
        ...


class _TokenizerVariant(Protocol):
    def process(self, doc: _TokenizerInput) -> doc.Document:
        ...

    def bulk_process(
            self,
            docs: list[doc.Document],
        ) -> list[doc.Document]:
        ...


def _require_offsets(unit: _OffsetUnit) -> tuple[int, int]:
    start_char = unit.start_char
    end_char = unit.end_char
    if start_char is None or end_char is None:
        raise ValueError(f"Missing character offsets for {unit!r}")
    return start_char, end_char


def _shift_offsets(unit: _OffsetUnit, offset_delta: int) -> None:
    start_char, end_char = _require_offsets(unit)
    unit._start_char = start_char + offset_delta
    unit._end_char = end_char + offset_delta


def parse_speaker_segments(
        text: str,
        opener: str,
        closer: str,
    ) -> list[_SpeakerSegment]:
    """
    Parse a raw text string containing inline speaker tags into a list of
    (speaker, segment_text, tag_start_in_original) tuples.

    Speaker tags have the form <opener><label><closer>, e.g. "<A>" when
    opener="<" and closer=">".  The label is everything between the two
    delimiter characters; it must be non-empty.

    The opener and closer are treated as literal strings (not regex
    metacharacters), so callers can pass e.g. opener="{{", closer="}}".

    Returns:
        A list of (speaker, segment_text, original_tag_end) tuples where:
          - speaker is the label string, or None for text before the first tag
          - segment_text is the text that follows the tag up to the next tag
            (or end of string)
          - original_tag_end is the character position in `text` where the
            segment content starts (i.e., right after the closing delimiter)
        Empty segments (segment_text.strip() == "") are returned as-is; the
        caller decides whether to warn or error on them.

    Raises:
        ValueError: if opener == closer (ambiguous delimiter)
        ValueError: if a tag is opened but never closed on the same line
          (we do not allow tags to span across segment text)
    """
    if opener == closer:
        raise ValueError(
            f"speaker_delim opener and closer must be different characters; got {opener!r} for both"
        )

    # Build a pattern that matches one complete tag.
    # We do not allow the closer character inside a tag label.
    escaped_opener = re.escape(opener)
    escaped_closer = re.escape(closer)
    tag_pattern = re.compile(
        f"{escaped_opener}([^{escaped_closer}{escaped_opener}]+){escaped_closer}"
    )

    segments: list[_SpeakerSegment] = []
    prev_speaker: Optional[str] = None
    prev_end = 0  # position in `text` where the current segment content started

    for m in tag_pattern.finditer(text):
        label = m.group(1)
        tag_start = m.start()
        tag_end = m.end()

        # Everything from prev_end to tag_start is the content of the
        # previous segment (or the unlabeled prefix).
        segment_text = text[prev_end:tag_start]
        segments.append((prev_speaker, segment_text, prev_end))

        prev_speaker = label
        prev_end = tag_end

    # Trailing segment after the last tag (or the entire string if no tags).
    segments.append((prev_speaker, text[prev_end:], prev_end))

    return segments


def _check_empty_segments(segments: Sequence[_SpeakerSegment]) -> None:
    """
    Warn if any labeled segment (i.e., after a speaker tag) has no
    non-whitespace content.  This catches consecutive tags like <A><B>text,
    where the <A> segment would be empty.

    We warn rather than error because the caller may legitimately want to
    allow trailing whitespace between a tag and the actual speech.
    """
    for i, (speaker, text, _) in enumerate(segments):
        if speaker is not None and not text.strip():
            # Is the *next* segment also labeled (consecutive tags)?
            if i + 1 < len(segments) and segments[i + 1][0] is not None:
                logger.warning(
                    "Speaker tag %r produced an empty segment (consecutive tags?). "
                    "It will be skipped during tokenization.",
                    speaker,
                )
            elif i + 1 == len(segments):
                logger.warning(
                    "Speaker tag %r at the end of input produced an empty segment.",
                    speaker,
                )


def _apply_speaker_to_sentences(
        result_doc: doc.Document,
        segments: Sequence[_SpeakerSegment],
        original_text: str,
    ) -> None:
    """
    Given a Document produced by tokenizing the joined (tag-stripped) text,
    and the original (speaker, segment_text, original_seg_start) segment list,
    this function:

      1. Assigns sentence.speaker for every sentence based on which segment
         it fell into.
      2. Corrects all token/word character offsets from "position in the
         joined stripped text" back to "position in the original input string".

    The correction works because within each segment the relative order of
    characters is identical in the stripped and original strings.  The only
    differences are:
      - Tags have been removed (shifting everything left by their total width).
      - Segments have been joined with "\n\n" instead of whatever whitespace
        surrounded the tags.

    We compute, for each segment i:
        stripped_seg_start[i]  = sum of len(seg_text) + 2 for all j < i
                                  (the +2 is for the "\n\n" separator)
        original_seg_start[i]  = stored in segments[i][2]

    For a token at stripped position p inside segment i:
        local_offset      = p - stripped_seg_start[i]
        original_position = original_seg_start[i] + local_offset
    """
    # Build the stripped-space start position for each segment.
    # We join with "\n\n" (len 2), matching what the caller passed to process().
    # Skip segments that were empty (they contributed nothing to the joined text).
    non_empty_segments = [(spk, seg, orig) for (spk, seg, orig) in segments if seg]
    stripped_seg_starts: list[int] = []
    cursor = 0
    for _, seg_text, _ in non_empty_segments:
        stripped_seg_starts.append(cursor)
        cursor += len(seg_text) + 2  # +2 for "\n\n"

    sentences = result_doc.sentences
    sent_idx = 0
    num_sentences = len(sentences)

    for seg_i, (speaker, seg_text, original_seg_start) in enumerate(non_empty_segments):
        if sent_idx >= num_sentences:
            break

        stripped_seg_start = stripped_seg_starts[seg_i]
        stripped_seg_end = stripped_seg_start + len(seg_text)

        # Determine which sentences belong to this segment.
        # A sentence belongs to segment i if its last token's end_char (in
        # stripped space) is <= stripped_seg_end.  This mirrors the logic
        # in bulk_process.
        seg_sent_start = sent_idx
        while sent_idx < num_sentences:
            sentence_tokens = sentences[sent_idx].tokens
            if not sentence_tokens:
                raise ValueError("Speaker-tagged sentences cannot be empty")
            _, sentence_end = _require_offsets(sentence_tokens[-1])
            if sentence_end > stripped_seg_end:
                break
            sent_idx += 1
        seg_sent_end = sent_idx

        offset_delta = original_seg_start - stripped_seg_start

        for sent in sentences[seg_sent_start:seg_sent_end]:
            sent.speaker = speaker

            for token in sent.tokens:
                _shift_offsets(token, offset_delta)
                if token.words:
                    for word in token.words:
                        _shift_offsets(word, offset_delta)

    # Fix up spaces_after/spaces_before using the original text, now that
    # offsets are corrected.  We do the same walk that Document.mark_whitespace
    # does, but directly rather than rebuilding the whole Document.
    all_sentences = result_doc.sentences
    for sentence in all_sentences:
        for prev_token, next_token in zip(sentence.tokens[:-1], sentence.tokens[1:]):
            if prev_token.end_char is not None and next_token.start_char is not None:
                prev_token.spaces_after = original_text[prev_token.end_char:next_token.start_char]
    for prev_sentence, next_sentence in zip(all_sentences[:-1], all_sentences[1:]):
        prev_token = prev_sentence.tokens[-1]
        next_token = next_sentence.tokens[0]
        if prev_token.end_char is not None and next_token.start_char is not None:
            prev_token.spaces_after = original_text[prev_token.end_char:next_token.start_char]
    if all_sentences:
        last_token = all_sentences[-1].tokens[-1]
        if last_token.end_char is not None:
            last_token.spaces_after = original_text[last_token.end_char:]
        first_token = all_sentences[0].tokens[0]
        if first_token.start_char is not None:
            first_token.spaces_before = original_text[:first_token.start_char]

    result_doc._text = original_text


# class for running the tokenizer
@register_processor(name=TOKENIZE)
class TokenizeProcessor(UDProcessor):
    _config: _TokenizeRuntimeConfig
    _trainer: Optional[_TokenizerTrainer]
    _postprocessor: Optional[TokenizerPostprocessor]
    _speaker_opener: Optional[str]
    _speaker_closer: Optional[str]

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([TOKENIZE])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([])
    # default max sequence length
    MAX_SEQ_LENGTH_DEFAULT = 1000

    def _set_up_model(
            self,
            config: _TokenizeSetupConfig,
            pipeline: Optional[Pipeline],
            device: Optional[ProcessorDevice],
        ) -> None:
        # set up trainer
        if config.get('pretokenized'):
            self._trainer = None
        else:
            model_path = config.get('model_path')
            if not isinstance(model_path, (str, os.PathLike)):
                raise ValueError(
                    "A neural tokenizer requires a string or path-like model_path"
                )
            normalized_model_path = os.fspath(model_path)
            if not isinstance(normalized_model_path, str):
                raise ValueError(
                    "A neural tokenizer requires a text model_path"
                )
            args = {'charlm_forward_file': config.get('forward_charlm_path', None)}
            foundation_cache = None if pipeline is None else pipeline.foundation_cache
            self._trainer = Trainer(
                args=args,
                model_file=normalized_model_path,
                device=device,
                foundation_cache=foundation_cache,
            )

        # get and typecheck the postprocessor
        postprocessor = config.get('postprocessor')
        if postprocessor and callable(postprocessor):
            self._postprocessor = postprocessor
        elif not postprocessor:
            self._postprocessor = None
        else:
            raise ValueError("Tokenizer received 'postprocessor' option of unrecognized type; postprocessor must be callable. Got %s" % postprocessor)

        # parse the speaker_delim option into (opener, closer) if provided
        speaker_delim = config.get('speaker_delim', None)
        if speaker_delim is not None:
            if not isinstance(speaker_delim, str):
                raise ValueError(
                    "speaker_delim must be a string containing exactly two characters. "
                    f"Got {speaker_delim!r}"
                )
            if len(speaker_delim) != 2:
                raise ValueError(
                    "speaker_delim must be exactly two characters (opener and closer), "
                    f"e.g. '<>' or '{{}}'. Got {speaker_delim!r}"
                )
            self._speaker_opener = speaker_delim[0]
            self._speaker_closer = speaker_delim[1]
        else:
            self._speaker_opener = None
            self._speaker_closer = None

    def _runtime_config(self) -> _TokenizeRuntimeConfig:
        return self._config

    def _neural_state(self) -> tuple[_TokenizerTrainer, Vocab]:
        trainer = self._trainer
        vocab = self._vocab
        if trainer is None:
            raise RuntimeError("The neural tokenizer model has not been loaded")
        if not isinstance(vocab, Vocab):
            raise RuntimeError("The neural tokenizer vocabulary has not been loaded")
        return trainer, vocab

    def _tokenizer_variant(self) -> _TokenizerVariant:
        variant: _TokenizerVariant = self._variant
        return variant

    def _speaker_delimiters(self) -> tuple[str, str]:
        opener = self._speaker_opener
        closer = self._speaker_closer
        if opener is None or closer is None:
            raise RuntimeError("Speaker delimiters have not been configured")
        return opener, closer

    def _max_sequence_length(self) -> int:
        max_seq_len = self._runtime_config().get(
            'max_seqlen',
            TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT,
        )
        if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool):
            raise ValueError(
                "max_seqlen must be an integer. "
                f"Got {max_seq_len!r}"
            )
        if max_seq_len <= 0:
            raise ValueError(
                "max_seqlen must be greater than zero. "
                f"Got {max_seq_len}"
            )
        return max_seq_len

    def _num_workers(self) -> int:
        num_workers = self._runtime_config().get('num_workers', 0)
        if not isinstance(num_workers, int) or isinstance(num_workers, bool):
            raise ValueError(
                "num_workers must be an integer. "
                f"Got {num_workers!r}"
            )
        if num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative. "
                f"Got {num_workers}"
            )
        return num_workers

    def _tokenize_neural(self, raw_text: str) -> _TokenizedData:
        trainer, vocab = self._neural_state()
        max_seq_len = self._max_sequence_length()
        config = self._runtime_config()

        batches = TokenizationDataset(
            config,
            input_text=raw_text,
            vocab=vocab,
            evaluation=True,
            dictionary=trainer.dictionary,
        )
        token_data: _TokenizedData
        with torch.no_grad():
            _, _, _, token_data = output_predictions(
                None,
                trainer,
                batches,
                vocab,
                None,
                max_seq_len,
                orig_text=raw_text,
                no_ssplit=bool(config.get('no_ssplit', False)),
                num_workers=self._num_workers(),
                postprocessor=self._postprocessor,
            )

        # replace excessively long tokens with <UNK> to avoid downstream GPU
        # memory issues in POS
        for sentence in token_data:
            for token in sentence:
                if len(token[doc.TEXT]) > max_seq_len:
                    token[doc.TEXT] = TOKEN_TOO_LONG_REPLACEMENT

        return token_data

    def process_pre_tokenized_text(
            self,
            input_src: _PretokenizedInput,
        ) -> tuple[str, _TokenizedData]:
        """
        Pretokenized text can be provided in 2 manners:

        1.) str, tokenized by whitespace, sentence split by newline
        2.) a flat token list representing one sentence, or a list of token
            lists in which each inner list represents a sentence

        generate dictionary data structure
        """

        sentences: list[list[str]]
        if isinstance(input_src, str):
            sentences = [sent.strip().split() for sent in input_src.strip().split('\n') if len(sent.strip()) > 0]
        elif isinstance(input_src, list):
            if not input_src:
                sentences = []
            elif all(isinstance(token, str) for token in input_src):
                sentence: list[str] = []
                for token in input_src:
                    if not isinstance(token, str):
                        raise ValueError("Pretokenized input cannot mix tokens and sentences")
                    sentence.append(token)
                sentences = [sentence]
            else:
                sentences = []
                for sentence_input in input_src:
                    if isinstance(sentence_input, str):
                        raise ValueError("Pretokenized input cannot mix tokens and sentences")
                    if not isinstance(sentence_input, list):
                        raise ValueError(
                            "Every pretokenized sentence must be a list of strings"
                        )
                    if not sentence_input:
                        raise ValueError("Pretokenized input cannot contain an empty sentence")
                    sentence = []
                    for token in sentence_input:
                        if not isinstance(token, str):
                            raise ValueError("Every pretokenized token must be a string")
                        sentence.append(token)
                    sentences.append(sentence)
        else:
            raise TypeError(
                "Pretokenized input must be a string, a token list, "
                "or a list of token lists"
            )

        document: _TokenizedData = []
        idx = 0
        for sentence in sentences:
            sent: list[doc.TokenEntry] = []
            for token_id, token in enumerate(sentence):
                entry: doc.TokenEntry = {
                    doc.ID: (token_id + 1,),
                    doc.TEXT: token,
                    doc.MISC: f'start_char={idx}|end_char={idx + len(token)}',
                }
                sent.append(entry)
                idx += len(token) + 1
            document.append(sent)
        raw_text = ' '.join([' '.join(sentence) for sentence in sentences])
        return raw_text, document

    def process(self, document: _TokenizerInput) -> doc.Document:
        config = self._runtime_config()
        if not (isinstance(document, str) or isinstance(document, doc.Document) or (config.get('pretokenized') or config.get('no_ssplit', False))):
            raise ValueError("If neither 'pretokenized' or 'no_ssplit' option is enabled, the input to the TokenizerProcessor must be a string or a Document instance.  Got %s" % str(type(document)))

        input_src: _PretokenizedInput
        if isinstance(document, doc.Document):
            if config.get('pretokenized'):
                return document
            document_text = document.text
            if not isinstance(document_text, str):
                raise TypeError(
                    "A Document passed to the neural tokenizer must contain string text"
                )
            input_src = document_text
        else:
            input_src = document

        if config.get('pretokenized'):
            raw_text, token_data = self.process_pre_tokenized_text(input_src)
            return doc.Document(token_data, raw_text)

        if hasattr(self, '_variant'):
            # Preserve the extension contract: variants receive the input in
            # the same shape supplied by the caller (except Document wrappers,
            # which historically expose their text).
            return self._tokenizer_variant().process(input_src)

        if isinstance(input_src, str):
            raw_text = input_src
        else:
            paragraphs: list[str] = []
            for paragraph in input_src:
                if not isinstance(paragraph, str):
                    raise ValueError(
                        "Without pretokenized=True, tokenizer list input must "
                        "contain only strings"
                    )
                paragraphs.append(paragraph)
            raw_text = '\n\n'.join(paragraphs)

        # Handle speaker-tagged input before the normal tokenization path so
        # speaker boundaries force sentence splits and sentences get speakers.
        if self._speaker_opener is not None:
            return self._process_speaker_tagged_text(raw_text)

        token_data = self._tokenize_neural(raw_text)
        return doc.Document(token_data, raw_text)

    def _process_speaker_tagged_text(self, original_text: str) -> doc.Document:
        """
        Tokenize a string that contains inline speaker tags such as "<A>text <B>more text".

        Steps:
          1. Parse the text into (speaker, segment, original_start) triples.
          2. Warn on empty labeled segments (consecutive tags).
          3. Join the non-empty segment texts with "\n\n", which forces sentence
             split boundaries at every speaker change.
          4. Run the normal neural tokenizer on that joined text.
          5. Walk the resulting sentences in segment order, assigning .speaker
             and correcting character offsets back into the original string.
        """
        opener, closer = self._speaker_delimiters()
        segments = parse_speaker_segments(original_text, opener, closer)
        _check_empty_segments(segments)

        non_empty_segments = [(spk, seg, orig) for (spk, seg, orig) in segments if seg]
        if not non_empty_segments:
            # Edge case: the entire input was tags with no content.
            return doc.Document([], original_text)

        joined_text = '\n\n'.join(seg for _, seg, _ in non_empty_segments)

        token_data = self._tokenize_neural(joined_text)

        # Build the Document against the joined (stripped) text first, so that
        # the internal offset machinery in Document.__init__ / mark_whitespace
        # runs correctly.  We then patch offsets and .text in _apply_speaker.
        result_doc = doc.Document(token_data, joined_text)

        _apply_speaker_to_sentences(result_doc, non_empty_segments, original_text)

        return result_doc

    def bulk_process(self, docs: list[doc.Document]) -> list[doc.Document]:
        """
        The tokenizer cannot use UDProcessor's sentence-level cross-document batching interface, and requires special handling.
        Essentially, this method concatenates the text of multiple documents with "\n\n", tokenizes it with the neural tokenizer,
        then splits the result into the original Documents and recovers the original character offsets.
        """
        if not docs:
            return []

        if hasattr(self, '_variant'):
            return self._tokenizer_variant().bulk_process(docs)

        if self._runtime_config().get('pretokenized'):
            res: list[doc.Document] = []
            for document in docs:
                if len(document.sentences) > 0:
                    # perhaps this is a document already tokenized,
                    # being sent back in for more analysis / reparsing / etc?
                    # in that case, no need to try to tokenize it
                    # based on whitespace tokenizing the document text
                    # which, interestingly, may not even exist depending on
                    # how the document was created)
                    # by making a whole deepcopy, the original Document is unchanged
                    res.append(copy.deepcopy(document))
                else:
                    input_text = document.text
                    if input_text is None:
                        raise TypeError(
                            "A pretokenized Document without sentences must contain text"
                        )
                    raw_text, token_data = self.process_pre_tokenized_text(input_text)
                    res.append(doc.Document(token_data, raw_text))
            return res

        # If speaker tagging is active we cannot use the single-pass \n\n join
        # across documents, because each document may itself contain speaker tags
        # that need to be parsed independently.  Fall back to per-document processing.
        # This is slightly less efficient than the normal bulk path but is correct
        # and avoids any cross-document speaker bleed.
        if self._speaker_opener is not None:
            return [self.process(document) for document in docs]

        texts: list[str] = []
        for thisdoc in docs:
            text = thisdoc.text
            if not isinstance(text, str):
                raise TypeError(
                    "Every Document passed to the tokenizer must contain string text"
                )
            texts.append(text)

        combined_text = '\n\n'.join(texts)
        processed_combined = self.process(doc.Document([], text=combined_text))

        # postprocess sentences and tokens to reset back pointers and char offsets
        charoffset = 0
        sentst = senten = 0
        for thisdoc, text in zip(docs, texts):
            while senten < len(processed_combined.sentences):
                sentence_tokens = processed_combined.sentences[senten].tokens
                if not sentence_tokens:
                    raise ValueError("Tokenized sentences cannot be empty")
                _, sentence_end = _require_offsets(sentence_tokens[-1])
                if sentence_end - charoffset > len(text):
                    break
                senten += 1

            sentences = processed_combined.sentences[sentst:senten]
            thisdoc.sentences = sentences
            for sent in sentences:
                # fix doc back pointers for sentences
                sent._doc = thisdoc

                # fix char offsets for tokens and words
                for token in sent.tokens:
                    _shift_offsets(token, -charoffset)
                    if token.words:  # not-yet-processed MWT can leave empty tokens
                        for word in token.words:
                            _shift_offsets(word, -charoffset)

            # Here we need to fix up the SpacesAfter for the very last token
            # and the SpacesBefore for the first token of the next doc
            # After all, we had connected the text with \n\n
            # Need to be careful about this - in a case such as
            #   " -text one- "
            #   " -text two- "
            # We want the SpacesBefore for the second document to reflect
            # the extra space at the start of its text
            # and the SpacesAfter for the first document to reflect
            # the whitespace after its text
            if len(sentences) > 0:
                last_token = sentences[-1].tokens[-1]
                _, last_end = _require_offsets(last_token)
                last_whitespace = text[last_end:]
                last_token.spaces_after = last_whitespace

                first_token = sentences[0].tokens[0]
                first_start, _ = _require_offsets(first_token)
                first_whitespace = text[:first_start]
                first_token.spaces_before = first_whitespace

            thisdoc.num_tokens = sum(len(sent.tokens) for sent in sentences)
            thisdoc.num_words = sum(len(sent.words) for sent in sentences)
            sentst = senten

            charoffset += len(text) + 2

        return docs
