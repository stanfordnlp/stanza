"""
Processor for performing tokenization
"""

import copy
import io
import logging
import re

import torch

from stanza.models.tokenization.data import TokenizationDataset
from stanza.models.tokenization.trainer import Trainer
from stanza.models.tokenization.utils import output_predictions
from stanza.pipeline._constants import *
from stanza.pipeline.processor import UDProcessor, register_processor
from stanza.pipeline.registry import PROCESSOR_VARIANTS
from stanza.models.common import doc

# these imports trigger the "register_variant" decorations
from stanza.pipeline.external.jieba import JiebaTokenizer
from stanza.pipeline.external.spacy import SpacyTokenizer
from stanza.pipeline.external.sudachipy import SudachiPyTokenizer
from stanza.pipeline.external.pythainlp import PyThaiNLPTokenizer

logger = logging.getLogger('stanza')

TOKEN_TOO_LONG_REPLACEMENT = "<UNK>"


def parse_speaker_segments(text, opener, closer):
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

    segments = []
    prev_speaker = None
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


def _check_empty_segments(segments):
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


def _apply_speaker_to_sentences(result_doc, segments, original_text):
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
    stripped_seg_starts = []
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
        while (sent_idx < num_sentences and
               sentences[sent_idx].tokens[-1].end_char <= stripped_seg_end):
            sent_idx += 1
        seg_sent_end = sent_idx

        offset_delta = original_seg_start - stripped_seg_start

        for sent in sentences[seg_sent_start:seg_sent_end]:
            sent.speaker = speaker

            for token in sent.tokens:
                token._start_char += offset_delta
                token._end_char += offset_delta
                if token.words:
                    for word in token.words:
                        word._start_char += offset_delta
                        word._end_char += offset_delta

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

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([TOKENIZE])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([])
    # default max sequence length
    MAX_SEQ_LENGTH_DEFAULT = 1000

    def _set_up_model(self, config, pipeline, device):
        # set up trainer
        if config.get('pretokenized'):
            self._trainer = None
        else:
            args = {'charlm_forward_file': config.get('forward_charlm_path', None)}
            self._trainer = Trainer(args=args, model_file=config['model_path'], device=device, foundation_cache=pipeline.foundation_cache)

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

    def process_pre_tokenized_text(self, input_src):
        """
        Pretokenized text can be provided in 2 manners:

        1.) str, tokenized by whitespace, sentence split by newline
        2.) list of token lists, each token list represents a sentence

        generate dictionary data structure
        """

        document = []
        if isinstance(input_src, str):
            sentences = [sent.strip().split() for sent in input_src.strip().split('\n') if len(sent.strip()) > 0]
        elif isinstance(input_src, list):
            sentences = input_src
        idx = 0
        for sentence in sentences:
            sent = []
            for token_id, token in enumerate(sentence):
                sent.append({doc.ID: (token_id + 1, ), doc.TEXT: token, doc.MISC: f'start_char={idx}|end_char={idx + len(token)}'})
                idx += len(token) + 1
            document.append(sent)
        raw_text = ' '.join([' '.join(sentence) for sentence in sentences])
        return raw_text, document

    def process(self, document):
        if not (isinstance(document, str) or isinstance(document, doc.Document) or (self.config.get('pretokenized') or self.config.get('no_ssplit', False))):
            raise ValueError("If neither 'pretokenized' or 'no_ssplit' option is enabled, the input to the TokenizerProcessor must be a string or a Document object.  Got %s" % str(type(document)))

        if isinstance(document, doc.Document):
            if self.config.get('pretokenized'):
                return document
            document = document.text

        if self.config.get('pretokenized'):
            raw_text, document = self.process_pre_tokenized_text(document)
            return doc.Document(document, raw_text)

        if hasattr(self, '_variant'):
            return self._variant.process(document)

        # Handle speaker-tagged input.  We do this before the normal tokenization
        # path so that the speaker boundaries force sentence splits and the
        # resulting sentences get .speaker annotations.
        if self._speaker_opener is not None and isinstance(document, str):
            return self._process_speaker_tagged_text(document)

        raw_text = '\n\n'.join(document) if isinstance(document, list) else document

        max_seq_len = self.config.get('max_seqlen', TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT)

        # set up batches
        batches = TokenizationDataset(self.config, input_text=raw_text, vocab=self.vocab, evaluation=True, dictionary=self.trainer.dictionary)
        # get dict data
        with torch.no_grad():
            _, _, _, document = output_predictions(None, self.trainer, batches, self.vocab, None,
                                                   max_seq_len,
                                                   orig_text=raw_text,
                                                   no_ssplit=self.config.get('no_ssplit', False),
                                                   num_workers = self.config.get('num_workers', 0),
                                                   postprocessor = self._postprocessor)

        # replace excessively long tokens with <UNK> to avoid downstream GPU memory issues in POS
        for sentence in document:
            for token in sentence:
                if len(token['text']) > max_seq_len:
                    token['text'] = TOKEN_TOO_LONG_REPLACEMENT

        return doc.Document(document, raw_text)

    def _process_speaker_tagged_text(self, original_text):
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
        segments = parse_speaker_segments(
            original_text, self._speaker_opener, self._speaker_closer
        )
        _check_empty_segments(segments)

        non_empty_segments = [(spk, seg, orig) for (spk, seg, orig) in segments if seg]
        if not non_empty_segments:
            # Edge case: the entire input was tags with no content.
            return doc.Document([], original_text)

        joined_text = '\n\n'.join(seg for _, seg, _ in non_empty_segments)

        max_seq_len = self.config.get('max_seqlen', TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT)

        batches = TokenizationDataset(self.config, input_text=joined_text, vocab=self.vocab, evaluation=True, dictionary=self.trainer.dictionary)
        with torch.no_grad():
            _, _, _, token_data = output_predictions(
                None, self.trainer, batches, self.vocab, None,
                max_seq_len,
                orig_text=joined_text,
                no_ssplit=self.config.get('no_ssplit', False),
                num_workers=self.config.get('num_workers', 0),
                postprocessor=self._postprocessor,
            )

        for sentence in token_data:
            for token in sentence:
                if len(token['text']) > max_seq_len:
                    token['text'] = TOKEN_TOO_LONG_REPLACEMENT

        # Build the Document against the joined (stripped) text first, so that
        # the internal offset machinery in Document.__init__ / mark_whitespace
        # runs correctly.  We then patch offsets and .text in _apply_speaker.
        result_doc = doc.Document(token_data, joined_text)

        _apply_speaker_to_sentences(result_doc, non_empty_segments, original_text)

        return result_doc

    def bulk_process(self, docs):
        """
        The tokenizer cannot use UDProcessor's sentence-level cross-document batching interface, and requires special handling.
        Essentially, this method concatenates the text of multiple documents with "\n\n", tokenizes it with the neural tokenizer,
        then splits the result into the original Documents and recovers the original character offsets.
        """
        if hasattr(self, '_variant'):
            return self._variant.bulk_process(docs)

        if self.config.get('pretokenized'):
            res = []
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
                    raw_text, document = self.process_pre_tokenized_text(document.text)
                    res.append(doc.Document(document, raw_text))
            return res

        # If speaker tagging is active we cannot use the single-pass \n\n join
        # across documents, because each document may itself contain speaker tags
        # that need to be parsed independently.  Fall back to per-document processing.
        # This is slightly less efficient than the normal bulk path but is correct
        # and avoids any cross-document speaker bleed.
        if self._speaker_opener is not None:
            return [self.process(document) for document in docs]

        combined_text = '\n\n'.join([thisdoc.text for thisdoc in docs])
        processed_combined = self.process(doc.Document([], text=combined_text))

        # postprocess sentences and tokens to reset back pointers and char offsets
        charoffset = 0
        sentst = senten = 0
        for thisdoc in docs:
            while senten < len(processed_combined.sentences) and processed_combined.sentences[senten].tokens[-1].end_char - charoffset <= len(thisdoc.text):
                senten += 1

            sentences = processed_combined.sentences[sentst:senten]
            thisdoc.sentences = sentences
            for sent in sentences:
                # fix doc back pointers for sentences
                sent._doc = thisdoc

                # fix char offsets for tokens and words
                for token in sent.tokens:
                    token._start_char -= charoffset
                    token._end_char -= charoffset
                    if token.words:  # not-yet-processed MWT can leave empty tokens
                        for word in token.words:
                            word._start_char -= charoffset
                            word._end_char -= charoffset

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
                last_whitespace = thisdoc.text[last_token.end_char:]
                last_token.spaces_after = last_whitespace

                first_token = sentences[0].tokens[0]
                first_whitespace = thisdoc.text[:first_token.start_char]
                first_token.spaces_before = first_whitespace

            thisdoc.num_tokens = sum(len(sent.tokens) for sent in sentences)
            thisdoc.num_words = sum(len(sent.words) for sent in sentences)
            sentst = senten

            charoffset += len(thisdoc.text) + 2

        return docs
