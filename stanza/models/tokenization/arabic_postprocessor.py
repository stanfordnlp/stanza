"""
Arabic-specific sentence boundary postprocessor for stanza tokenizer.

This module provides postprocessing to fix Arabic sentence segmentation issues
where the neural tokenizer may fail to split on Arabic punctuation marks.

Arabic sentence-ending punctuation:
- . (ASCII period - most common in Arabic text)
- ۔ (Arabic full stop, U+06D4)
- ؟ (Arabic question mark, U+061F)
- ； (Arabic semicolon, U+060D)
- '،' (Arabic comma, U+060C - can indicate sentence boundary when followed by space+capital)

Usage:
    import stanza
    from stanza.models.tokenization.arabic_postprocessor import arabic_sentence_postprocessor

    nlp = stanza.Pipeline(lang='ar', tokenize_postprocessor=arabic_sentence_postprocessor)

Or via processor variant:
    nlp = stanza.Pipeline(lang='ar', tokenize='arabic_postprocess')

Issue: https://github.com/stanfordnlp/stanza/issues/1393
"""

import re

from stanza.models import doc as doc_module
from stanza.pipeline._constants import TOKENIZE
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant


def arabic_sentence_postprocessor(sentences):
    """
    Postprocessor to fix Arabic sentence segmentation.
    
    The neural tokenizer may fail to split Arabic sentences at punctuation
    marks. This function adds Arabic-specific sentence boundary detection
    by examining tokens for Arabic sentence-ending punctuation.
    
    Parameters
    ----------
    sentences : List[List[tuple]]
        List of sentences, each sentence is a list of (word, mwt_bool) tuples.
        This is the output format expected by stanza's postprocessor interface.
        
    Returns
    -------
    List[List[tuple]]
        Sentences with additional splits based on Arabic punctuation.
        
    Note
    ----
    This postprocessor detects Arabic script in the text and applies
    language-specific splitting rules. For non-Arabic text, it returns
    the sentences unchanged.
    """
    if not sentences:
        return sentences
    
    # Arabic sentence-ending punctuation patterns
    # Includes: ASCII period (most common in Arabic), Arabic full stop, question mark, semicolon, comma
    ARABIC_END_PUNCT = set(['.', '۔', '؟', '؛', '،'])
    
    # Check if text contains Arabic script
    has_arabic = False
    for sent in sentences:
        for token in sent:
            token_text = token[0] if isinstance(token, tuple) else str(token)
            if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', token_text):
                has_arabic = True
                break
        if has_arabic:
            break
    
    if not has_arabic:
        return sentences
    
    new_sentences = []
    
    for sent in sentences:
        if not sent:
            new_sentences.append(sent)
            continue
        
        # Split at ALL punctuation marks in the sentence (not just the first one)
        current = []
        for token in sent:
            current.append(token)
            token_text = token[0] if isinstance(token, tuple) else str(token)
            
            # Check if token ends with any sentence-ending punctuation
            if token_text and token_text[-1] in ARABIC_END_PUNCT:
                # Only split if this is not the last token
                # Use more robust check - check remaining tokens
                remaining_tokens = len(sent) - len(current)
                if remaining_tokens > 0:
                    new_sentences.append(current)
                    current = []
        
        # Append any remaining tokens
        if current:
            new_sentences.append(current)
    
    return new_sentences


def rebuild_document_with_sentences(doc, new_sentence_tokens):
    """
    Rebuild a Document object with new sentence boundaries.
    
    Parameters
    ----------
    doc : Document
        Original Document object
    new_sentence_tokens : List[List[tuple]]
        New token lists for each sentence
        
    Returns
    -------
    Document
        New Document with updated sentence boundaries
    """
    from stanza.models.common.doc import Document
    
    raw_text = doc.text if hasattr(doc, 'text') else ''
    
    # Build new sentence structure
    new_sentences = []
    char_offset = 0
    
    for sent_tokens in new_sentence_tokens:
        sent_words = []
        token_list = []
        
        for token_tuple in sent_tokens:
            if isinstance(token_tuple, tuple):
                token_text = token_tuple[0]
            else:
                token_text = str(token_tuple)
            
            # Find position in raw text
            try:
                pos = raw_text.index(token_text, char_offset)
            except ValueError:
                pos = char_offset
            
            # Create token
            token_entry = {
                doc_module.ID: (len(token_list) + 1,),
                doc_module.TEXT: token_text,
                doc_module.MISC: f'start_char={pos}|end_char={pos + len(token_text)}'
            }
            token_list.append(token_entry)
            
            # Create word (same as token for now)
            word_entry = {
                doc_module.ID: len(sent_words) + 1,
                doc_module.TEXT: token_text,
                doc_module.MISC: f'start_char={pos}|end_char={pos + len(token_text)}'
            }
            sent_words.append(word_entry)
            char_offset = pos + len(token_text) + 1
        
        # Create sentence
        sent_obj = doc_module.Sentence(
            tokens=token_list,
            words=sent_words,
            text=' '.join(t[0] if isinstance(t, tuple) else str(t) for t in sent_tokens)
        )
        new_sentences.append(sent_obj)
    
    # Create new document
    new_doc = Document(new_sentences, raw_text)
    return new_doc


@register_processor_variant(TOKENIZE, 'arabic_postprocess')
class ArabicPostprocessVariant(ProcessorVariant):
    """
    Arabic tokenizer variant with enhanced sentence boundary detection.
    
    This variant wraps the default tokenizer with additional Arabic-specific
    sentence boundary detection to fix issues where the neural model fails
    to split on Arabic punctuation marks.
    
    Example
    -------
    >>> import stanza
    >>> nlp = stanza.Pipeline(lang='ar', tokenize='arabic_postprocess')
    >>> doc = nlp("نص عربي. نص آخر. سؤال؟ إجابة")
    >>> len(doc.sentences)  # Should be 3+ sentences instead of 1
    """
    
    def __init__(self, config):
        from stanza.pipeline.tokenize_processor import TokenizeProcessor
        
        # Create the base tokenizer
        self._processor = TokenizeProcessor(config)
        
    def process(self, document):
        """Process document with Arabic sentence boundary enhancement."""
        # First run normal tokenization
        doc = self._processor.process(document)
        
        # Now apply Arabic postprocessing to fix sentence boundaries
        if hasattr(doc, 'sentences') and len(doc.sentences) > 0:
            # Extract tokens from each sentence in (word, mwt_bool) format
            sentence_tokens = []
            for sent in doc.sentences:
                tokens = [(token.text, False) for token in sent.tokens]
                sentence_tokens.append(tokens)
            
            # Apply postprocessing to get new sentence splits
            new_sentence_tokens = arabic_sentence_postprocessor(sentence_tokens)
            
            # Only rebuild if we actually have more sentences now
            if len(new_sentence_tokens) > len(sentence_tokens):
                try:
                    doc = rebuild_document_with_sentences(doc, new_sentence_tokens)
                except Exception:
                    # If reconstruction fails, return original document
                    pass
        
        return doc


def create_arabic_postprocessor():
    """
    Factory function to create the Arabic postprocessor.
    
    Returns
    -------
    callable
        A postprocessor function compatible with stanza's
        tokenize_postprocessor configuration option.
        
    Example
    -------
    >>> import stanza
    >>> from stanza.models.tokenization.arabic_postprocessor import create_arabic_postprocessor
    >>> nlp = stanza.Pipeline(lang='ar', tokenize_postprocessor=create_arabic_postprocessor())
    """
    return arabic_sentence_postprocessor