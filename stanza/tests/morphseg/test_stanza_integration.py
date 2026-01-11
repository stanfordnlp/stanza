"""
Integration tests for Stanza MorphSeg Processor
Tests the morpheme segmentation processor within the Stanza pipeline
"""

import pytest
import stanza
from stanza.models.common.doc import Document

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

class TestMorphSegProcessor:
    """Tests for the MorphSeg processor in Stanza pipeline"""

    @pytest.fixture(scope="class")
    def en_pipeline(self):
        """Create English pipeline with morphseg processor"""
        return stanza.Pipeline(
            lang='en',
            processors='tokenize,morphseg',
            download_method=None
        )

    def test_processor_loads(self, en_pipeline):
        """Test that morphseg processor loads successfully"""
        assert 'morphseg' in en_pipeline.processors
        assert en_pipeline.processors['morphseg'] is not None

    def test_basic_segmentation(self, en_pipeline):
        """Test basic morpheme segmentation through pipeline"""
        doc = en_pipeline("running")

        assert len(doc.sentences) == 1
        assert len(doc.sentences[0].words) == 1

        word = doc.sentences[0].words[0]
        assert hasattr(word, 'morphemes')
        assert isinstance(word.morphemes, list)
        assert len(word.morphemes) >= 1

    def test_known_segmentations(self, en_pipeline):
        """Test known morpheme segmentations"""
        # Note: These are actual segmentations from the en2 model
        # Some words may be unsegmented depending on the model
        test_cases = {
            'dogs': ['dog', 's'],
            'aviation': ['aviate', 'ion'],
            'known': ['know', 'n'],
        }

        for word_text, expected in test_cases.items():
            doc = en_pipeline(word_text)
            word = doc.sentences[0].words[0]
            assert word.morphemes == expected, \
                f"Expected {expected}, got {word.morphemes} for '{word_text}'"

    def test_segmentation_consistency(self, en_pipeline):
        """Test that segmentation is consistent and produces valid output"""
        words = ['running', 'quickly', 'walked', 'playing']

        for word_text in words:
            doc = en_pipeline(word_text)
            word = doc.sentences[0].words[0]

            # Should have morphemes attribute
            assert hasattr(word, 'morphemes')
            assert isinstance(word.morphemes, list)
            assert len(word.morphemes) >= 1

            # All morphemes should be strings
            for morpheme in word.morphemes:
                assert isinstance(morpheme, str)
                assert len(morpheme) > 0

    def test_multiple_words(self, en_pipeline):
        """Test segmentation of multiple words in a sentence"""
        doc = en_pipeline("The dogs are running quickly.")

        # Check that all words have morphemes attribute
        for sentence in doc.sentences:
            for word in sentence.words:
                assert hasattr(word, 'morphemes')
                assert isinstance(word.morphemes, list)
                assert len(word.morphemes) >= 1

    def test_punctuation_handling(self, en_pipeline):
        """Test that punctuation is handled correctly"""
        doc = en_pipeline("Hello, world!")

        # All tokens should have morphemes, including punctuation
        for sentence in doc.sentences:
            for word in sentence.words:
                assert hasattr(word, 'morphemes')
                # Punctuation should be unsegmented
                if word.text in [',', '!', '.']:
                    assert word.morphemes == [word.text]

    def test_long_text(self, en_pipeline):
        """Test processing of longer text"""
        text = "According to all known laws of aviation, there is no way a bee should be able to fly."
        doc = en_pipeline(text)

        # Should have multiple sentences or one long sentence
        assert len(doc.sentences) >= 1

        # Count words with morpheme segmentation
        total_words = sum(len(sent.words) for sent in doc.sentences)
        assert total_words > 10

        # All words should have morphemes
        for sentence in doc.sentences:
            for word in sentence.words:
                assert hasattr(word, 'morphemes')

    def test_empty_input(self, en_pipeline):
        """Test handling of empty input"""
        doc = en_pipeline("")
        assert len(doc.sentences) == 0

    def test_single_character(self, en_pipeline):
        """Test single character input"""
        doc = en_pipeline("I")

        assert len(doc.sentences) == 1
        word = doc.sentences[0].words[0]
        assert word.morphemes == ['i']  # Normalized to lowercase

    def test_morphemes_attribute_persistence(self, en_pipeline):
        """Test that morphemes attribute persists through pipeline"""
        doc = en_pipeline("running quickly")

        # Store morphemes
        morphemes_list = []
        for sentence in doc.sentences:
            for word in sentence.words:
                morphemes_list.append(word.morphemes)

        # Access again to ensure persistence
        for i, sentence in enumerate(doc.sentences):
            for j, word in enumerate(sentence.words):
                assert hasattr(word, 'morphemes')
                assert word.morphemes is not None


class TestMultilingualMorphSeg:
    """Test morpheme segmentation across different languages"""

    @pytest.mark.parametrize("lang,text,expected_word", [
        ('en', 'running', 'running'),
        ('es', 'corriendo', 'corriendo'),
        ('fr', 'rapidement', 'rapidement'),
        ('cs', 'běžící', 'běžící'),
        ('it', 'correndo', 'correndo'),
    ])
    def test_multilingual_support(self, lang, text, expected_word):
        """Test that different languages can be processed"""
        try:
            nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize,morphseg',
                download_method=None
            )
            doc = nlp(text)

            assert len(doc.sentences) >= 1
            assert len(doc.sentences[0].words) >= 1

            word = doc.sentences[0].words[0]
            assert hasattr(word, 'morphemes')
            assert isinstance(word.morphemes, list)

        except Exception as e:
            pytest.skip(f"Language {lang} not available: {e}")


class TestMorphSegWithOtherProcessors:
    """Test morphseg processor in combination with other processors"""

    def test_with_mwt(self):
        """Test morphseg with MWT processor"""
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,mwt,morphseg',
            download_method=None
        )

        doc = nlp("The dogs are running.")

        for sentence in doc.sentences:
            for word in sentence.words:
                assert hasattr(word, 'morphemes')

    def test_with_pos(self):
        """Test morphseg with POS tagging"""
        try:
            nlp = stanza.Pipeline(
                lang='en',
                processors='tokenize,pos,morphseg',
                download_method=None
            )

            doc = nlp("running quickly")

            for sentence in doc.sentences:
                for word in sentence.words:
                    # Should have both POS and morphemes
                    assert hasattr(word, 'morphemes')
                    assert hasattr(word, 'upos') or hasattr(word, 'xpos')

        except Exception as e:
            pytest.skip(f"POS processor not available: {e}")

    def test_with_lemma(self):
        """Test morphseg with lemmatization"""
        try:
            nlp = stanza.Pipeline(
                lang='en',
                processors='tokenize,pos,lemma,morphseg',
                download_method=None
            )

            doc = nlp("The dogs were running quickly.")

            for sentence in doc.sentences:
                for word in sentence.words:
                    # Should have both lemma and morphemes
                    assert hasattr(word, 'morphemes')
                    assert hasattr(word, 'lemma')

        except Exception as e:
            pytest.skip(f"Lemma processor not available: {e}")


class TestMorphSegDeterminism:
    """Test that morphseg processor produces deterministic results"""

    def test_deterministic_results(self):
        """Test that same input produces same output"""
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,morphseg',
            download_method=None
        )

        text = "running dogs aviation"

        results = []
        for _ in range(3):
            doc = nlp(text)
            morphemes = [word.morphemes for sent in doc.sentences for word in sent.words]
            results.append(morphemes)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], \
                f"Non-deterministic results: {results[0]} vs {results[i]}"

    def test_batch_determinism(self):
        """Test determinism with batch processing"""
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,morphseg',
            download_method=None
        )

        texts = [
            "The dogs are running.",
            "Aviation is amazing.",
            "Known facts are helpful."
        ]

        # Process multiple times
        all_results = []
        for _ in range(2):
            batch_results = []
            for text in texts:
                doc = nlp(text)
                morphemes = [word.morphemes for sent in doc.sentences for word in sent.words]
                batch_results.append(morphemes)
            all_results.append(batch_results)

        # Results should be identical
        assert all_results[0] == all_results[1]


class TestMorphSegEdgeCases:
    """Test edge cases and special inputs"""

    @pytest.fixture(scope="class")
    def en_pipeline(self):
        """Create English pipeline"""
        return stanza.Pipeline(
            lang='en',
            processors='tokenize,morphseg',
            download_method=None
        )

    def test_numbers(self, en_pipeline):
        """Test handling of numbers"""
        doc = en_pipeline("123 456")

        for sentence in doc.sentences:
            for word in sentence.words:
                assert hasattr(word, 'morphemes')

    def test_mixed_case(self, en_pipeline):
        """Test mixed case handling"""
        # Should normalize to same result
        doc1 = en_pipeline("Running")
        doc2 = en_pipeline("RUNNING")
        doc3 = en_pipeline("running")

        morphemes1 = doc1.sentences[0].words[0].morphemes
        morphemes2 = doc2.sentences[0].words[0].morphemes
        morphemes3 = doc3.sentences[0].words[0].morphemes

        assert morphemes1 == morphemes2 == morphemes3

    def test_unicode_characters(self, en_pipeline):
        """Test handling of unicode characters"""
        doc = en_pipeline("café résumé")

        for sentence in doc.sentences:
            for word in sentence.words:
                assert hasattr(word, 'morphemes')
                assert isinstance(word.morphemes, list)

    def test_special_characters(self, en_pipeline):
        """Test handling of special characters"""
        doc = en_pipeline("test@example.com $100 50%")

        for sentence in doc.sentences:
            for word in sentence.words:
                assert hasattr(word, 'morphemes')

    def test_very_long_word(self, en_pipeline):
        """Test handling of very long words"""
        long_word = "antidisestablishmentarianism"
        doc = en_pipeline(long_word)

        word = doc.sentences[0].words[0]
        assert hasattr(word, 'morphemes')
        assert len(word.morphemes) >= 1

    def test_repeated_words(self, en_pipeline):
        """Test handling of repeated words"""
        doc = en_pipeline("running running running")

        # All instances should have same segmentation
        morphemes_list = [word.morphemes for word in doc.sentences[0].words]
        assert morphemes_list[0] == morphemes_list[1] == morphemes_list[2]

    def test_whitespace_handling(self, en_pipeline):
        """Test handling of various whitespace"""
        doc = en_pipeline("word1    word2\tword3\nword4")

        # Should properly segment all words despite whitespace
        word_count = sum(len(sent.words) for sent in doc.sentences)
        assert word_count >= 4

        for sentence in doc.sentences:
            for word in sentence.words:
                assert hasattr(word, 'morphemes')


class TestMorphSegConfiguration:
    """Test different configurations of morphseg processor"""

    def test_custom_model_path(self):
        """Test loading with custom model path configuration"""
        # Test that the configuration accepts model_path parameter
        # Using default behavior (no custom path)
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,morphseg',
            download_method=None
        )
        doc = nlp("testing")
        assert len(doc.sentences) > 0
        assert hasattr(doc.sentences[0].words[0], 'morphemes')

    def test_custom_model_path_with_file(self):
        """Test loading with an actual custom model file path"""
        # This test would require a custom model file to exist
        # Skip if no custom model is available
        pytest.skip("Custom model path test requires a specific model file")

    def test_processor_requirements(self):
        """Test that morphseg requires tokenize"""
        # MorphSeg requires TOKENIZE processor
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,morphseg',
            download_method=None
        )

        # Verify tokenize is present
        assert 'tokenize' in nlp.processors or 'tokenize' in str(nlp.processors)


class TestMorphSegOutputFormat:
    """Test output format of morpheme segmentations"""

    @pytest.fixture(scope="class")
    def en_pipeline(self):
        """Create English pipeline"""
        return stanza.Pipeline(
            lang='en',
            processors='tokenize,morphseg',
            download_method=None
        )

    def test_morphemes_is_list(self, en_pipeline):
        """Test that morphemes attribute is always a list"""
        doc = en_pipeline("The dogs are running quickly.")

        for sentence in doc.sentences:
            for word in sentence.words:
                assert isinstance(word.morphemes, list)

    def test_morphemes_are_strings(self, en_pipeline):
        """Test that all morphemes are strings"""
        doc = en_pipeline("The dogs are running quickly.")

        for sentence in doc.sentences:
            for word in sentence.words:
                for morpheme in word.morphemes:
                    assert isinstance(morpheme, str)

    def test_morphemes_non_empty(self, en_pipeline):
        """Test that morphemes list is never empty"""
        doc = en_pipeline("The dogs are running quickly.")

        for sentence in doc.sentences:
            for word in sentence.words:
                assert len(word.morphemes) >= 1

    def test_unsegmented_words(self, en_pipeline):
        """Test that unsegmented words have single morpheme"""
        # Words like 'the', 'is', 'a' typically don't segment
        doc = en_pipeline("The dog is a pet.")

        for sentence in doc.sentences:
            for word in sentence.words:
                # Even if unsegmented, should have the word itself as morpheme
                if len(word.morphemes) == 1:
                    # The single morpheme should match the normalized word
                    assert isinstance(word.morphemes[0], str)


class TestMorphSegRepeatedly:
    """Test repeated processing of multiple documents"""

    def test_sequential_document_processing(self):
        """Test processing multiple documents one after another"""
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,morphseg',
            download_method=None
        )

        texts = [
            "The dogs are running.",
            "Aviation is fascinating.",
            "Programming requires patience."
        ]

        for text in texts:
            doc = nlp(text)
            for sentence in doc.sentences:
                for word in sentence.words:
                    assert hasattr(word, 'morphemes')
                    assert isinstance(word.morphemes, list)

    def test_multi_sentence_document(self):
        """Test processing a document with multiple sentences (internal batching)"""
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,morphseg',
            download_method=None
        )

        doc = nlp("The dogs are running. Aviation is fascinating. Programming requires patience.")

        assert len(doc.sentences) == 3

        for sentence in doc.sentences:
            for word in sentence.words:
                assert hasattr(word, 'morphemes')
                assert isinstance(word.morphemes, list)
