"""
Tests for MorphemeSegmenter class
"""

import pytest
from morphseg import MorphemeSegmenter

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

class TestMorphemeSegmenter:

    @pytest.fixture(scope="class")
    def english_segmenter(self):
        """Load English model once for all tests"""
        return MorphemeSegmenter('en')

    def test_basic_segmentation(self, english_segmenter):
        """Test basic morpheme segmentation"""
        result = english_segmenter.segment('running', output_string=False)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) >= 1

    def test_multiple_words(self, english_segmenter):
        """Test segmentation of multiple words"""
        result = english_segmenter.segment('running quickly', output_string=False)

        assert isinstance(result, list)
        assert len(result) == 2
        for segmentation in result:
            assert isinstance(segmentation, list)
            assert len(segmentation) >= 1

    def test_known_segmentations(self, english_segmenter):
        """Test known morpheme segmentations"""
        test_cases = {
            'dogs': ['dog', 's'],
            'aviation': ['aviate', 'ion'],
            'known': ['know', 'n'],
        }

        for word, expected in test_cases.items():
            result = english_segmenter.segment(word, output_string=False)
            assert result[0] == expected, f"Expected {expected}, got {result[0]} for '{word}'"

    def test_output_string_mode(self, english_segmenter):
        """Test string output mode"""
        result = english_segmenter.segment('running quickly', output_string=True)

        assert isinstance(result, str)
        assert ' @@' in result  # Default delimiter

    def test_custom_delimiter(self, english_segmenter):
        """Test custom delimiter in output"""
        result = english_segmenter.segment('running', output_string=True, delimiter='-')

        assert isinstance(result, str)
        assert '-' in result or result == 'running'  # May be unsegmented

    def test_empty_input(self, english_segmenter):
        """Test handling of empty input"""
        result = english_segmenter.segment('', output_string=False)
        assert result == []

        result = english_segmenter.segment('', output_string=True)
        assert result == ""

    def test_single_character(self, english_segmenter):
        """Test single character input"""
        result = english_segmenter.segment('a', output_string=False)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == ['a']

    def test_punctuation(self, english_segmenter):
        """Test handling of punctuation"""
        result = english_segmenter.segment('Hello, world!', output_string=False)

        assert isinstance(result, list)
        # Should segment only words, not punctuation
        assert len(result) > 0


class TestDeterminism:
    """
    Tests to ensure predictions are deterministic
    """

    def test_deterministic_predictions(self):
        """Test that same input produces same output consistently"""
        segmenter = MorphemeSegmenter('en')

        test_words = ['running', 'dogs', 'quickly', 'aviation']

        for word in test_words:
            results = []
            for _ in range(5):
                result = segmenter.segment(word, output_string=False)
                results.append(result)

            # All results should be identical
            for i in range(1, len(results)):
                assert results[i] == results[0], \
                    f"Non-deterministic results for '{word}': {results[0]} vs {results[i]}"

    def test_deterministic_batch(self):
        """Test determinism with batch processing"""
        segmenter = MorphemeSegmenter('en')

        text = "The dogs are running quickly through the fields."

        results = []
        for _ in range(3):
            result = segmenter.segment(text, output_string=False)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], \
                f"Non-deterministic batch results: {results[0]} vs {results[i]}"


class TestMultilingual:

    @pytest.mark.parametrize("lang", ['cs', 'en', 'es', 'fr', 'hu', 'it', 'la', 'ru'])
    def test_language_loading(self, lang):
        """Test that all supported languages can be loaded"""
        segmenter = MorphemeSegmenter(lang)
        assert segmenter.lang == lang
        assert segmenter.sequence_labeller is not None

    @pytest.mark.parametrize("lang,word", [
        ('en', 'running'),
        ('es', 'corriendo'),
        ('fr', 'rapidement'),
        ('ru', 'бегущий'),  # Russian instead of German
    ])
    def test_multilingual_segmentation(self, lang, word):
        """Test segmentation across languages"""
        if lang not in MorphemeSegmenter.PRETRAINED_MODEL_LANGS:
            pytest.skip(f"Language {lang} not supported")

        segmenter = MorphemeSegmenter(lang)
        result = segmenter.segment(word, output_string=False)

        assert isinstance(result, list)
        assert len(result) >= 1


class TestErrorHandling:

    def test_invalid_language(self):
        """Test handling of invalid language code"""
        with pytest.warns(UserWarning):
            segmenter = MorphemeSegmenter('invalid_lang')
            assert segmenter.sequence_labeller is None

    def test_invalid_input_type(self):
        """Test handling of invalid input types"""
        segmenter = MorphemeSegmenter('en')

        with pytest.raises(ValueError, match="Input sequence must be a string"):
            segmenter.segment(123)

        with pytest.raises(ValueError, match="Input sequence must be a string"):
            segmenter.segment(['not', 'a', 'string'])

    def test_invalid_output_string_type(self):
        """Test handling of invalid output_string parameter"""
        segmenter = MorphemeSegmenter('en')

        with pytest.raises(ValueError, match="output_string must be a boolean"):
            segmenter.segment('test', output_string='yes')

    def test_invalid_delimiter_type(self):
        """Test handling of invalid delimiter parameter"""
        segmenter = MorphemeSegmenter('en')

        with pytest.raises(ValueError, match="Delimiter must be a string"):
            segmenter.segment('test', delimiter=123)

    def test_model_not_trained(self):
        """Test error when using untrained model"""
        segmenter = MorphemeSegmenter('en')
        segmenter.sequence_labeller = None

        with pytest.raises(RuntimeError, match="Model not trained"):
            segmenter.segment('test')


class TestModelState:
    """
    Tests to ensure model is in correct state
    """

    def test_model_in_eval_mode(self):
        """Test that loaded model is in eval mode"""
        segmenter = MorphemeSegmenter('en')

        # Check that model is in eval mode
        assert not segmenter.sequence_labeller.model.model.training, \
            "Model should be in eval mode after loading"

    def test_model_stays_in_eval_mode(self):
        """Test that model stays in eval mode after predictions"""
        segmenter = MorphemeSegmenter('en')

        # Make several predictions
        for _ in range(3):
            segmenter.segment('running', output_string=False)

        # Model should still be in eval mode
        assert not segmenter.sequence_labeller.model.model.training, \
            "Model should remain in eval mode after predictions"
