"""
Integration tests for morphseg
"""

import pytest
from morphseg import MorphemeSegmenter

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

class TestIntegration:

    def test_full_pipeline(self):
        """Test complete segmentation pipeline"""
        segmenter = MorphemeSegmenter('en')

        text = "According to all known laws of aviation, there is no way a bee should be able to fly."
        result = segmenter.segment(text, output_string=False)

        # Should segment multiple words
        assert len(result) > 10

        # Each word should have at least one morpheme
        for word_morphemes in result:
            assert len(word_morphemes) >= 1

    def test_consistency_across_modes(self):
        """Test that list and string output modes are consistent"""
        segmenter = MorphemeSegmenter('en')

        words = ['running', 'dogs', 'aviation']

        for word in words:
            list_result = segmenter.segment(word, output_string=False)
            string_result = segmenter.segment(word, output_string=True, delimiter=' @@')

            # String result should be reconstructable from list result
            expected_string = ' @@'.join(list_result[0])
            assert string_result == expected_string, \
                f"List and string outputs don't match for '{word}'"

    def test_unicode_handling(self):
        """Test handling of unicode characters"""
        segmenter = MorphemeSegmenter('fr')

        text = "café résumé"
        result = segmenter.segment(text, output_string=False)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_mixed_case(self):
        """Test handling of mixed case input"""
        segmenter = MorphemeSegmenter('en')

        # Should normalize to lowercase
        result1 = segmenter.segment('Running', output_string=False)
        result2 = segmenter.segment('RUNNING', output_string=False)
        result3 = segmenter.segment('running', output_string=False)

        # All should produce the same result
        assert result1 == result2 == result3
