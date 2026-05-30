import glob
import os
import shutil
import tempfile
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

import stanza
from stanza.models.common.foundation_cache import FoundationCache, load_charlm
from stanza.tests import TEST_MODELS_DIR

pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

BERT_MODEL = "hf-internal-testing/tiny-bert"

def make_tiny_bert():
    """Load the tiny-bert model and tokenizer directly."""
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, add_prefix_space=True)
    tokenizer.model_max_length = 512
    return model, tokenizer


def test_charlm_cache():
    models_path = os.path.join(TEST_MODELS_DIR, "en", "backward_charlm", "*")
    models = glob.glob(models_path)
    # we expect at least one English model downloaded for the tests
    assert len(models) >= 1
    model_file = models[0]

    cache = FoundationCache()
    with tempfile.TemporaryDirectory(dir=".") as test_dir:
        temp_file = os.path.join(test_dir, "charlm.pt")
        shutil.copy2(model_file, temp_file)
        # this will work
        model = load_charlm(temp_file)

        # this will save the model
        model = cache.load_charlm(temp_file)

    # this should no longer work
    with pytest.raises(FileNotFoundError):
        model = load_charlm(temp_file)

    # it should remember the cached version
    model = cache.load_charlm(temp_file)

class TestFoundationCacheLoadBert:
    def test_loads_successfully(self):
        cache = FoundationCache()
        model, tokenizer = cache.load_bert(BERT_MODEL)
        assert model is not None
        assert tokenizer is not None

    def test_same_object_returned_on_second_call(self):
        cache = FoundationCache()
        model1, _ = cache.load_bert(BERT_MODEL)
        model2, _ = cache.load_bert(BERT_MODEL)
        assert model1 is model2

    def test_none_model_name_returns_none(self):
        cache = FoundationCache()
        model, tokenizer = cache.load_bert(None)
        assert model is None
        assert tokenizer is None

    def test_gradient_checkpointing_enabled_on_cached_model(self):
        cache = FoundationCache()
        # First call without checkpointing
        model1, _ = cache.load_bert(BERT_MODEL, enable_gradient_checkpointing=False)
        assert not model1.is_gradient_checkpointing
        # Second call enables it on the same (cached) object
        model2, _ = cache.load_bert(BERT_MODEL, enable_gradient_checkpointing=True)
        assert model1 is model2
        assert model2.is_gradient_checkpointing

    def test_gradient_checkpointing_disabled_by_default(self):
        cache = FoundationCache()
        model, _ = cache.load_bert(BERT_MODEL)
        assert not model.is_gradient_checkpointing


class TestFoundationCacheLoadBertWithPeft:
    def test_returns_peft_name(self):
        cache = FoundationCache()
        model, tokenizer, peft_name = cache.load_bert_with_peft(BERT_MODEL, "depparse")
        assert peft_name is not None
        assert "depparse" in peft_name

    def test_peft_name_increments_on_second_call(self):
        cache = FoundationCache()
        _, _, peft_name1 = cache.load_bert_with_peft(BERT_MODEL, "depparse")
        _, _, peft_name2 = cache.load_bert_with_peft(BERT_MODEL, "depparse")
        assert peft_name1 != peft_name2

    def test_no_peft_name_returns_none(self):
        cache = FoundationCache()
        _, _, peft_name = cache.load_bert_with_peft(BERT_MODEL, None)
        assert peft_name is None

    def test_none_model_name_returns_none(self):
        cache = FoundationCache()
        model, tokenizer, peft_name = cache.load_bert_with_peft(None, "depparse")
        assert model is None
        assert tokenizer is None
        assert peft_name is None

    def test_gradient_checkpointing_enabled_before_peft_name_assigned(self):
        """
        The critical ordering test: gradient checkpointing must be enabled on
        the base model before PEFT wraps it.  We verify this by checking that
        the model has gradient checkpointing on when the peft_name is returned,
        meaning the flag was set before any PEFT adapter was registered.

        Since foundation_cache only assigns the peft_name (wrapping is done
        by the caller), what we're really testing here is that checkpointing
        is applied to the base model at the right point in load_bert_with_peft.
        """
        cache = FoundationCache()
        model, _, peft_name = cache.load_bert_with_peft(
            BERT_MODEL, "depparse", enable_gradient_checkpointing=True
        )
        assert model.is_gradient_checkpointing

    def test_gradient_checkpointing_disabled_by_default_with_peft(self):
        cache = FoundationCache()
        model, _, _ = cache.load_bert_with_peft(BERT_MODEL, "depparse")
        assert not model.is_gradient_checkpointing

    def test_gradient_checkpointing_persists_on_cached_model(self):
        """Once enabled on a cached model, gradient checkpointing stays on
        for subsequent callers that also request it."""
        cache = FoundationCache()
        model1, _, _ = cache.load_bert_with_peft(
            BERT_MODEL, "depparse", enable_gradient_checkpointing=True
        )
        model2, _, _ = cache.load_bert_with_peft(
            BERT_MODEL, "ner", enable_gradient_checkpointing=True
        )
        assert model1 is model2
        assert model2.is_gradient_checkpointing

class TestFoundationCacheThreadSafety:
    def test_concurrent_loads_return_same_object(self):
        """
        Multiple threads loading the same model should all receive the same
        object and the model should only be loaded once.

        To make the race window wide enough to reliably expose non-thread-safe
        implementations, we patch AutoModel.from_pretrained to sleep briefly.
        Without the lock in FoundationCache, multiple threads would each see
        the cache as empty and each start loading, potentially constructing
        multiple model instances.  With the lock, only one thread loads and
        the rest wait and then receive the cached result.
        """

        cache = FoundationCache()
        results = []
        errors = []
        real_model, real_tokenizer = make_tiny_bert()

        def slow_from_pretrained(name, **kwargs):
            # Sleep long enough for all threads to enter the function
            # before any of them returns, maximising the chance of a race
            time.sleep(0.1)
            return real_model

        with patch('transformers.AutoModel.from_pretrained',
                   side_effect=slow_from_pretrained):
            def load():
                try:
                    model, _ = cache.load_bert(BERT_MODEL)
                    results.append(id(model))
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=load) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors, f"Errors during concurrent load: {errors}"
        assert len(set(results)) == 1, (
            "All threads should receive the same cached model object. "
            f"Got {len(set(results))} distinct objects — the cache is not thread-safe."
        )

    def test_model_loaded_exactly_once(self):
        """
        Complementary to the above: verify the slow load function is only
        called once even under concurrent access, not once per thread.
        """
        cache = FoundationCache()
        real_model, _ = make_tiny_bert()
        load_count = []

        def slow_from_pretrained(name, **kwargs):
            time.sleep(0.1)
            load_count.append(1)
            return real_model

        with patch('transformers.AutoModel.from_pretrained',
                   side_effect=slow_from_pretrained):
            threads = [threading.Thread(target=lambda: cache.load_bert(BERT_MODEL))
                       for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(load_count) == 1, (
            f"Model was loaded {len(load_count)} times instead of once — "
            "the lock is not preventing redundant loads."
        )


class TestNoTransformerFoundationCache:
    def test_does_not_return_cached_model(self):
        """
        NoTransformerFoundationCache bypasses the bert cache by delegating to
        the module-level load_bert function in foundation_cache, which in turn
        calls bert_embedding.load_bert directly.  The result is a fresh model
        instance that won't accidentally share finetuned weights with other users
        of the same transformer name.
        """
        from stanza.models.common.foundation_cache import (
            FoundationCache, NoTransformerFoundationCache
        )
        cache = FoundationCache()
        # Prime the cache with a model
        model1, _ = cache.load_bert(BERT_MODEL)

        # NoTransformerFoundationCache wraps the same cache but bypasses it
        # for transformer lookups
        no_cache = NoTransformerFoundationCache(cache)
        model2, _ = no_cache.load_bert(BERT_MODEL)

        assert model1 is not model2, (
            "NoTransformerFoundationCache should return a fresh model "
            "rather than the cached one, so finetuned weights are not "
            "accidentally shared with other models"
        )

    def test_repeated_calls_also_bypass_cache(self):
        """
        Even a second call through NoTransformerFoundationCache should return
        a fresh model, not a cached one from either the underlying cache or
        a hypothetical per-instance cache.
        """
        from stanza.models.common.foundation_cache import (
            FoundationCache, NoTransformerFoundationCache
        )
        cache = FoundationCache()
        no_cache = NoTransformerFoundationCache(cache)
        model1, _ = no_cache.load_bert(BERT_MODEL)
        model2, _ = no_cache.load_bert(BERT_MODEL)
        assert model1 is not model2

    def test_gradient_checkpointing_still_works(self):
        from stanza.models.common.foundation_cache import (
            FoundationCache, NoTransformerFoundationCache
        )
        cache = FoundationCache()
        no_cache = NoTransformerFoundationCache(cache)
        model, _ = no_cache.load_bert(
            BERT_MODEL, enable_gradient_checkpointing=True
        )
        assert model.is_gradient_checkpointing

