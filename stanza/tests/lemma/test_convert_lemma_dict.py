"""
Tests for convert_lemma_dict.py, including its handling of hostile input.

The hostile checkpoints use os.mkdir on a marker directory as their
payload: harmless, but easy to detect if it ever runs.
"""

import gzip
import os
import pickle
from collections import OrderedDict

import orjson
import pytest
import torch

from stanza.utils.lemma import convert_lemma_dict

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

POS_DICT = {"*": {"ran": "run"}, "NOUN": {"saw": "saw"}}


class MkdirPayload:
    """Pickles as a call to os.mkdir(marker)."""
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


def save_checkpoint(path, dicts, dicts_version=None):
    checkpoint = {
        "model": {"weight": torch.zeros(2)},
        "vocab": {"char": {"lang": "en"}},
        "config": {"dict_only": False, "charlm_forward_file": None, "charlm_backward_file": None},
        "dicts": dicts,
    }
    if dicts_version is not None:
        checkpoint["dicts_version"] = dicts_version
    torch.save(checkpoint, str(path), _use_new_zipfile_serialization=False)


def load_converted_dict(path):
    checkpoint = torch.load(str(path), weights_only=True)
    assert checkpoint["dicts_version"] == convert_lemma_dict._DICTS_VERSION_POS
    return orjson.loads(gzip.decompress(checkpoint["dicts"]))


def assert_refused(tmp_path, input_path, marker=None):
    """Conversion raises, writes nothing, and the payload (if any) did not run."""
    output_path = tmp_path / "output.pt"
    with pytest.raises(RuntimeError, match="(?i)do not use this file"):
        convert_lemma_dict.convert(str(input_path), str(output_path))
    assert not output_path.exists()
    assert not (tmp_path / "output.pt.tmp").exists()
    if marker is not None:
        assert not marker.exists(), "payload was executed"


def test_convert_legacy(tmp_path):
    word_dict = {"ran": "run"}
    composite_dict = {("ran", "VERB"): "run", ("saw", "NOUN"): "saw"}
    input_path = tmp_path / "v1.pt"
    save_checkpoint(input_path, (word_dict, composite_dict))
    output_path = tmp_path / "v3.pt"
    convert_lemma_dict.convert(str(input_path), str(output_path))
    assert load_converted_dict(output_path) == POS_DICT


def test_convert_v2(tmp_path):
    input_path = tmp_path / "v2.pt"
    save_checkpoint(input_path, gzip.compress(pickle.dumps(POS_DICT)),
                    convert_lemma_dict._DICTS_VERSION_POS_PICKLE)
    output_path = tmp_path / "v3.pt"
    convert_lemma_dict.convert(str(input_path), str(output_path))
    assert load_converted_dict(output_path) == POS_DICT


def test_v2_dict_payload_refused(tmp_path):
    """A v2 dict that pickles a function call is refused without running it."""
    marker = tmp_path / "payload_ran"
    hostile = {"*": {"ran": MkdirPayload(marker)}}
    # sanity check: an unrestricted unpickler would run the payload
    pickle.loads(pickle.dumps(hostile))
    assert marker.exists()
    os.rmdir(marker)

    input_path = tmp_path / "hostile_v2.pt"
    save_checkpoint(input_path, gzip.compress(pickle.dumps(hostile)),
                    convert_lemma_dict._DICTS_VERSION_POS_PICKLE)
    assert_refused(tmp_path, input_path, marker)


def test_checkpoint_payload_refused(tmp_path):
    """A payload in the checkpoint itself, outside the dict, is refused without running it."""
    marker = tmp_path / "payload_ran"
    input_path = tmp_path / "hostile_checkpoint.pt"
    save_checkpoint(input_path, MkdirPayload(marker), convert_lemma_dict._DICTS_VERSION_POS_PICKLE)
    assert_refused(tmp_path, input_path, marker)


def test_v2_dict_wrong_shape_refused(tmp_path):
    input_path = tmp_path / "bad_shape_v2.pt"
    save_checkpoint(input_path, gzip.compress(pickle.dumps({"*": {"ran": 3}})),
                    convert_lemma_dict._DICTS_VERSION_POS_PICKLE)
    assert_refused(tmp_path, input_path)


def test_v2_dict_not_gzip_refused(tmp_path):
    input_path = tmp_path / "not_gzip_v2.pt"
    save_checkpoint(input_path, pickle.dumps(POS_DICT), convert_lemma_dict._DICTS_VERSION_POS_PICKLE)
    with pytest.raises(gzip.BadGzipFile):
        convert_lemma_dict.convert(str(input_path), str(tmp_path / "output.pt"))
    assert not (tmp_path / "output.pt").exists()


def test_unpickler_rejects_harmless_classes():
    """Even classes that are safe in principle are refused, since a lemma dict needs none."""
    with pytest.raises(pickle.UnpicklingError, match="collections.OrderedDict"):
        convert_lemma_dict._load_v2_pos_dict(gzip.compress(pickle.dumps(OrderedDict(POS_DICT))))


def test_bounded_gunzip():
    data = gzip.compress(b"\0" * 1000)
    assert len(convert_lemma_dict._bounded_gunzip(data, limit=1000)) == 1000
    with pytest.raises(ValueError, match="exceeds"):
        convert_lemma_dict._bounded_gunzip(data, limit=999)
