"""
Unit tests for stanza.utils.byte_reader

Run with:
    pytest stanza/tests/utils/test_byte_reader.py
"""

import gzip
import io
import lzma
import struct

import pytest

from stanza.utils.byte_reader import ByteReader

pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

# a header, two NUL terminated strings, then 100 numbered int32
DATA = struct.pack('<iq', 7, 123456789012) + b'unban\0mox\0' + struct.pack('<100i', *range(100))

def make_reader(data=DATA, chunk_size=None):
    reader = ByteReader(io.BytesIO(data))
    if chunk_size is not None:
        reader.CHUNK_SIZE = chunk_size
    return reader

@pytest.mark.parametrize("chunk_size", [None, 1, 3])
def test_read_sequence(chunk_size):
    """
    The same sequence of reads works whether or not records straddle chunk boundaries
    """
    reader = make_reader(chunk_size=chunk_size)
    assert reader.unpack('<iq') == (7, 123456789012)
    assert reader.read_cstring() == b'unban'
    assert reader.read_cstring() == b'mox'
    assert reader.unpack('<3i') == (0, 1, 2)
    assert reader.read(4) == struct.pack('<i', 3)

@pytest.mark.parametrize("chunk_size", [None, 2])
def test_skip(chunk_size):
    """
    Skipping works both within the buffer and past the end of it
    """
    reader = make_reader(chunk_size=chunk_size)
    reader.skip(12)
    assert reader.read_cstring() == b'unban'
    reader.skip(4)
    # skip within the buffered data
    reader.skip(4 * 10)
    assert reader.unpack('<i') == (10,)
    # skip much further than anything buffered
    reader.skip(4 * 80)
    assert reader.unpack('<i') == (91,)

@pytest.mark.parametrize("opener", [gzip, lzma])
def test_compressed_skip(opener):
    """
    Skipping uses seek, which also works on compressed streams
    """
    compressed = io.BytesIO()
    with opener.open(compressed, "wb") as fout:
        fout.write(DATA)
    compressed.seek(0)
    with opener.open(compressed, "rb") as fin:
        reader = ByteReader(fin)
        reader.CHUNK_SIZE = 8
        assert reader.unpack('<i') == (7,)
        reader.skip(8 + len(b'unban\0mox\0') + 4 * 50)
        assert reader.unpack('<2i') == (50, 51)

def test_read_eof():
    reader = make_reader(DATA[:10])
    with pytest.raises(EOFError):
        reader.read(11)

def test_cstring_eof():
    reader = make_reader(b'no terminator', chunk_size=4)
    with pytest.raises(EOFError):
        reader.read_cstring()

def test_empty_cstring():
    reader = make_reader(b'\0\0x\0')
    assert reader.read_cstring() == b''
    assert reader.read_cstring() == b''
    assert reader.read_cstring() == b'x'
