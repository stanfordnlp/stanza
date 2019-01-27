from __future__ import absolute_import

from io import BytesIO

from google.protobuf.internal.encoder import _EncodeVarint
from google.protobuf.internal.decoder import _DecodeVarint
from .CoreNLP_pb2 import *

def parseFromDelimitedString(obj, buf, offset=0):
    """
    Stanford CoreNLP uses the Java "writeDelimitedTo" function, which
    writes the size (and offset) of the buffer before writing the object.
    This function handles parsing this message starting from offset 0.

    @returns how many bytes of @buf were consumed.
    """
    size, pos = _DecodeVarint(buf, offset)
    obj.ParseFromString(buf[offset+pos:offset+pos+size])
    return pos+size

def writeToDelimitedString(obj, stream=None):
    """
    Stanford CoreNLP uses the Java "writeDelimitedTo" function, which
    writes the size (and offset) of the buffer before writing the object.
    This function handles parsing this message starting from offset 0.

    @returns how many bytes of @buf were consumed.
    """
    if stream is None:
        stream = BytesIO()

    _EncodeVarint(stream.write, obj.ByteSize(), True)
    stream.write(obj.SerializeToString())
    return stream

def to_text(sentence):
    """
    Helper routine that converts a Sentence protobuf to a string from
    its tokens.
    """
    text = ""
    for i, tok in enumerate(sentence.token):
        if i != 0:
            text += tok.before
        text += tok.word
    return text
