"""
A sequential reader for binary files with fixed size records and NUL terminated strings

Works on regular files as well as the compressed streams returned by
stanza.models.common.utils.open_read_binary.  Use it like this:

  with open_read_binary(filename) as fin:
      reader = ByteReader(fin)
      magic, version = reader.unpack('<ii')
      name = reader.read_cstring()
"""

import os
import struct

class ByteReader:
    """
    Sequential reader over a binary stream with a lookahead buffer

    Supports fixed size reads, struct unpacking, NUL terminated
    strings, and skipping forward.  Skipping uses seek, which is fast
    on a regular file and still works on gzip, xz, and zip streams,
    where it decompresses and discards the skipped bytes.
    """
    CHUNK_SIZE = 1 << 20

    def __init__(self, fin):
        self.fin = fin
        self.buf = b''
        self.pos = 0

    def available(self):
        """
        Number of bytes in the buffer which have not been read yet
        """
        return len(self.buf) - self.pos

    def fill(self, needed):
        """
        Buffer at least needed bytes, or as many as are left in the stream
        """
        have = self.available()
        if have >= needed:
            return
        chunks = [self.buf[self.pos:]]
        while have < needed:
            data = self.fin.read(max(self.CHUNK_SIZE, needed - have))
            if not data:
                break
            chunks.append(data)
            have += len(data)
        self.buf = b''.join(chunks)
        self.pos = 0

    def read(self, size):
        """
        Read exactly size bytes, raising EOFError if the stream ends first
        """
        self.fill(size)
        if self.available() < size:
            raise EOFError("Unexpected end of file: wanted %d bytes, only %d left" % (size, self.available()))
        data = self.buf[self.pos:self.pos+size]
        self.pos += size
        return data

    def unpack(self, fmt):
        """
        Read and unpack one struct with the given format
        """
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def read_cstring(self):
        """
        Read bytes up to the next NUL, consuming the NUL but not returning it
        """
        while True:
            end = self.buf.find(b'\0', self.pos)
            if end >= 0:
                data = self.buf[self.pos:end]
                self.pos = end + 1
                return data
            have = self.available()
            self.fill(have + 1)
            if self.available() == have:
                raise EOFError("Unexpected end of file while reading a string")

    def skip(self, size):
        """
        Move forward size bytes without returning them
        """
        have = self.available()
        if size <= have:
            self.pos += size
            return
        self.buf = b''
        self.pos = 0
        self.fin.seek(size - have, os.SEEK_CUR)
