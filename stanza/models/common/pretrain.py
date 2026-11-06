"""
Supports for pretrained data.
"""
import codecs
import csv
import os
import re
import struct

import lzma
import logging
import numpy as np
import torch

from .vocab import BaseVocab, VOCAB_PREFIX, UNK_ID

from stanza.models.common.utils import open_read_binary, open_read_text
from stanza.resources.common import DEFAULT_MODEL_DIR

from pickle import UnpicklingError
import warnings

logger = logging.getLogger('stanza')

class PretrainedWordVocab(BaseVocab):
    def build_vocab(self):
        self._id2unit = VOCAB_PREFIX + self.data
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

    def normalize_unit(self, unit):
        unit = super().normalize_unit(unit)
        if unit:
            unit = unit.replace(" ","\xa0")
        return unit

class Pretrain:
    """ A loader and saver for pretrained embeddings. """

    def __init__(self, filename=None, vec_filename=None, max_vocab=-1, save_to_file=True, csv_filename=None):
        self.filename = filename
        self._vec_filename = vec_filename
        self._csv_filename = csv_filename
        self._max_vocab = max_vocab
        self._save_to_file = save_to_file

    def __len__(self):
        return len(self.vocab)

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self.load()
        return self._vocab

    @property
    def emb(self):
        if not hasattr(self, '_emb'):
            self.load()
        return self._emb

    def load(self):
        if self.filename is not None and os.path.exists(self.filename):
            try:
                try:
                    data = torch.load(self.filename, lambda storage, loc: storage, weights_only=True)
                except UnpicklingError as e:
                    raise UnpicklingError("Unpickling %s failed.  If this is because it is an older model that needs weights_only=False, please convert it with a Stanza version 1.12.1 or earlier by loading and then saving." % self.filename) from e
                logger.debug("Loaded pretrain from {}".format(self.filename))
                if not isinstance(data, dict):
                    raise RuntimeError("File {} exists but is not a stanza pretrain file.  It is not a dict, whereas a Stanza pretrain should have a dict with 'emb' and 'vocab'".format(self.filename))
                if 'emb' not in data or 'vocab' not in data:
                    raise RuntimeError("File {} exists but is not a stanza pretrain file.  A Stanza pretrain file should have 'emb' and 'vocab' fields in its state dict".format(self.filename))
                self._vocab = PretrainedWordVocab.load_state_dict(data['vocab'])
                self._emb = data['emb']
                if isinstance(self._emb, np.ndarray):
                    self._emb = torch.from_numpy(self._emb)
                return
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                if not self._vec_filename and not self._csv_filename:
                    raise
                logger.warning("Pretrained file exists but cannot be loaded from {}, due to the following exception:\n\t{}".format(self.filename, e))
                vocab, emb = self.read_pretrain()
        else:
            if not self._vec_filename and not self._csv_filename:
                raise FileNotFoundError("Pretrained file {} does not exist, and no text/xz file was provided".format(self.filename))
            if self.filename is not None:
                logger.info("Pretrained filename %s specified, but file does not exist.  Attempting to load from text file" % self.filename)
            vocab, emb = self.read_pretrain()

        self._vocab = vocab
        self._emb = emb

        if self._save_to_file:
            # save to file
            assert self.filename is not None, "Filename must be provided to save pretrained vector to file."
            self.save(self.filename)

    def save(self, filename):
        directory, _ = os.path.split(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # should not infinite loop since the load function sets _vocab and _emb before trying to save
        data = {'vocab': self.vocab.state_dict(), 'emb': self.emb}
        try:
            torch.save(data, filename, _use_new_zipfile_serialization=False)
            logger.info("Saved pretrained vocab and vectors to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.warning("Saving pretrained data failed due to the following exception... continuing anyway.\n\t{}".format(e))


    def write_text(self, filename, header=False):
        """
        Write the vocab & values to a text file
        """
        with open(filename, "w") as fout:
            if header:
                word_dim = self.emb[0].shape[0]
                fout.write("%d %d\n" % (len(self.vocab), word_dim))
            for word_idx, word in enumerate(self.vocab):
                row = self.emb[word_idx].to("cpu")
                fout.write(word)
                fout.write(" ")
                fout.write(" ".join(["%.6f" % x.item() for x in row]))
                fout.write("\n")


    def read_pretrain(self):
        # load from pretrained filename
        if self._vec_filename is not None:
            words, emb, failed = self.read_from_file(self._vec_filename, self._max_vocab)
        elif self._csv_filename is not None:
            words, emb = self.read_from_csv(self._csv_filename)
        else:
            raise RuntimeError("Vector file is not provided.")

        if len(emb) - len(VOCAB_PREFIX) != len(words):
            raise RuntimeError("Loaded number of vectors does not match number of words.")

        # Use a fixed vocab size
        if self._max_vocab > len(VOCAB_PREFIX) and self._max_vocab < len(words) + len(VOCAB_PREFIX):
            words = words[:self._max_vocab - len(VOCAB_PREFIX)]
            emb = emb[:self._max_vocab]

        vocab = PretrainedWordVocab(words)
        
        return vocab, emb

    @staticmethod
    def read_from_csv(filename):
        """
        Read vectors from CSV

        Skips the first row
        """
        logger.info("Reading pretrained vectors from csv file %s ...", filename)
        with open_read_text(filename) as fin:
            csv_reader = csv.reader(fin)
            # the header of the thai csv vector file we have is just the number of columns
            # so we read past the first line
            for line in csv_reader:
                break
            lines = [line for line in csv_reader]

        rows = len(lines)
        cols = len(lines[0]) - 1

        emb = torch.zeros((rows + len(VOCAB_PREFIX), cols), dtype=torch.float32)
        for i, line in enumerate(lines):
            emb[i+len(VOCAB_PREFIX)] = torch.tensor([float(x) for x in line[-cols:]], dtype=torch.float32)
        words = [line[0].replace(' ', '\xa0') for line in lines]
        return words, emb

    @staticmethod
    def read_from_file(filename, max_vocab=None):
        """
        Read vectors from a text, word2vec binary, or fastText .bin file

        The format is detected from the contents of the file rather
        than its extension.  Any of the formats can be compressed in
        the ways open_read_binary understands.

        Returns words, emb, failed, where emb has the VOCAB_PREFIX rows
        at the start and failed is the number of entries skipped
        because they could not be decoded.
        """
        vector_format = detect_vector_format(filename)
        if vector_format == "fasttext":
            return read_fasttext_binary(filename, max_vocab)
        if vector_format == "word2vec_binary":
            return read_word2vec_binary(filename, max_vocab)
        return Pretrain.read_from_text_file(filename, max_vocab)

    @staticmethod
    def read_from_text_file(filename, max_vocab=None):
        """
        Read vectors from a text file with one word and its values per line

        The "rows cols" header line is optional.
        """
        logger.info("Reading pretrained vectors from %s ...", filename)

        # some vector files, such as Google News, use tabs
        tab_space_pattern = re.compile(r"[ \t]+")
        first = True
        cols = None
        lines = []
        failed = 0
        unk_line = None
        with open_read_binary(filename) as f:
            for i, line in enumerate(f):
                try:
                    line = line.decode()
                except UnicodeDecodeError:
                    failed += 1
                    continue
                line = line.rstrip()
                if not line:
                    continue
                pieces = tab_space_pattern.split(line)
                if first:
                    # the first line contains the number of word vectors and the dimensionality
                    # note that a 1d embedding with a number as the first entry
                    # will fail to read properly.  we ignore that case
                    first = False
                    if len(pieces) == 2:
                        cols = int(pieces[1])
                        continue

                if pieces[0] == '<unk>':
                    if unk_line is not None:
                        logger.error("More than one <unk> line in the pretrain!  Keeping the most recent one")
                    else:
                        logger.debug("Found an unk line while reading the pretrain")
                    unk_line = pieces
                else:
                    if not max_vocab or max_vocab < 0 or len(lines) < max_vocab:
                        lines.append(pieces)

        if cols is None:
            # another failure case: all words have spaces in them
            cols = min(len(x) for x in lines) - 1
        rows = len(lines)
        emb = torch.zeros((rows + len(VOCAB_PREFIX), cols), dtype=torch.float32)
        if unk_line is not None:
            emb[UNK_ID] = torch.tensor([float(x) for x in unk_line[-cols:]], dtype=torch.float32)
        for i, line in enumerate(lines):
            emb[i+len(VOCAB_PREFIX)] = torch.tensor([float(x) for x in line[-cols:]], dtype=torch.float32)

        # if there were word pieces separated with spaces, rejoin them with nbsp instead
        # this way, the normalize_unit method in PretrainedWordVocab can find the word at test time
        words = ['\xa0'.join(line[:-cols]) for line in lines]
        if failed > 0:
            logger.info("Failed to read %d lines from embedding", failed)
        return words, emb, failed


# every fastText .bin model starts with this int32
FASTTEXT_MAGIC = 793712314
# newest .bin layout this reader understands
FASTTEXT_MAX_VERSION = 12
# fastText hashes each byte as a sign extended int8, so bytes >= 0x80
# turn into 0xFFFFFF80 .. 0xFFFFFFFF before being mixed into the hash
FASTTEXT_HASH_BYTES = [b if b < 0x80 else b | 0xFFFFFF00 for b in range(256)]

def detect_vector_format(filename):
    """
    Guess the format of a word vector file from its first few bytes

    Returns one of "fasttext", "word2vec_binary", or "text"

    fastText .bin models start with a fixed magic number.  word2vec
    binary files and text files both start with a "rows cols" header,
    so for those, the bytes after the first word are checked.  A text
    file has a printable list of numbers there, whereas a binary file
    has raw float32 values, which in practice include control
    characters or invalid UTF-8.
    """
    with open_read_binary(filename) as fin:
        start = fin.read(4)
        if len(start) == 4 and struct.unpack('<i', start)[0] == FASTTEXT_MAGIC:
            return "fasttext"

        # a short header such as "3 4\n" can be entirely inside the first four bytes
        if b'\n' not in start:
            start = start + fin.readline(1000)
        header, _, first_word = start.partition(b'\n')
        header = header.split()
        if len(header) != 2 or not all(piece.isdigit() for piece in header):
            # no header, or not a "rows cols" header, so this can only be text
            return "text"
        cols = int(header[1])

        first_word = first_word + fin.read(1000)
        space = first_word.find(b' ')
        if space < 0:
            return "text"
        sample = first_word[space+1:space+1+4*cols]
        if len(sample) < 4*cols:
            sample = sample + fin.read(4*cols - len(sample))

    # incremental decoding tolerates a character cut off at the end of the sample
    decoder = codecs.getincrementaldecoder('utf-8')()
    try:
        text = decoder.decode(sample, final=False)
    except UnicodeDecodeError:
        return "word2vec_binary"
    if any(c < ' ' and c not in '\t\r\n' for c in text):
        return "word2vec_binary"
    return "text"

def build_embedding(vectors, cols, unk_vector=None):
    """
    Stack numpy vectors into an embedding with the VOCAB_PREFIX rows at the start
    """
    emb = torch.zeros((len(vectors) + len(VOCAB_PREFIX), cols), dtype=torch.float32)
    if unk_vector is not None:
        emb[UNK_ID] = torch.from_numpy(np.array(unk_vector, dtype=np.float32))
    if len(vectors) > 0:
        emb[len(VOCAB_PREFIX):] = torch.from_numpy(np.stack(vectors).astype(np.float32))
    return emb

def read_word2vec_binary(filename, max_vocab=None):
    """
    Read vectors in the binary format written by word2vec and gensim

    The file starts with a "rows cols" text header.  Each entry is the
    word, a space, and cols little endian float32 values, optionally
    followed by a newline.
    """
    logger.info("Reading word2vec binary vectors from %s ...", filename)
    limit = max_vocab if max_vocab and max_vocab > 0 else None

    words = []
    vectors = []
    unk_vector = None
    failed = 0
    with open_read_binary(filename) as fin:
        header = fin.readline().split()
        rows, cols = int(header[0]), int(header[1])
        row_bytes = 4 * cols
        for row in range(rows):
            if limit is not None and len(words) >= limit:
                break
            word = bytearray()
            while True:
                c = fin.read(1)
                if not c:
                    raise ValueError("Unexpected end of file in %s after %d of %d words" % (filename, row, rows))
                if c == b' ':
                    break
                if c == b'\n' and not word:
                    # newline left over from the end of the previous vector
                    continue
                word += c
            data = fin.read(row_bytes)
            if len(data) != row_bytes:
                raise ValueError("Unexpected end of file in %s after %d of %d words" % (filename, row, rows))
            try:
                word = word.decode('utf-8')
            except UnicodeDecodeError:
                failed += 1
                continue
            vector = np.frombuffer(data, dtype='<f4')
            if word == '<unk>':
                if unk_vector is not None:
                    logger.error("More than one <unk> line in the pretrain!  Keeping the most recent one")
                unk_vector = vector
            else:
                words.append(word)
                vectors.append(vector)

    if failed > 0:
        logger.info("Failed to read %d words from embedding", failed)
    return words, build_embedding(vectors, cols, unk_vector), failed

class ByteReader:
    """
    Sequential reader over a binary stream with a lookahead buffer

    Supports fixed size reads, NUL terminated strings, and skipping
    forward, which covers everything the fastText reader needs.
    Skipping uses seek, which is fast on a regular file and still
    works on the compressed streams open_read_binary returns.
    """
    CHUNK_SIZE = 1 << 20

    def __init__(self, fin):
        self.fin = fin
        self.buf = b''
        self.pos = 0

    def available(self):
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
        self.fill(size)
        if self.available() < size:
            raise EOFError("Unexpected end of file: wanted %d bytes, only %d left" % (size, self.available()))
        data = self.buf[self.pos:self.pos+size]
        self.pos += size
        return data

    def unpack(self, fmt):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def read_cstring(self):
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
        have = self.available()
        if size <= have:
            self.pos += size
            return
        self.buf = b''
        self.pos = 0
        self.fin.seek(size - have, os.SEEK_CUR)

def fasttext_subword_rows(word, minn, maxn, bucket, nwords, prune_map=None):
    """
    Return the input matrix rows for the character n-grams of a word

    word is the UTF-8 bytes of a dictionary word.  This follows
    Dictionary::computeSubwords in fastText: the n-grams come from the
    word wrapped in < and >, are measured in UTF-8 characters from
    minn to maxn long, and skip the single character n-grams < and >.
    Each n-gram is hashed with 32 bit FNV-1a, using the sign extended
    bytes in FASTTEXT_HASH_BYTES.

    prune_map is the n-gram remapping stored in pruned models.  If it
    is given, n-grams missing from it have no row.
    """
    if maxn <= 0 or bucket <= 0:
        return []
    wrapped = b'<' + word + b'>'
    length = len(wrapped)
    rows = []
    for i in range(length):
        if wrapped[i] & 0xC0 == 0x80:
            # n-grams only start at the beginning of a character
            continue
        ngram_hash = 2166136261
        j = i
        n = 1
        while j < length and n <= maxn:
            # extend the n-gram by one full character
            ngram_hash = ((ngram_hash ^ FASTTEXT_HASH_BYTES[wrapped[j]]) * 16777619) & 0xFFFFFFFF
            j += 1
            while j < length and wrapped[j] & 0xC0 == 0x80:
                ngram_hash = ((ngram_hash ^ FASTTEXT_HASH_BYTES[wrapped[j]]) * 16777619) & 0xFFFFFFFF
                j += 1
            if n >= minn and not (n == 1 and (i == 0 or j == length)):
                ngram_id = ngram_hash % bucket
                if prune_map is None:
                    rows.append(nwords + ngram_id)
                elif ngram_id in prune_map:
                    rows.append(nwords + prune_map[ngram_id])
            n += 1
    return rows

def read_fasttext_binary(filename, max_vocab=None):
    """
    Read word vectors from a fastText .bin model

    fastText's vector for a word is the average of the word's own row
    in the input matrix and the rows for its character n-grams.  The
    vectors returned here match get_word_vector in the fastText
    library, which is also what fastText writes to a .vec file.  Words
    come back in dictionary order, which fastText sorts by frequency.

    The whole dictionary has to be scanned to find the matrix, but
    only the matrix rows used by the kept words are read, so a
    max_vocab limit saves both time and memory.  The n-gram rows for
    every kept word are collected in Python lists, which costs a few
    KB per word, so converting a model with millions of words needs
    tens of GB.  A limit of a few hundred thousand words avoids that.

    Quantized models (.ftz) are not supported.
    """
    logger.info("Reading fastText binary vectors from %s ...", filename)
    limit = max_vocab if max_vocab and max_vocab > 0 else None

    with open_read_binary(filename) as fin:
        reader = ByteReader(fin)
        magic, version = reader.unpack('<ii')
        if magic != FASTTEXT_MAGIC:
            raise ValueError("%s is not a fastText .bin model" % filename)
        if version > FASTTEXT_MAX_VERSION:
            raise ValueError("%s is fastText format version %d, but only versions up to %d are supported" % (filename, version, FASTTEXT_MAX_VERSION))

        # dim, ws, epoch, minCount, neg, wordNgrams, loss, model, bucket, minn, maxn, lrUpdateRate, t
        model_args = reader.unpack('<12id')
        dim, bucket, minn, maxn = model_args[0], model_args[8], model_args[9], model_args[10]

        size, nwords, nlabels, ntokens, pruneidx_size = reader.unpack('<iiiqq')
        logger.info("fastText model has %d words, %d dimensions, n-grams of length %d to %d in %d buckets", nwords, dim, minn, maxn, bucket)

        words = []
        # (dictionary index, UTF-8 bytes) for each kept word
        entries = []
        unk_entry = None
        failed = 0
        for entry_id in range(size):
            if entry_id > 0 and entry_id % 1000000 == 0:
                logger.debug("Read %d of %d dictionary entries", entry_id, size)
            raw_word = reader.read_cstring()
            count, entry_type = reader.unpack('<qb')
            # entry_type 0 is a word, 1 is a supervised label
            if entry_type != 0 or (limit is not None and len(words) >= limit):
                continue
            try:
                word = raw_word.decode('utf-8')
            except UnicodeDecodeError:
                failed += 1
                continue
            if word == '<unk>':
                unk_entry = (entry_id, raw_word)
                continue
            words.append(word.replace(' ', '\xa0'))
            entries.append((entry_id, raw_word))

        prune_map = None
        if pruneidx_size == 0:
            # every n-gram was pruned away
            maxn = 0
        elif pruneidx_size > 0:
            pairs = struct.unpack('<%di' % (2 * pruneidx_size), reader.read(8 * pruneidx_size))
            prune_map = dict(zip(pairs[0::2], pairs[1::2]))

        quantized = reader.unpack('<?')[0]
        if quantized:
            raise ValueError("%s is a quantized fastText model, which is not supported" % filename)
        rows, cols = reader.unpack('<qq')
        if cols != dim:
            raise ValueError("%s has a %d column input matrix, but the model dimension is %d" % (filename, cols, dim))

        # pair each output row with the input rows which are averaged into it
        targets = [(row + len(VOCAB_PREFIX), entry) for row, entry in enumerate(entries)]
        if unk_entry is not None:
            targets.append((UNK_ID, unk_entry))
        output_rows = []
        input_rows = []
        counts = np.ones(len(words) + len(VOCAB_PREFIX), dtype=np.float32)
        for output_row, (entry_id, raw_word) in targets:
            subwords = [entry_id]
            # fastText does not compute n-grams for the end of sentence token
            if raw_word != b'</s>':
                subwords.extend(fasttext_subword_rows(raw_word, minn, maxn, bucket, nwords, prune_map))
            output_rows.extend([output_row] * len(subwords))
            input_rows.extend(subwords)
            counts[output_row] = len(subwords)

        emb = torch.zeros((len(words) + len(VOCAB_PREFIX), dim), dtype=torch.float32)
        if input_rows:
            input_rows = np.array(input_rows, dtype=np.int64)
            output_rows = np.array(output_rows, dtype=np.int64)
            order = np.argsort(input_rows, kind='stable')
            input_rows = input_rows[order]
            output_rows = output_rows[order]
            if input_rows[-1] >= rows:
                raise ValueError("%s needs row %d of the input matrix, but the matrix only has %d rows" % (filename, input_rows[-1], rows))

            # read the matrix forward in blocks, adding each needed row
            # into every output vector which uses it
            row_bytes = 4 * dim
            block_rows = max(1, (1 << 26) // row_bytes)
            pair_batch = max(1, (1 << 26) // row_bytes)
            last_needed = int(input_rows[-1])
            current_row = 0
            start = 0
            while start < len(input_rows):
                first = int(input_rows[start])
                reader.skip((first - current_row) * row_bytes)
                last = min(first + block_rows, last_needed + 1)
                block = np.frombuffer(reader.read((last - first) * row_bytes), dtype='<f4').reshape(last - first, dim)
                end = int(np.searchsorted(input_rows, last, side='left'))
                for batch_start in range(start, end, pair_batch):
                    batch_end = min(batch_start + pair_batch, end)
                    selected = block[input_rows[batch_start:batch_end] - first]
                    emb.index_add_(0, torch.from_numpy(output_rows[batch_start:batch_end]), torch.from_numpy(selected))
                current_row = last
                start = end
        emb /= torch.from_numpy(counts).unsqueeze(1)

    if failed > 0:
        logger.info("Failed to read %d words from embedding", failed)
    return words, emb, failed


def find_pretrain_file(wordvec_pretrain_file, save_dir, shorthand, lang):
    """
    When training a model, look in a few different places for a .pt file

    If a specific argument was passsed in, prefer that location
    Otherwise, check in a few places:
      saved_models/{model}/{shorthand}.pretrain.pt
      saved_models/{model}/{shorthand}_pretrain.pt
      ~/stanza_resources/{language}/pretrain/{shorthand}_pretrain.pt
    """
    if wordvec_pretrain_file:
        return wordvec_pretrain_file

    default_pretrain_file = os.path.join(save_dir, '{}.pretrain.pt'.format(shorthand))
    if os.path.exists(default_pretrain_file):
        logger.debug("Found existing .pt file in %s" % default_pretrain_file)
        return default_pretrain_file
    else:
        logger.debug("Cannot find pretrained vectors in %s" % default_pretrain_file)

    pretrain_file = os.path.join(save_dir, '{}_pretrain.pt'.format(shorthand))
    if os.path.exists(pretrain_file):
        logger.debug("Found existing .pt file in %s" % pretrain_file)
        return pretrain_file
    else:
        logger.debug("Cannot find pretrained vectors in %s" % pretrain_file)

    if shorthand.find("_") >= 0:
        # try to assemble /home/user/stanza_resources/vi/pretrain/vtb.pt for example
        pretrain_file = os.path.join(DEFAULT_MODEL_DIR, lang, 'pretrain', '{}.pt'.format(shorthand.split('_', 1)[1]))
        if os.path.exists(pretrain_file):
            logger.debug("Found existing .pt file in %s" % pretrain_file)
            return pretrain_file
        else:
            logger.debug("Cannot find pretrained vectors in %s" % pretrain_file)

    # if we can't find it anywhere, just return the first location searched...
    # maybe we'll get lucky and the original .txt file can be found
    return default_pretrain_file


if __name__ == '__main__':
    with open('test.txt', 'w') as fout:
        fout.write('3 2\na 1 1\nb -1 -1\nc 0 0\n')
    # 1st load: save to pt file
    pretrain = Pretrain('test.pt', 'test.txt')
    print(pretrain.emb)
    # verify pt file
    x = torch.load('test.pt', weights_only=True)
    print(x)
    # 2nd load: load saved pt file
    pretrain = Pretrain('test.pt', 'test.txt')
    print(pretrain.emb)

