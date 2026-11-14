"""
Comments recording which version of an external lexicon a lemma dictionary was built from

Each converter writes these comments at the start of the first sentence
of its dictionary file, such as

  # specialist_sha256 = 3f2a...
  # specialist_modified = 2026-01-15

The lemmatizer copies the comments at the start of its --secondary_file
into the model's config as secondary_file_source, so a trained model
keeps the record even after the dictionary file is rebuilt.
"""

import datetime
import hashlib
import os

def files_sha256(filenames):
    """
    A sha256 of the basenames and contents of filenames, in the order given
    """
    digest = hashlib.sha256()
    for filename in filenames:
        digest.update(os.path.basename(filename).encode('utf-8'))
        with open(filename, 'rb') as fin:
            for block in iter(lambda: fin.read(1 << 20), b''):
                digest.update(block)
    return digest.hexdigest()

def modified_date(filenames):
    """
    The latest modification date of filenames as YYYY-MM-DD in UTC, or None if there are no files

    For a downloaded file, this is the server's date for the file if the
    download kept it (as wget does by default), and otherwise the date
    it was downloaded.
    """
    if not filenames:
        return None
    latest = max(os.path.getmtime(filename) for filename in filenames)
    return datetime.datetime.fromtimestamp(latest, datetime.timezone.utc).date().isoformat()

def file_comments(prefix, filenames):
    """
    Comments with the sha256 and modification date of the files a dictionary was built from
    """
    comments = ["# %s_sha256 = %s" % (prefix, files_sha256(filenames))]
    date = modified_date(filenames)
    if date is not None:
        comments.append("# %s_modified = %s" % (prefix, date))
    return comments
