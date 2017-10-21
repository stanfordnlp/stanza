import os
import sys

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SOURCE_DIR, 'data')

if not os.path.isdir(DATA_DIR):
    try:
        os.makedirs(DATA_DIR)
    except Exception as e:
        sys.stderr.write("Could not create data directory at {}.\n{}".format(DATA_DIR, e))