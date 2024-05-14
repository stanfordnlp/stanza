import argparse
import logging
import os 
import torch
import sys
import torch.nn as nn
import torch.optim as optim

# To add Stanza modules, TODO remove this and just EXPORT this to the sys path manually before running
ROOT = '/Users/alexshan/Desktop/stanza'
sys.path.append(ROOT)

from stanza.models.common.utils import default_device
from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.summarization.constants import * 
from stanza.models.summarization.src.model import *
from stanza.utils.get_tqdm import get_tqdm

from typing import List, Tuple, Any, Mapping

torch.set_printoptions(threshold=100, edgeitems=5, linewidth=100)
logger = logging.getLogger('stanza.summarization')  # TODO: update these with Stanza-specific logging modules
logger.propagate = False

# Check if the logger has handlers already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

tqdm = get_tqdm()


def main():
    logger.info("This is a test")


if __name__ == "__main__":
    main()