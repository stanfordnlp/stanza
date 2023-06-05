"""
Lists all the tensors in a constituency model.

Currently useful in combination with torchshow for displaying a series of tensors as they change.
"""

import sys

from stanza.models.constituency.trainer import Trainer


trainer = Trainer.load(sys.argv[1])
model = trainer.model

for name, param in model.named_parameters():
    print(name, param.requires_grad)
