"""
Shared util methods specific to the dependency parser
"""

import torch

from stanza.models.common import utils

def predict_dataset(trainer, dev_batch):
    with torch.no_grad(), utils.evaluating(trainer.model):
        dev_preds = []
        if len(dev_batch) > 0:
            for batch in dev_batch:
                preds = trainer.predict(batch)
                dev_preds += preds
            dev_preds = utils.unsort(dev_preds, dev_batch.data_orig_idx)
    return dev_preds

