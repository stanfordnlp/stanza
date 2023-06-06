import argparse
import glob

import numpy as np
import torch
from PIL import Image as im
import torchshow as ts

from stanza.models.constituency.trainer import Trainer

def to_image(param, bound):
    return param

parser = argparse.ArgumentParser()
parser.add_argument('--tensor_name', type=str, default="word_lstm.weight_ih_l0", help='Which tensor to visualize')
parser.add_argument('paths', type=str, nargs='+', help='Paths to read for the visualization')
args = parser.parse_args()

all_tensors = []
for path in args.paths:
    filenames = sorted(glob.glob(path))
    tensors = []
    for filename in filenames:
        print(filename)
        trainer = Trainer.load(filename)
        model = trainer.model
        param = model.get_parameter(args.tensor_name)
        param = param.detach().numpy()
        tensors.append(param)
    print(len(tensors), tensors[0].shape)
    all_tensors.append(tensors)

min_value = min(min(x.min() for x in tensors) for tensors in all_tensors)
max_value = max(max(x.max() for x in tensors) for tensors in all_tensors)

max_abs = max(-min_value, max_value)
print(min_value, max_value, max_abs)

for image_idx, tensors in enumerate(zip(*all_tensors)):
    all_red = []
    all_green = []
    all_blue = []
    for idx, tensor in enumerate(tensors):
        if idx > 0:
            grey = np.ones((50, tensor.shape[1]), tensor.dtype) / 2.0
            all_red.append(grey)
            all_green.append(grey)
            all_blue.append(np.zeros_like(grey))
        tensor = tensor / max_abs
        red = -tensor.clip(-1.0, 0)
        green = tensor.clip(0, 1.0)
        blue = np.zeros_like(tensor)
        print(blue.min(), blue.max())
        all_red.append(red)
        all_green.append(green)
        all_blue.append(blue)
    all_red =   np.concatenate(all_red, axis=0)
    all_green = np.concatenate(all_green, axis=0)
    all_blue =  np.concatenate(all_blue, axis=0)
    image = np.stack([all_red, all_green, all_blue], axis=2)
    print(image.shape)
    print(image[:, :, 0].min(), image[:, :, 0].max())
    print(image[:, :, 1].min(), image[:, :, 1].max())
    print(image[:, :, 2].min(), image[:, :, 2].max())
    image = im.fromarray(image, mode="RGB")
    image.save(args.tensor_name + ".%d.png" % image_idx)

#ts.save(all_tensors, args.tensor_name + ".png", nrows=2048, ncols=3292, figsize=(100, 100), dpi=200)
