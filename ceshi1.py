import numpy as np
from torch import nn
import torch
import os

label_colours=[(255,255,255),(0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0)]

def color_label(label):
    label=label.clone().cpu().data.numpy()
    colored_label=np.vectorize(lambda x: label_colours[int(x)])

    colored=np.asarray(colored_label(label)).astype((np.float32))
    colored=colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])

colored_label=np.vectorize(lambda x: label_colours[int(x)])
print(type(colored_label))
print(colored_label)