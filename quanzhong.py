import torch
from torch import nn
import math
import torch.utils.model_zoo as model_zoo
from utils import utils
from torch.utils.checkpoint import checkpoint
import numpy as np

pretrain_dict = model_zoo.load_url(utils.model_urls['resnet50'],"./")
pretrain_dict1=model_zoo.load_url(utils.model_urls['resnet18'],"./")
pretrain_dict2=model_zoo.load_url(utils.model_urls['resnet34'],"./")
pretrain_dict3=model_zoo.load_url(utils.model_urls['resnet101'],"./")
pretrain_dict4=model_zoo.load_url(utils.model_urls['resnet152'],"./")
print(type(pretrain_dict))