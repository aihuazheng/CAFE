#The label map in the code is changed to the label map of its own network,
a#nd the number of categories is six

import torch

label_map = torch.randint(low=0, high=num_classes, size=(batch_size, height, width))
class_counts = torch.histc(label_map.float(), bins=num_classes, min=0, max=num_classes-1)
class_ratios = class_counts / torch.sum(class_counts)

