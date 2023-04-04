#In the process of category-level feature fusion,
#the label map is separated by category,
#that is, the process of converting into a one-hot vector.
#Here change the label map input to the label map of your own network

import torch

label_map = torch.randint(low=0, high=num_classes, size=(batch_size, height, width))

one_hot = torch.nn.functional.one_hot(label_map, num_classes=num_classes)

print(one_hot.shape)  # 输出 (batch_size, num_classes, height, width)
