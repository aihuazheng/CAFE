import argparse
import torch
import imageio
import skimage.transform
import torchvision
import numpy as np
import torch.optim
import cafe_model
from utils import utils
from utils.utils import load_ckpt
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm


data_path=""
data_path1=""


def decode_segmap( temp, plot=False):
    Imps = [255, 255, 255]
    Building = [0, 0, 255]
    Lowvg = [0, 255, 255]
    Tree = [0, 255, 0]
    Car = [255, 255, 0]
    bg = [255, 0, 0]

    label_colours = np.array(
        [
            Imps,
            Building,
            Lowvg,
            Tree,
            Car,
            bg,
        ]
    )
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 6):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]
    # rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

rgblist=[]
dsmlist=[]
rgb=os.listdir(data_path)
dsm=os.listdir(data_path1)
for i in rgb:
    portion=os.path.splitext(i)
    if portion[1]==".tif":
        rgblist.append(portion[0])
for j in dsm:
    portion=os.path.splitext(j)
    if portion[1]==".tif":
        dsmlist.append(portion[0])



device=torch.device("cpu" if  torch.cuda.is_available() else "cpu")
image_w=512
image_h=512
model=cafe_model.CaFE(pretrained=False)
model.eval()
model.to(device)
checkpoint=torch.load("",map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])

for i in tqdm(range()):
    path1=""
    path2=""
    image=plt.imread(path1)
    depth=plt.imread(path2)
    image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                     mode='reflect', preserve_range=True)
    # Nearest-neighbor
    depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                     mode='reflect', preserve_range=True)

    image = image / 255
    image = torch.from_numpy(image).float()
    depth = torch.from_numpy(depth).float()
    image = image.permute(2, 0, 1)
    depth.unsqueeze_(0)

    image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(image)
    depth = torchvision.transforms.Normalize(mean=[19050],
                                             std=[9650])(depth)

    image = image.to(device).unsqueeze_(0)
    depth = depth.to(device).unsqueeze_(0)
    predicts = model(image, depth)
    predicts=predicts.squeeze()
    predicts = F.softmax(predicts, dim=0)
    predicts = predicts.argmax(dim=0)
    predicts = predicts.numpy()
    predicts = decode_segmap(predicts)
    path3=""
    plt.imsave(path3,predicts)


