import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
class SegmentationMetric(object):
    def __init__(self,numclass):
        self.numclass=numclass
        self.confusionmatrix=np.zeros((self.numclass,)*2)

    def pixelaccuracy(self):
        acc=np.diag(self.confusionmatrix).sum() / self.confusionmatrix.sum()
        return acc

    def meanpixelaccuracy(self):
        classacc=np.diag(self.confusionmatrix) / self.confusionmatrix.sum(axis=0)
        return classacc


    def miou(self):
        intersection=np.diag(self.confusionmatrix)
        union=np.sum(self.confusionmatrix,axis=1) + np.sum(self.confusionmatrix,axis=0) - np.diag(self.confusionmatrix)
        iou=intersection / union
        miou=np.nanmean(iou)
        return miou


    def getconfusionmartix(self,imgpredict,imglabel):
        mask=(imglabel>=0)&(imglabel<self.numclass)
        label=self.numclass*imglabel[mask]+imgpredict[mask]
        count=np.bincount(label,minlength=self.numclass**2)
        confusionmartix=count.reshape(self.numclass,self.numclass)
        return confusionmartix

    def addbacth(self,imgpredict,imglabel):
        # imgpredcit=imgpredict.cpu()
        # imglabel=imglabel.cpu()
        # assert imgpredict.shape==imglabel.shape
        self.confusionmatrix+=self.getconfusionmartix(imgpredict,imglabel)

    def reset(self):
        self.confusionmatrix=np.zeros((self.numclass,self.numclass))

    def decode_segmap(self, temp, plot=False):
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
        for l in range(0, self.n_classes):
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



#






