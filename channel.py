#This is the channel attention module of the network during training,
#which is retained during model training and testing.
import torch
from torch import nn

def efficientchannelattention(x,gamma=2,b=1):
    N,C,H,W=x.size()
    t=int(abs((log(C,2)+b)/gamma))
    k=t if t%2 else t+1

    avg_pool=nn.AdaptivaAvgPool2d(i)
    conv=nn.Conv1d(1,1,kernel_size=k,padding=int(k/2),bias=False)
    sigmoid=nn.Sigmoid()

    y=avg_pool(x)
    y=conv(y.squeeze(-1).transpose(-1,-2))
    y=y.transpose(-1,-2).unsqueeze(-1)
    y=sigmoid(y)

    return x*y.expand_as(x)