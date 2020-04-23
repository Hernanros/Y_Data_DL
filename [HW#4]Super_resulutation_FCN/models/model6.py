import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16

#augmentation library
import albumentations as A

class sixth_model(nn.Module):
  
  def __init__(self,in_channels = 3, out_channels = 3, num_filters = 64, kernel_size = 3, stride = 1 , padding = 1, keep_probab = 0.4):
    '''
    input : tensor of dimensions (batch_size*72*72*3)
    output: 2 tensor of dimension 1.(batchsize*144*144*3) 2. (batchsize*288*288*3)

    1. 2d conv with 32 filters, kernel size 1*1; input(batch_size*72*72*3), output (batch_size*72*72*32)

    2. Dilation block:
           2.1 2d dilated conv with 32 filters, kernel size 3*3 (dilation = 1, i.o.w same density); input(batch_size*72*72*32), output(batch_size*72*72*32)
           2.2 2d dilated conv with 32 filters, kernel size 3*3 (dilation = 2, i.o.w 5*5), add padding=2; input(batch_size*72*72*32), output(batch_size*72*72*32)
           2.3 2d dilated conv with 32 filters, kernel size 3*3 (dilation = 4, i.o.w 9*9), add padding=4; input(batch_size*72*72*32), output(batch_size*72*72*32)
           2.4 Concatenate 2.1,2.2,2.3 into (batch_size*72*72*96)
           2.5 Relu activation
           2.6 2d conv with 32 filter, kernel size 3*3; input (batch_size*72*72*96), output (batch_size*72*72*32)

    3. Dilation block (see 2)
    4. Upsampling 2d; input (batchsize*72*72*32), output (batchsize*144*144*32)

    5a. 2d conv with 3 filters (RGB), kernel size 1*1; input (batch_size*144*144*32), output (batchsize*144*144*3) - store this output as x_1
    
    5b. Dilation block (see 2)
        5b1. Upsampling 2d; input (batchsize*144*144*32), output (batchsize*288*288*32)
        5b2. 2d conv with 3 filters (RGB), kernel size 1*1; input (batch_size*288*288*32), output (batchsize*288*288*3) - store this output as x_2
    '''
    super().__init__()
  
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = int(padding)
    self.keep_probab= keep_probab
    
    self.pre_vgg = nn.Conv2d(self.in_channels,self.num_filters,1,1,0)
    self.vgg_layer = self.vgg16_layer()
    self.conv_layer = nn.Sequential(
                      nn.Conv2d(self.in_channels,self.num_filters,self.kernel_size,self.stride,self.padding),
                      nn.Conv2d(self.num_filters,self.num_filters,self.kernel_size,self.stride,self.padding)
    )
    
    self.label_1 = nn.Conv2d(self.num_filters//2, self.out_channels, 1, self.stride,padding=0)
    self.pre_shuffle2 = nn.Conv2d(self.out_channels,self.num_filters//4,1,1,0)
    self.label_2 = nn.Conv2d(self.num_filters//16, self.out_channels, 1, self.stride,padding=0) 
    self.pool = nn.MaxPool2d(2, 2)
    

  def vgg16_layer(self):
    layer = vgg16(pretrained=True).features[2]
    for param in layer.parameters():
      param.requires_grad = False
    return layer

  def forward(self, x):
    
    x = x.permute(0, 3, 1, 2).contiguous()

    x_1 = self.pre_vgg(x)
    x_1 = self.vgg_layer(x_1)
    x_2 = self.conv_layer(x)
    x = torch.cat((x_1,x_2),1)
    x = F.pixel_shuffle(x,2)
    x_1 = self.label_1(x)
    x_2 = self.pre_shuffle2(x_1)
    x_2 = F.pixel_shuffle(x_2,2)
    x_2 = self.label_2(x_2)

    return x_1.permute(0,2,3,1), x_2.permute(0,2,3,1)