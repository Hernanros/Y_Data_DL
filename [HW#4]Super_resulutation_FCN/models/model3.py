import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#augmentation library
import albumentations as A

class third_model(nn.Module):
  
  def __init__(self,in_channels = 3, out_channels = 3, num_filters = 32, kernel_size = 3, stride = 1 , padding = 1, keep_probab = 0.4):
    '''
    input : tensor of dimensions (batch_size*72*72*3)
    output: 2 tensor of dimension 1.(batchsize*144*144*3) 2. (batchsize*288*288*3)

    1. 2d conv with 32 filters, kernel size 1*1; input(batch_size*72*72*3), output (batch_size*72*72*32)
    2. Residual block:
            2.1 2d conv with 32 filters, kernel size 3*3; input(batch_size*72*72*32), output(batch_size*72*72*32)
            2.2 2d conv with 32 filters, kernel size 3*3; input(batch_size*72*72*32), output(batch_size*72*72*32)
            2.3 Add original x with convolutions, dimensionality doesn't change
            2.4 Relu activation
    3. Residual block (see 2)
    4. Upsampling 2d; input (batchsize*72*72*32), output (batchsize*144*144*32)

    5a. 2d conv with 3 filters (RGB), kernel size 1*1; input (batch_size*144*144*32), output (batchsize*144*144*3) - store this output as x_1
    
    5b. Residual block (see 2)
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


    self.double_conv = nn.Sequential(
                     nn.Conv2d(self.num_filters,self.num_filters,self.kernel_size,self.stride,self.padding),
                     nn.Conv2d(self.num_filters,self.num_filters,self.kernel_size,self.stride,self.padding)
    )
    self.initial_conv = nn.Conv2d(self.in_channels, self.num_filters,1,self.stride,padding=0)
    self.upsampling_mid = nn.Upsample((144,144), mode= 'bilinear')
    self.upsampling_large = nn.Upsample((288,288), mode = "bilinear")
    self.pool = nn.MaxPool2d(2, 2)
    self.label = nn.Conv2d(self.num_filters, self.out_channels, 1, self.stride,padding=0)

  def residual_block(self,x):
    res = x
    x = self.double_conv(x)
    x = res + x
    return F.relu(x)

  def forward(self, x):
    
    x = x.permute(0, 3, 1, 2).contiguous()
    x = self.initial_conv(x)
    x = self.residual_block(x)
    x = self.residual_block(x)
    x = self.upsampling_mid(x)
    x_1 = self.label(x)
    x_2 = self.residual_block(x)
    x_2 = self.upsampling_large(x_2)
    x_2 = self.label(x_2)

    return x_1.permute(0,2,3,1), x_2.permute(0,2,3,1)