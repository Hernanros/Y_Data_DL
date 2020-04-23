import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#augmentation library
import albumentations as A

class fourth_model(nn.Module):
  
  def __init__(self,in_channels = 3, out_channels = 3, num_filters = 32, kernel_size = 3, stride = 1 , padding = 1, keep_probab = 0.4):
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
    
    self.initial_conv = nn.Conv2d(self.in_channels, self.num_filters,1,self.stride,padding=0)

    self.dil1 = nn.Conv2d(self.num_filters,self.num_filters,self.kernel_size,self.stride,self.padding,dilation = 1)
    self.dil2 = nn.Conv2d(self.num_filters,self.num_filters,self.kernel_size,self.stride,self.padding*2,dilation = 2)
    self.dil4 = nn.Conv2d(self.num_filters,self.num_filters,self.kernel_size,self.stride,self.padding*4,dilation = 4)

    self.concatConv = nn.Conv2d(self.num_filters*3,self.num_filters,self.kernel_size,self.stride,self.padding)

    self.upsampling_mid = nn.Upsample((144,144), mode= 'bilinear')
    self.upsampling_large = nn.Upsample((288,288), mode = "bilinear")
    self.pool = nn.MaxPool2d(2, 2)
    self.label = nn.Conv2d(self.num_filters, self.out_channels, 1, self.stride,padding=0)

  def dilation_block(self,x):
    d1 = self.dil1(x)
    d2 = self.dil2(x)
    d4 = self.dil4(x)
    x = torch.cat((d1,d2,d4), 1)
    x = F.relu(x)
    return self.concatConv(x)

  def forward(self, x):
    
    x = x.permute(0, 3, 1, 2).contiguous()

    x = self.initial_conv(x)
    x = self.dilation_block(x)
    x = self.dilation_block(x)
    x = self.upsampling_mid(x)
    x_1 = self.label(x)
    x_2 = self.dilation_block(x)
    x_2 = self.upsampling_large(x_2)
    x_2 = self.label(x_2)

    return x_1.permute(0,2,3,1), x_2.permute(0,2,3,1)