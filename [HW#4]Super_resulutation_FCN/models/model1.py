import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#augmentation library
import albumentations as A

class firstmodel(nn.Module):
  
  def __init__(self,in_channels = 3, out_channels = 3, num_filters = 64, kernel_size = 3, stride = 1 , padding = 1, keep_probab = 0.4):
    '''
    input : tensor of dimensions (batch_size*72*72*3)
    output: tensor of dimension (batchsize*144*144*3)
    4 blocks:
      1. 2d conv with 64 filters, kernel size 3*3 (one padding=1); input  (batch_size*72*72*3), output (batchsize*72*72*64)
      2. 2d conv with 64 filters, kernel size 3*3 (one padding=1);  (batch_size*72*72*64) output (batchsize*72*72*64)
      3. upsampling 2d; input (batchsize*72*72*64), output (batchsize*144*144*64)
      4. 2d conv with 3 filters (RGB), kernel size 1*1; input (batch_size*144*144*64), output (batchsize*144*144*3)

    '''
    super().__init__()
  
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = int(padding)
    self.keep_probab= keep_probab
      
    self.conv1 = nn.Conv2d(self.in_channels, self.num_filters, self.kernel_size, self.stride, self.padding )
    self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, self.stride, self.padding )
    self.upsampling = nn.Upsample((144,144), mode= 'bilinear')
    self.dropout = nn.Dropout(self.keep_probab)
    self.pool = nn.MaxPool2d(2, 2)
    self.label = nn.Conv2d(self.num_filters, self.out_channels, 1, self.stride)
    self.norm = nn.BatchNorm2d(self.num_filters)


  def forward(self, x):
    x= x.permute(0, 3, 1, 2)

    ### TRY WITHOUT NORM ###
    # x = self.pool(F.relu(self.norm(self.conv1(x))))
    # x = self.pool(F.relu(self.norm(self.conv2(x))))

    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))

    x = self.upsampling(x)
    x = self.label(x)
    return x.permute(0,2,3,1)