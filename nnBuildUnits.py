
'''
    * Building units for neural networks: conv23D units, residual units, unet units, upsampling unit and so on.
    * all kinds of loss functions: softmax, 2d softmax, 3d softmax, dice, multi-organ dice, focal loss, attention based loss...
    * kinds of test units
    * First implemented in Dec. 2016, and the latest updation is Dec. 2017.
    * Dong Nie
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function
import SimpleITK as sitk
from itertools import repeat



'''
    To have an easy switch between nn.Conv2d and nn.Conv3d
'''
class conv23DUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(conv23DUnit, self).__init__()
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd==2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
        elif nd==3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
       
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        
    def forward(self, x):
        return self.conv(x)

'''
    To have an easy switch between nn.Conv2d and nn.Conv3d, together with BN
'''
class conv23D_bn_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(conv23D_bn_Unit, self).__init__()
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd==2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        elif nd==3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)
       
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
#         self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.bn(self.conv(x))


'''
    To have an easy switch between nn.Conv2d and nn.Conv3d, together with GN
'''
class conv23D_gn_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, dilation=1,
                 nd=2):
        super(conv23D_gn_Unit, self).__init__()

        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                                  bias=bias, dilation=dilation)
        elif nd == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                                  bias=bias, dilation=dilation)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                                  bias=bias, dilation=dilation)

        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.gn = nn.GroupNorm(out_channels//8, out_channels)

    #         self.relu = nn.ReLU()

    def forward(self, x):
        return self.gn(self.conv(x))


'''
    To have an easy switch between nn.Conv2d and nn.Conv3d, together with BN and relu
'''
class conv23D_bn_relu_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(conv23D_bn_relu_Unit, self).__init__()
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd==2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        elif nd==3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)
       
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.relu = nn.ReLU()
        
    def forward(self, x):

        return self.relu(self.bn(self.conv(x)))


'''
    To have an easy switch between nn.Conv2d and nn.Conv3d, together with GN and relu
'''
class conv23D_gn_relu_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, dilation=1,
                 nd=2):
        super(conv23D_gn_relu_Unit, self).__init__()

        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                                  bias=bias, dilation=dilation)
        elif nd == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                                  bias=bias, dilation=dilation)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
                                  bias=bias, dilation=dilation)

        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.gn = nn.GroupNorm(out_channels//8, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.relu(self.gn(self.conv(x)))



class batchNorm23D(nn.Module):
    def __init__(self, chn, nd=2):
        super(batchNorm23D, self).__init__()
        assert nd==1 or nd==2 or nd==3, 'ERROR: NDim should be 1, 2 or 3'
        if nd==2:
            self.bn = nn.BatchNorm2d(chn)
        elif nd==1:
            self.bn = nn.BatchNorm1d(chn)
        else:
            self.bn = nn.BatchNorm3d(chn)

    def forward(self, x):
        return self.bn(x)

'''
    To have an easy switch between nn.ConvTranspose2d and nn.ConvTranspose3d
'''
class convTranspose23DUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(convTranspose23DUnit, self).__init__()
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd==2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        elif nd==3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
       
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        
    def forward(self, x):
        return self.conv(x)

'''
    To have an easy switch between nn.ConvTranspose2d and nn.ConvTranspose3d, together with BN
'''
class convTranspose23D_bn_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(convTranspose23D_bn_Unit, self).__init__()
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd==2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        elif nd==3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)
       
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
#         self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.bn(self.conv(x))


'''
    To have an easy switch between nn.ConvTranspose2d and nn.ConvTranspose3d, together with GN
'''
class convTranspose23D_gn_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, nd=2):
        super(convTranspose23D_gn_Unit, self).__init__()

        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        elif nd == 3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)

        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.gn = nn.GroupNorm(out_channels//8, out_channels)

    #         self.relu = nn.ReLU()

    def forward(self, x):
        return self.gn(self.conv(x))

'''
    To have an easy switch between nn.ConvTranspose2d and nn.ConvTranspose3d, together with BN and relu
'''
class convTranspose23D_bn_relu_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, nd=2):
        super(convTranspose23D_bn_relu_Unit, self).__init__()
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd==2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        elif nd==3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)
       
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


'''
    To have an easy switch between nn.ConvTranspose2d and nn.ConvTranspose3d, together with GN and relu
'''
class convTranspose23D_gn_relu_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, nd=2):
        super(convTranspose23D_gn_relu_Unit, self).__init__()

        assert nd == 1 or nd == 2 or nd == 3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd == 2:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        elif nd == 3:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)

        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.gn = nn.GroupNorm(out_channels // 8, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


'''
    To have an easy switch between nn.Dropout2d and nn.Dropout3d
'''
class dropout23DUnit(nn.Module):
    def __init__(self, prob=0, nd=2):
        super(dropout23DUnit, self).__init__()
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd==2:
            self.dp = nn.Dropout2d(p=prob)
        elif nd==3:
            self.dp = nn.Dropout3d(p=prob)
        else:
            self.dp = nn.Dropout(p=prob)   
    def forward(self, x):
        return self.dp(x)

'''
    To have an easy switch between nn.maxPool2D and nn.maxPool3D
'''
class maxPool23DUinit(nn.Module):
    def __init__(self, kernel_size, stride, padding=1, dilation=1, nd=2):
        super(maxPool23DUinit, self).__init__()
        
        assert nd==1 or nd==2 or nd==3, 'nd is not correctly specified!!!!, it should be {1,2,3}'
        if nd==2:
            self.pool1 = nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding, dilation=dilation)
        elif nd==3:
            self.pool1 = nn.MaxPool3d(kernel_size=kernel_size,stride=stride,padding=padding, dilation=dilation)
        else:
            self.pool1 = nn.MaxPool1d(kernel_size=kernel_size,stride=stride,padding=padding, dilation=dilation)
    def forward(self, x):
        return self.pool1(x)
        

'''
ordinary conv block
'''
class convUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu, how2norm='BN'):
        super(convUnit, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride, padding)
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        if how2norm=='GN':
            self.bn = nn.GroupNorm(out_size//8, out_size)
        else:
            self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
'''    
 two-layer residual unit: two conv without BN and identity mapping
'''
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu, nd=2):
        super(residualUnit, self).__init__()
        self.conv1 = conv23DUnit(in_size, out_size, kernel_size, stride, padding, nd=nd)

        self.conv2 = conv23DUnit(out_size, out_size, kernel_size, stride, padding, nd=nd)


    def forward(self, x):
        return F.relu(self.conv2(F.elu(self.conv1(x))) + x)
    
'''
    two-layer residual unit: two conv with BN and identity mapping
'''    
class residualUnit1(nn.Module): 
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, how2norm='BN', activation=F.relu, nd=2):
        super(residualUnit1, self).__init__()
        if how2norm=='GN':
            self.conv1_bn_relu = conv23D_gn_relu_Unit(in_size, out_size, kernel_size, stride, padding, nd=nd)
            self.conv2_bn_relu = nn.conv23D_gn_relu_Unit(out_size, out_size, kernel_size, stride, padding, nd=nd)
        else:
            self.conv1_bn_relu = conv23D_bn_relu_Unit(in_size, out_size, kernel_size, stride, padding, nd=nd)
            self.conv2_bn_relu = nn.conv23D_bn_relu_Unit(out_size, out_size, kernel_size, stride, padding, nd=nd)

        self.relu = nn.ReLU()
        
    def forward(self, x): 
        identity_data = x
        output = self.conv1_bn_relu(x)
        output = self.conv2_bn_relu(output)
#         output = self.relu(self.bn1(self.conv1(x)))
#         output = self.bn2(self.conv2(output))
        output = torch.add(output,identity_data)
        output = self.relu(output)
        return output 

'''
three-layer residual unit: three conv with BN and identity mapping
this one doesn't change the size of channels, which means the in_size is same with out_size

input: 
    x
output:
    bottleneck residual block
By Dong Nie
'''
class residualUnit3(nn.Module): 
    def __init__(self, in_size, out_size, isDilation=None, isEmptyBranch1=None, how2norm='BN', activation=F.relu, nd=2):
        super(residualUnit3, self).__init__()
        # mid_size = in_size/2
        mid_size = out_size/2 ###I think it should better be half the out size instead of the input size
#         print 'line 74, in and out size are, ',in_size,' ',mid_size

        if how2norm=='GN':
            if isDilation:
                self.conv1_bn_relu = conv23D_gn_relu_Unit(in_channels=in_size, out_channels=mid_size, kernel_size=1, stride=1, padding=0, dilation=2, nd=nd)
            else:
                self.conv1_bn_relu = conv23D_gn_relu_Unit(in_channels=in_size, out_channels=mid_size, kernel_size=1, stride=1, padding=0, nd=nd)

            if isDilation:
                self.conv2_bn_relu = conv23D_gn_relu_Unit(in_channels=mid_size, out_channels=mid_size, kernel_size=3, stride=1, padding=2, dilation=2, nd=nd)
            else:
                self.conv2_bn_relu = conv23D_gn_relu_Unit(in_channels=mid_size, out_channels=mid_size, kernel_size=3, stride=1, padding=1, nd=nd)

            if isDilation:
                self.conv3_bn = conv23D_gn_Unit(in_channels=mid_size, out_channels=out_size, kernel_size=1, stride=1, padding=0, dilation=2, nd=nd)
            else:
                self.conv3_bn = conv23D_gn_Unit(in_channels=mid_size, out_channels=out_size, kernel_size=1, stride=1, padding=0, nd=nd)

            self.isEmptyBranch1 = isEmptyBranch1
            if in_size!=out_size or isEmptyBranch1==False:
                if isDilation:
                    self.convX_bn = conv23D_gn_Unit(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1, padding=0, dilation=2, nd=nd)
                else:
                    self.convX_bn = conv23D_gn_Unit(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1, padding=0, nd=nd)
    #             self.bnX = nn.BatchNorm2d(out_size)
        else:
            if isDilation:
                self.conv1_bn_relu = conv23D_bn_relu_Unit(in_channels=in_size, out_channels=mid_size, kernel_size=1,
                                                          stride=1, padding=0, dilation=2, nd=nd)
            else:
                self.conv1_bn_relu = conv23D_bn_relu_Unit(in_channels=in_size, out_channels=mid_size, kernel_size=1,
                                                          stride=1, padding=0, nd=nd)

            if isDilation:
                self.conv2_bn_relu = conv23D_bn_relu_Unit(in_channels=mid_size, out_channels=mid_size, kernel_size=3,
                                                          stride=1, padding=2, dilation=2, nd=nd)
            else:
                self.conv2_bn_relu = conv23D_bn_relu_Unit(in_channels=mid_size, out_channels=mid_size, kernel_size=3,
                                                          stride=1, padding=1, nd=nd)

            if isDilation:
                self.conv3_bn = conv23D_bn_Unit(in_channels=mid_size, out_channels=out_size, kernel_size=1, stride=1,
                                                padding=0, dilation=2, nd=nd)
            else:
                self.conv3_bn = conv23D_bn_Unit(in_channels=mid_size, out_channels=out_size, kernel_size=1, stride=1,
                                                padding=0, nd=nd)

            self.isEmptyBranch1 = isEmptyBranch1
            if in_size != out_size or isEmptyBranch1 == False:
                if isDilation:
                    self.convX_bn = conv23D_bn_Unit(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1,
                                                    padding=0, dilation=2, nd=nd)
                else:
                    self.convX_bn = conv23D_bn_Unit(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1,
                                                    padding=0, nd=nd)
            #             self.bnX = nn.BatchNorm2d(out_size)
            self.relu = nn.ReLU()

        self.relu = nn.ReLU()
        
    def forward(self, x): 
        identity_data = x
        
        output = self.conv1_bn_relu(x)
        output = self.conv2_bn_relu(output)
        output = self.conv3_bn(output)
        
        
        outSZ = output.size()
        idSZ = identity_data.size()
        if outSZ[1]!=idSZ[1] or self.isEmptyBranch1==False:
            identity_data = self.convX_bn(identity_data)
#             identity_data = self.bnX(self.convX(identity_data))
#         print output.size(), identity_data.size()
        output = torch.add(output,identity_data)
        output = self.relu(output)
        return output 

    
'''
 long-term residual unit, there is a long way (a lot of convs) before the residual addition
 
 By Dong Nie
'''
class longResidualUnit(nn.Module): 
    def __init__(self,in_size, out_size, kernel_size=3,stride=1, padding=1, how2norm='BN', activation=F.relu, nd=2):
        super(longResidualUnit, self).__init__()
        if how2norm=='GN':
            self.conv1_bn = conv23D_gn_Unit(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, stride=stride, padding=padding, nd=nd)
        else:
            self.conv1_bn = conv23D_bn_Unit(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, stride=stride, padding=padding, nd=nd)

#         init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2.0)) #or gain=1
#         init.constant(self.conv1.bias, 0)
#         self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x): 
        identity_data = x
        output = self.conv1_bn(x)
        output = torch.add(output,identity_data)
        output = self.relu(output)
        return output 


'''
    Residual upsampling block with long-range residual connection
input:
    x: the current layer you want to consider
    bridge: the one you want to combine (using summary instead of concatenation) from lower layers
output:
    residual output
    space_dropout_rate: the rate to make several of the feature maps to be zero (to avoid the correlation within a feature map, we use 
    spatial dropout instead of traditional dropout
    By Dong Nie
'''
class ResUpUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, spatial_dropout_rate=0, how2norm='BN',isConvDilation=None, nd=2):
        super(ResUpUnit, self).__init__()
        self.nd = nd
        if how2norm=='GN':
            self.up = convTranspose23D_gn_relu_Unit(in_size, out_size, kernel_size=4, stride=2, padding=1, nd=nd)
        else:
            self.up = convTranspose23D_bn_relu_Unit(in_size, out_size, kernel_size=4, stride=2, padding=1, nd=nd)

        self.conv = residualUnit3(out_size, out_size, isDilation=isConvDilation, how2norm=how2norm, nd=nd)
        self.dp = dropout23DUnit(prob=spatial_dropout_rate,nd=nd)
        self.spatial_dropout_rate = spatial_dropout_rate
        self.conv2 = residualUnit3(out_size, out_size, isDilation=isConvDilation, isEmptyBranch1=False, how2norm=how2norm, nd=nd)
        self.relu = nn.ReLU()

    def center_crop(self, layer, target_size): #we should make it adust to 2d/3d
        if self.nd ==2:
            batch_size, n_channels, layer_width, layer_height = layer.size()
        elif self.nd==3:
            batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        
        if self.nd==3:
            return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size),  xy1:(xy1 + target_size)]
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):#bridge is the corresponding lower layer
#         print 'x.shape: ', x.size()
        up = self.up(x)

        crop1 = bridge
#         crop1_dp = self.SpatialDroput(crop1)
        if self.spatial_dropout_rate>0:
            crop1 = self.dp(crop1)
        out = self.relu(torch.add(up, crop1))
        out = self.conv(out)

        out = self.conv2(out)
        return out


'''
    long-range residual connection, implement by element-wise addition
input:
    x: the current layer you want to consider
    bridge: the one you want to combine (using summary instead of concatenation) from lower layers, which has been upsampled
output:
    residual output
    space_dropout_rate: the rate to make several of the feature maps to be zero (to avoid the correlation within a feature map, we use 
    spatial dropout instead of traditional dropout
    By Dong Nie
'''
class LongSKipAddUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, spatial_dropout_rate=0, how2norm='BN', isConvDilation=None, nd=2):
        super(LongSKipAddUnit, self).__init__()
        self.nd = nd
        # self.up = convTranspose23D_bn_relu_Unit(in_size, out_size, kernel_size=4, stride=2, padding=1, nd=nd)
        self.conv = residualUnit3(in_size, out_size, isDilation=isConvDilation, how2norm=how2norm, nd=nd)
        self.dp = dropout23DUnit(prob=spatial_dropout_rate, nd=nd)
        self.spatial_dropout_rate = spatial_dropout_rate
        self.conv2 = residualUnit3(out_size, out_size, isDilation=isConvDilation, isEmptyBranch1=False, how2norm=how2norm, nd=nd)

        self.relu = nn.ReLU()

    def center_crop(self, layer, target_size):  # we should make it adust to 2d/3d
        if self.nd == 2:
            batch_size, n_channels, layer_width, layer_height = layer.size()
        elif self.nd == 3:
            batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2

        if self.nd == 3:
            return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):  # bridge is the corresponding lower layer
        #         print 'x.shape: ', x.size()
        # up = self.up(x)

        crop1 = bridge
        #         crop1_dp = self.SpatialDroput(crop1)
        if self.spatial_dropout_rate > 0:
            crop1 = self.dp(crop1)
        out = self.relu(torch.add(x, crop1))
        out = self.conv(out)
        out = self.conv2(out)
        return out

'''
    Dilated residual module with two residual block (with diltion k)
    Note, here we keep the same resolution size among successive layers
input:
    x: input feature map
    k: dilation k
output:
    y
'''
class DilatedResUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, dilation=2, how2norm='BN', nd=2):
        super(DilatedResUnit,self).__init__()
        self.nd = nd
        mid_size = out_size/1
        padding = dilation*(kernel_size-1)/2
        if how2norm=='GN':
            self.conv1_bn_relu = conv23D_gn_relu_Unit(in_channels=in_size, out_channels=mid_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, nd=nd)
            self.conv2_bn_relu = conv23D_gn_relu_Unit(in_channels=mid_size, out_channels=mid_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, nd=nd)
        else:
            self.conv1_bn_relu = conv23D_bn_relu_Unit(in_channels=in_size, out_channels=mid_size,
                                                      kernel_size=kernel_size, stride=stride, padding=padding,
                                                      dilation=dilation, nd=nd)
            self.conv2_bn_relu = conv23D_bn_relu_Unit(in_channels=mid_size, out_channels=mid_size,
                                                      kernel_size=kernel_size, stride=stride, padding=padding,
                                                      dilation=dilation, nd=nd)
        self.relu = nn.ReLU()
    def forward(self, x):
        #dilated module 1
        conv1_1 = self.conv1_bn_relu(x)
        conv1_2 = self.conv2_bn_relu(conv1_1)
        out1 = torch.add(x,conv1_2) # we should make sure x is same size with conv2
        #dilated module 2
        conv2_1 = self.conv1_bn_relu(out1)
        conv2_2 = self.conv2_bn_relu(conv2_1)
        out = torch.add(conv2_1,conv2_2)
        return out

'''
    Semantic guidance for shallow layer feature learning module
input:
    x: input feature map (concatenating low and high layers)
output: 
    y: semantic guided shallow layer feature maps
'''
class SemGuidedModule(nn.Module):
    def __init__(self, channel, reduction=2, nd=2):
        super(SemGuidedModule, self).__init__()
        self.nd = nd
        if self.nd == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif self.nd == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        else:
            assert self.nd == 2 or self.nd == 3, 'ERROR, nd should be 2 or 3'
        self.ca = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel//reduction, channel//reduction),
                nn.Sigmoid()
        )
        if self.nd==2:
            self.sa = nn.Sequential(
                    conv23DUnit(channel,1,(1,1),nd=self.nd),
                    nn.Sigmoid()
            )
        elif self.nd==3:
            self.sa = nn.Sequential(
                    conv23DUnit(channel,1,(1,1,1),nd=self.nd),
                    nn.Sigmoid()
            )
        else:
            assert self.nd == 2 or self.nd == 3, 'ERROR, nd should be 2 or 3'

    def forward(self, x,y): # low-layer feature maps, high-layer feature maps
        # channel-wise attention
        z = torch.cat((x,y),dim=1)
        shape = z.size()
        camap = self.avg_pool(z).view(shape[0], shape[1])
        # print 'line682: shape is ',shape,' camap shape is ',camap.shape
        if self.nd == 2:
            camap = self.ca(camap).view(shape[0], shape[1]//2, 1, 1)
        elif self.nd == 3:
            camap = self.ca(camap).view(shape[0], shape[1]//2, 1, 1, 1)
        # cz = x * y
        # print 'line689: x.shape is ',x.shape, ' camap shape is ',camap.shape
        cz = torch.mul(x, camap)

        # spatial-wise attention
        samap = self.sa(z)
        if self.nd==2:
            samap = samap.view(shape[0], 1, shape[2], shape[3])
        elif self.nd==3:
            samap = samap.view(shape[0], 1, shape[2], shape[3], shape[4])

        sz = torch.mul(x, samap)
        return cz+sz



'''
 basic squeeze-and-excitation module for 23d
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, nd=2):
        super(SELayer, self).__init__()
        self.nd = nd
        if self.nd==2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif self.nd==3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        else:
            assert self.nd==2 or self.nd==3, 'ERROR, nd should be 2 or 3'
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        shape = x.size()
        # tt = self.avg_pool(x)
        y = self.avg_pool(x).view(shape[0], shape[1])
        if self.nd==2:
            y = self.fc(y).view(shape[0], shape[1], 1, 1)
        elif self.nd==3:
            y = self.fc(y).view(shape[0], shape[1], 1, 1, 1)
        # z = x*y
        return x * y

'''
  basic SE-RES-Block (two conv3x3 layers)
'''
class SEResidualUnit2(nn.Module):
    expansion = 1

    def __init__(self, in_size, out_size, stride=1, downsample=None, reduction=16, how2norm='BN', nd=2):
        super(SEResidualUnit2, self).__init__()
        assert nd == 2 or nd == 3, 'NDim should be 2 or 3'

        self.conv1 = conv23DUnit(in_size, out_size, kernel_size=3, stride=stride, padding=1, nd=nd)
        if how2norm=='GN':
            self.bn1 = nn.GroupNorm(out_size//8, out_size)
        else:
            self.bn1 = batchNorm23D(chn=out_size, nd=nd)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv23DUnit(out_size, out_size, kernel_size=3, stride=stride, padding=1, nd=nd)
        self.bn2 = batchNorm23D(chn=out_size, nd=nd)
        self.se = SELayer(out_size, reduction=reduction, nd=nd)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

'''
  SE-RES-Module (3 conv layers in the res module)
'''
class SEResidualUnit3(nn.Module):
    expansion = 4

    def __init__(self, in_size, out_size, stride=1, downsample=None, reduction=16, nd=2):
        assert nd == 2 or nd == 3, 'NDim should be 2 or 3'
        super(SEResidualUnit3, self).__init__()

        self.conv1 = conv23DUnit(in_size, out_size, kernel_size=1, nd=nd)
        self.bn1 = batchNorm23D(chn=out_size, nd=nd)
        self.conv2 = conv23DUnit(out_size, out_size, kernel_size=3, stride=stride, padding=1, nd=nd)
        self.bn2 = batchNorm23D(chn=out_size, nd=nd)
        self.conv3 = conv23DUnit(out_size, out_size * 2, kernel_size=1, nd=nd)
        self.bn3 = batchNorm23D(chn=out_size*2, nd=nd)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(out_size * 2, reduction=reduction, nd = nd)
        self.downsample = downsample
        self.stride = stride
        self.convF = conv23D_bn_relu_Unit(out_size*2, out_size, kernel_size=1, stride=1, nd = nd)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.convF(out)
        #out = self.relu(out)

        return out


'''
    Basic Residual upsampling block with long-range residual connection. Note we didn't have two 
    short residual blocks after the long-range residual operation. 
input:
    x: the current layer you want to consider
    bridge: the one you want to combine (using summary instead of concatenation) from lower layers
output:
    residual output
    
    By Dong Nie
'''
class BaseResUpUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False, nd=2):
        super(BaseResUpUnit, self).__init__()
        self.nd = nd
        self.up = convTranspose23D_bn_relu_Unit(in_size, out_size, kernel_size=4, stride=2, padding=1, nd=nd)

        self.relu = nn.ReLU()

    def center_crop(self, layer, target_size): #we should make it adust to 2d/3d
        if self.nd ==2:
            batch_size, n_channels, layer_width, layer_height = layer.size()
        elif self.nd==3:
            batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        
        if self.nd==3:
            return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size),  xy1:(xy1 + target_size)]
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):#bridge is the corresponding lower layer
        up = self.up(x)

        crop1 = bridge
        out = self.relu(torch.add(up, crop1))

        return out

'''
    upsample unit: first upsample (directly interpolation, and then conv_bn_relu
'''    
class upsampleUnit(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels, nd=2):
        super(upsampleUnit, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1_bn_relu = conv23D_bn_relu_Unit(in_channels, out_channels, 3, stride=1, padding=1, nd=nd)

    def forward(self, x):
        return self.conv1_bn_relu(x)

'''
    unetConvUnit: actually, a basic unit composed of two-layer convolutional layers
'''
class unetConvUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, nd=2):
        super(unetConvUnit, self).__init__()
        self.conv = conv23D_bn_relu_Unit(in_size, out_size, kernel_size=3, stride=1, padding=1, nd=nd)
        self.conv2 = conv23D_bn_relu_Unit(out_size, out_size, kernel_size=3, stride=1, padding=1, nd=nd)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv2(out)

        return out

'''
    unet upsampling block
'''
class unetUpUnit(nn.Module):
    def __init__(self, in_size, out_size, concat_in_size, kernel_size=3, activation=F.relu, space_dropout=False, isTModule=None, isFModule=False, nd=2):
        super(unetUpUnit, self).__init__()

        self.up = convTranspose23DUnit(in_size, out_size, kernel_size=4, stride=2, padding=1, nd=nd)
        self.bnup = batchNorm23D(out_size, nd=nd)
        self.conv = conv23D_bn_relu_Unit(in_size, out_size, kernel_size=3, stride=1, padding=1, nd=nd) #has some problem with the in_size

        self.conv2 = conv23D_bn_relu_Unit(out_size, out_size, kernel_size=3, stride=1, padding=1, nd=nd)
        self.activation = activation
        self.nd = nd
        self.isTModule = isTModule
        self.isFModule = isFModule
        if isTModule=='HR':
            self.t_block1 = DilatedResUnit(concat_in_size, 32, kernel_size=3, stride=1, dilation=2, nd=3)
            self.t_block2 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=3, nd=3)
            self.t_block3 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=4, nd=3)
            self.t_block4 = DilatedResUnit(32, out_size, kernel_size=3, stride=1, dilation=5, nd=3)

        if isFModule:
            self.f_block = SEResidualUnit3(out_size, out_size, nd=nd)

    def center_crop(self, layer, target_size): #we should make it adust to 2d/3d
        if self.nd ==2:
            batch_size, n_channels, layer_width, layer_height = layer.size()
        elif self.nd==3:
            batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        
        if self.nd==3:
            return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size),  xy1:(xy1 + target_size)]
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):#bridge is the corresponding lower layer
        up = self.activation(self.bnup(self.up(x)))
         
#         crop1 = self.center_crop(bridge, up.size()[2])
        if self.isTModule:
            crop4 = self.t_block1(bridge)
            crop3 = self.t_block2(crop4)
            crop2 = self.t_block3(crop3)
            crop1 = self.t_block4(crop2)
        else:
            crop1 = bridge
        out = torch.cat([up, crop1], 1)
        if self.isFModule:
            out = self.f_block(out)
        else:
            out = self.activation(self.conv(out))
            out = self.activation(self.conv2(out))

        return out


'''
    residual unet upsampling block
'''
class resUnetUpUnit(nn.Module):
    def __init__(self, in_size, out_size, concat_in_size, kernel_size=3, activation=F.relu, space_dropout=False, isTModule=None,
                 isFModule=False, isConvDilation=False, spatial_dropout_rate=0, nd=2):
        super(resUnetUpUnit, self).__init__()

        self.up = convTranspose23DUnit(in_size, out_size, kernel_size=4, stride=2, padding=1, nd=nd)
        self.bnup = batchNorm23D(out_size, nd=nd)
        self.activation = activation

        self.nd = nd
        self.isTModule = isTModule
        self.isFModule = isFModule

        if isTModule=='HR':
            self.t_block1 = DilatedResUnit(concat_in_size, 32, kernel_size=3, stride=1, dilation=2, nd=nd)
            self.t_block2 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=3, nd=nd)
            self.t_block3 = DilatedResUnit(32, 32, kernel_size=3, stride=1, dilation=4, nd=nd)
            self.t_block4 = DilatedResUnit(32, out_size, kernel_size=3, stride=1, dilation=5, nd=nd)
        elif isTModule=='DP':
            self.t_block = dropout23DUnit(prob=spatial_dropout_rate,nd=nd)
        elif isTModule=='CC':
            self.t_block = conv23D_bn_relu_Unit(concat_in_size, out_size,kernel_size=1,nd=nd)


        if isFModule:
            self.f_block = SEResidualUnit3(out_size*2, out_size, nd=nd)
        else:
            self.conv = residualUnit3(out_size * 2, out_size, isDilation=isConvDilation, nd=nd)
        self.conv = residualUnit3(out_size * 2, out_size, isDilation=isConvDilation, nd=nd) #we donot need it, we should comment it
        self.conv2 = residualUnit3(out_size, out_size, isDilation=isConvDilation, isEmptyBranch1=False, nd=nd)

    def center_crop(self, layer, target_size):  # we should make it adust to 2d/3d
        if self.nd == 2:
            batch_size, n_channels, layer_width, layer_height = layer.size()
        elif self.nd == 3:
            batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2

        if self.nd == 3:
            return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):  # bridge is the corresponding lower layer
        up = self.activation(self.bnup(self.up(x)))

        #         crop1 = self.center_crop(bridge, up.size()[2])
        if self.isTModule=='HR':
            crop4 = self.t_block1(bridge)
            crop3 = self.t_block2(crop4)
            crop2 = self.t_block3(crop3)
            crop1 = self.t_block4(crop2)
        elif self.isTModule=='DP':
            crop1 = self.t_block(bridge)
        elif self.isTModule=='CC':
            crop1 = self.t_block(bridge)
        else:
            crop1 = bridge
        out = torch.cat([up, crop1], 1)
        if self.isFModule:
            out = self.f_block(out)
            out = self.activation(self.conv2(out))
        else:
            out = self.activation(self.conv(out))
            out = self.activation(self.conv2(out))

        return out


'''
    The weighted cross entropy loss for 3D data, usually used in voxel wise segmentation.
input:
    predict: 5D tensor, even Variable: NxCxHXWXD
    target: 4D tensor, even Variable: NxHxWxD
    weight_map: 4D tensor, NxHxWxD
output:
    loss
    Feb, 2018
    By Dong Nie
'''
class WeightedCrossEntropy3d(nn.Module):

    def __init__(self, weight = None, size_average=True, reduce = True, ignore_label=255):
        '''weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"'''
        super(WeightedCrossEntropy3d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.nll_loss = nn.NLLLoss(weight, size_average=False, reduce=False)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, predict, target, weight_map=None):
        """
            Args:
                predict:(n, c, h, w, d)
                target:(n, h, w, d): 0,1,...,C-1       
        """
        assert not target.requires_grad
        assert predict.dim() == 5
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
        assert predict.size(4) == target.size(3), "{0} vs {1} ".format(predict.size(4), target.size(3))
        n, c, h, w, d = predict.size()
        logits = self.logsoftmax(predict) #NxCxWxHxD
        voxel_loss = self.nll_loss(logits, target) #NxWxHxD
        weighted_voxel_loss = weight_map*voxel_loss
        loss = torch.sum(weighted_voxel_loss)/(n*h*w*d)
#         print 'cross-entropy-loss: ',type(loss)
        return loss

'''
    The cross entropy loss for 3D data, usually used in voxel wise segmentation.
input:
    predict: 5D tensor, even Variable: NxCxHXWXD
    target: 4D tensor, even Variable: NxHxWxD
output:
    loss
    
    By Dong Nie
'''
class CrossEntropy3d(nn.Module):

    def __init__(self, weight = None, size_average=True, ignore_label=255):
        '''weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"'''
        super(CrossEntropy3d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w, d)
                target:(n, h, w, d): 0,1,...,C-1       
        """
        assert not target.requires_grad
        assert predict.dim() == 5
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
        assert predict.size(4) == target.size(3), "{0} vs {1} ".format(predict.size(4), target.size(3))
        n, c, h, w, d = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label) #actually, it doesn't convert to one-hot format
        target = target[target_mask] #N*1
        predict = predict.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous() # n, h, w, d, c
        predict = predict[target_mask.view(n, h, w, d, 1).repeat(1, 1, 1, 1, c)].view(-1, c) #N*C
        loss = F.cross_entropy(predict, target, weight = self.weight, size_average = self.size_average)
#         print 'cross-entropy-loss: ',type(loss)
        return loss


'''
    The cross entropy loss for 2D data, usually used in pixel wise segmentation.
input:
    predict: 4D tensor, even Variable
    target: 3D tensor, even Variable
output:
    loss
    
    By Dong Nie
'''
class CrossEntropy2d(nn.Module):

    def __init__(self, weight = None, size_average=True, ignore_label=255):
        '''weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"'''
        super(CrossEntropy2d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w): 0,1,...,C-1       
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight = self.weight, size_average = self.size_average)
#         print 'cross-entropy-loss: ',type(loss)
        return loss



'''
    The cross entropy loss for 2D dataset (usually used in FCN based segmentation)
    implemented by using nn.LLLoss2d, we use F.log_softmax to implement log_softmax in 2D.
    This one is believed to be stable and also faster
'''    
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
     
 
    
'''
    This criterion is a implementation of Focal Loss, which is proposed in 
    Focal Loss for Dense Object Detection.
    
    Now I implement it in the environment of segmentation

        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.

Input:
    class_num: the number of categories
    alpha(1D Tensor, Variable) : the scalar factor for this criterion
    gamma: aim at reducing the relative loss for the well classified examples, focusing more on hard, misclassified example
    size_average: true by default, the losses are averaged over observations for each minibatch, if set by false, it will summaried for each minibatch
    inputs: a 4D tensor for the predicted segmentation maps (before softmax), NXCXWXH
    targets: a 3D tensor for the ground truth segmentation maps, NXWXH
Output:
    The averaged losses

10/31/2017
By Dong Nie
'''
class myFocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(myFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        assert inputs.dim()==4,'inputs size should be 4: NXCXWXH'
        N = inputs.size(0)
        C = inputs.size(1)
        W = inputs.size(2)
        H = inputs.size(3)
        P = F.softmax(inputs,dim=1)

        ## one hot embeding for the targets ##
        class_mask = inputs.data.new(N, C, W, H).fill_(0)
        class_mask = Variable(class_mask)
        
        targets = torch.unsqueeze(targets,1) #Nx1xHxW
        class_mask.scatter_(1, targets, 1) #scatter along the 'numOfDims' dimension
        
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.) 
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
        alpha = 0.25

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

'''
    Dice loss for binary segmentation
'''
def dice_loss(input, target):

    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    IoU = (intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return 1 - 2 * IoU

'''
    Focal loss for binary segmentation implemented in log space for numerical stability
    gamma: hyper parameter for focal loss, can be 1,2,3,4,5
    input: output from last conv layer
    target: ground truth in original form 0,1

    passed test, succeed
    Sep, 2017
    By Dong Nie
'''
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super(FocalLoss,self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (targets.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), inputs.size()))

        inputs = F.log_softmax(inputs, dim=1)
        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * targets + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log() #log-loss

        # It gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        oneMprobs = F.logsigmoid(-inputs * (targets * 2 - 1))

        loss = (oneMprobs * self.gamma).exp() * loss

        return loss.mean()

'''
    One-hot encoding
'''
def make_one_hot(labels, C=2):

    if labels.dim()==4:
        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    else:
        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3), labels.size(4)).zero_()
    # print 'one_hot.shape: ',one_hot.shape, ' labels.shape: ',labels.shape
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)

    return target

'''
focal loss for multi-category segmentation implemented in log space for numerical stability
    weight: a manual rescaling weight given to each class, we can give each weight to each class
    gamma: can be [1,5], usually we assign 2.
    
    passed test, succeed
    Sep, 2017
    By Dong Nie
'''
class FocalLossMultiCategory(nn.Module):
    def __init__(self, gamma=2, weight=[1,1]):
        super(FocalLossMultiCategory, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(weight=weight, reduce=False) #input should be logits

    def forward(self, inputs, targets):
        """
            Args:
                inputs:(n, c, h, w, d), should be output of the last convolutional layer
                targets:(n, h, w, d): 0,1,...,C-1
        """
        assert not targets.requires_grad
        assert inputs.dim() == 5 or inputs.dim()==4, inputs.shape
        assert targets.dim() == 4 or targets.dim()==3, targets.shape
        assert inputs.dim() == targets.dim()+1,"{0} vs {1} ".format(inputs.dim(), targets.dim())
        assert inputs.size(0) == targets.size(0), "{0} vs {1} ".format(inputs.size(0), targets.size(0))
        assert inputs.size(2) == targets.size(1), "{0} vs {1} ".format(inputs.size(2), targets.size(1))
        assert inputs.size(3) == targets.size(2), "{0} vs {1} ".format(inputs.size(3), targets.size(2))
        if inputs.dim()==5:
            assert inputs.size(4) == targets.size(3), "{0} vs {1} ".format(inputs.size(4), targets.size(3))

        inputs = F.log_softmax(inputs, dim=1)
        loss = self.nll(inputs, targets) #input should be logits, so we need to first convert inputs(convLast) to logits

        one_hot = make_one_hot(targets.unsqueeze(dim=1), inputs.size()[1])
        oneMprobs = 1 - inputs.exp() #Returns a new tensor with the exponential of the elements of input, because inputs have converted to log space
        focal_weights = (oneMprobs * one_hot).sum(dim=1) ** self.gamma
        loss = loss * focal_weights

        return loss.mean()


'''
    Dice cost function for a single organ
input:
    input: a torch variable of size BatchxnclassesxHxW representing probabilities for each class
    target: a a also tensor, with batchx1xHxW
output:
    Variable scalar loss
    succeed for two type segmentation
'''
def myDiceLoss4Organ(input,target):
#     assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4 or input.dim() == 5, "Input must be a 4D Tensor or a 5D tensor."
    eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
    one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
    two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))

    target1 = Variable(torch.unsqueeze(target.data,1)) #Nx1xHxW or Nx1xHxWxD
    target_one_hot = Variable(torch.cuda.FloatTensor(input.size()).zero_()) #NxCxHxW or NxCxHxWxD
#     target_one_hot = target_one_hot.permute(0,2,3,1) #NxHxWxC
    target_one_hot.scatter_(1, target1, 1) #scatter along the 'numOfDims' dimension
    
    uniques=np.unique(target_one_hot.data.cpu().numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

#     print 'line 330: size: ',target_one_hot.size()
    
    probs = F.softmax(input,dim=1) #maybe it is not necessary
#     print 'line 331: size: ',probs.size()
    
    target = target_one_hot.contiguous().view(-1,1).squeeze(1)
    result = probs.contiguous().view(-1,1).squeeze(1)
    #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
    
#     intersect = torch.dot(result, target) #it doesn't support autograd
    
    intersect_vec = result * target
    intersect = torch.sum(intersect_vec)
        
    target_sum = torch.sum(target)
    result_sum = torch.sum(result)
    union = result_sum + target_sum + (two*eps)
#     print 'type of union: ',type(union)

    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#             out = torch.add(out, IoU.data*2)
    dice_total = one - two*IoU

#     dice_total = -1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
#     print 'type of dice_total: ', type(dice_total)
    return dice_total


'''
    This is dice loss for more than one organs, which means you can compute dice loss for more than one organ at a time, 
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
output:
    loss: Variable scalar
    
    succeed for multiple class segmentation problem
'''
def myDiceLoss4Organs(inputs, targets):
    
    eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
    one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
    two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))
                      
    inputSZ = inputs.size() #it should be sth like NxCxHxW
    
    inputs = F.softmax(inputs,dim=1)
    _, results_ = inputs.max(1)
    results = torch.squeeze(results_) #NxHxW

    
    numOfCategories = inputSZ[1]
    
    ####### Convert categorical to one-hot format
    targetSZ = results.size() #NxHxW
    ## We consider NxHxW 3D tensor
#     result1 = torch.unsqueeze(results, 1) #Nx1xHxW
#     results_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_()) #NxCxHxW
#     results_one_hot.scatter_(1,result1,1) #scatter along the 'numOfDims' dimension
    results_one_hot = inputs
    
    
    target1 = Variable(torch.unsqueeze(targets.data,1)) #Nx1xHxW
    targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_()) #NxCxHxW
#     targets_one_hot = targets_one_hot.permute(0,2,3,1) #NxHxWxC
    targets_one_hot.scatter_(1, target1, 1) #scatter along the 'numOfDims' dimension
#     print 'line 367: one_hot size: ',targets_one_hot.size()
    
    
    ###### Now the prediction and target has become one-hot format
    ###### Compute the dice for each organ
#     intersects = Variable(torch.FloatTensor(numOfCategories).zero_())
#     unions = Variable(torch.FloatTensor(numOfCategories).zero_())
    out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad = True)
#     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
#     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)

    for organID in range(0, numOfCategories):
#         target = targets_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
#         result = results_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
        target = targets_one_hot[:,organID,...].contiguous().view(-1,1).squeeze(1) #can be used as 2D/3D
        result = results_one_hot[:,organID,...].contiguous().view(-1,1).squeeze(1) #can be used as 2D/3D
#             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
        
#         intersect = torch.dot(result, target)
        intersect_vec = result * target
        intersect = torch.sum(intersect_vec)
#         print type(intersect)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
#         print type(result_sum)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (two*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
#             out = torch.add(out, IoU.data*2)
        out = out + one - two*IoU
#         intersects[organID], unions[organID] = intersect, union
#         print('organID: {} union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
# organID, union.data[0], intersect.data[0], target_sum.data[0], result_sum.data[0], IoU.data[0])

        
    denominator = Variable(torch.cuda.FloatTensor(1).fill_(numOfCategories))
    
    out = out / denominator
#     print type(out)
    return out



'''
    This is dice loss for one organ, which means you can compute dice for more than one organs at a time, 
    but it adapts to 0/1/2/3... (foreground,organ1,organ2,... problem)
input:
    inputs: predicted segmentation map, tensor type,  Variable (NxCxHxW)
    targets: real segmentation map, tensor type,  Variable (NxHxW)
output:
    loss: Variable scalar
    
    succeed for multiple organs
'''
   
class myWeightedDiceLoss4Organs(nn.Module):
    def __init__(self, organIDs = [1], organWeights=[1]):
        super(myWeightedDiceLoss4Organs, self).__init__()
        self.organIDs = organIDs
        self.organWeights = organWeights
#         pass

    def forward(self, inputs, targets, save=True):
        """
            Args:
                inputs:(n, c, h, w, d)
                targets:(n, h, w, d): 0,1,...,C-1       
        """
        assert not targets.requires_grad
        assert inputs.dim() == 5, inputs.shape
        assert targets.dim() == 4, targets.shape
        assert inputs.size(0) == targets.size(0), "{0} vs {1} ".format(inputs.size(0), targets.size(0))
        assert inputs.size(2) == targets.size(1), "{0} vs {1} ".format(inputs.size(2), targets.size(1))
        assert inputs.size(3) == targets.size(2), "{0} vs {1} ".format(inputs.size(3), targets.size(2))
        assert inputs.size(4) == targets.size(3), "{0} vs {1} ".format(inputs.size(4), targets.size(3))
        
        
        eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
        one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
        two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))
                          
        inputSZ = inputs.size() #it should be sth like NxCxHxW
        
        inputs = F.softmax(inputs, dim=1)
#         _, results_ = inputs.max(1)
#         results = torch.squeeze(results_) #NxHxW
    
        numOfCategories = inputSZ[1]
        assert numOfCategories==len(self.organWeights), 'organ weights is not matched with organs (bg should be included)'
        ####### Convert categorical to one-hot format

        results_one_hot = inputs
        
        
        target1 = Variable(torch.unsqueeze(targets.data,1)) #Nx1xHxW
        targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_()) #NxCxHxW
        targets_one_hot.scatter_(1, target1, 1) #scatter along the 'numOfDims' dimension
 
        ###### Now the prediction and target has become one-hot format
        ###### Compute the dice for each organ
        out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad = True)
    #     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
    #     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
    
        for organID in range(0, numOfCategories):
#             target = targets_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
#             result = results_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
            target = targets_one_hot[:,organID,...].contiguous().view(-1,1).squeeze(1) #for 2D or 3D
            result = results_one_hot[:,organID,...].contiguous().view(-1,1).squeeze(1) #for 2D or 3D
    #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
            
    #         intersect = torch.dot(result, target)
            intersect_vec = result * target
            intersect = torch.sum(intersect_vec)
    #         print type(intersect)
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result)
    #         print type(result_sum)
            target_sum = torch.sum(target)
            union = result_sum + target_sum + (two*eps)
    
            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0
            IoU = intersect / union
    #             out = torch.add(out, IoU.data*2)
            out = out + self.organWeights[organID] * (one - two*IoU)
    #         print('organID: {} union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
    # organID, union.data[0], intersect.data[0], target_sum.data[0], result_sum.data[0], IoU.data[0])
    
        denominator = Variable(torch.cuda.FloatTensor(1).fill_(sum(self.organWeights)))    
#         denominator = Variable(torch.cuda.FloatTensor(1).fill_(numOfCategories))
        
        out = out / denominator
    #     print type(out)
        return out        


'''
    This is generalized dice loss for organs, which means you can compute dice for more than one organ at a time
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
    
output:
    loss: scalar
    
    April, 2018, let's test, succeed
'''
   
class GeneralizedDiceLoss4Organs(nn.Module):
    def __init__(self,  organIDs = [1], size_average=True):   
        super(GeneralizedDiceLoss4Organs,self).__init__()
        self.organIDs = organIDs
        self.size_average = size_average
        
    def forward(self, inputs, targets, save=True):
        """
            Args:
                inputs:(n, c, h, w, d)
                targets:(n, h, w, d): 0,1,...,C-1       
        """
        assert not targets.requires_grad
        assert inputs.dim() == 5 or inputs.dim()==4, inputs.shape
        assert targets.dim() == 4 or targets.dim()==3, targets.shape
        assert inputs.dim() == targets.dim()+1,"{0} vs {1} ".format(inputs.dim(), targets.dim())
        assert inputs.size(0) == targets.size(0), "{0} vs {1} ".format(inputs.size(0), targets.size(0))
        assert inputs.size(2) == targets.size(1), "{0} vs {1} ".format(inputs.size(2), targets.size(1))
        assert inputs.size(3) == targets.size(2), "{0} vs {1} ".format(inputs.size(3), targets.size(2))
        if inputs.dim()==5:
            assert inputs.size(4) == targets.size(3), "{0} vs {1} ".format(inputs.size(4), targets.size(3))

        # print 'inputs.shape, ', inputs.shape,' targets.shape, ',targets.shape,' np.unique(targets), ',np.unique(targets.data.cpu().numpy())
        
        eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
        one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
        two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))
                          
        inputSZ = inputs.size() #it should be sth like NxCxHxW
        
        inputs = F.softmax(inputs, dim=1)
        # print 'max: ',np.amax(inputs.data.cpu().numpy()), 'min: ',np.amin(inputs.data.cpu().numpy())
        # preLabel = inputs.data.cpu().numpy().argmax(axis=1)
        # print 'preLabel.shape, ',preLabel.shape,' unique of prelabel: ', np.unique(preLabel), 'inputSZ:',inputSZ
        #
        # volout = sitk.GetImageFromArray(preLabel[0,...])
        # sitk.WriteImage(volout,'prePatch'+'.nii.gz')
        # volout = sitk.GetImageFromArray(targets.data.cpu().numpy()[0,...])
        # sitk.WriteImage(volout,'realPatch'+'.nii.gz')

        numOfCategories = inputSZ[1]
        assert numOfCategories==len(self.organIDs), 'organ weights is not matched with organs (bg should be included)'
        ####### Convert categorical to one-hot format

        results_one_hot = inputs
        
        
        target1 = Variable(torch.unsqueeze(targets.data,1)) #Nx1xHxW
        targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_()) #NxCxHxW
        targets_one_hot.scatter_(1, target1, 1) #scatter along the 'numOfDims' dimension
 
        ###### Now the prediction and target has become one-hot format
        ###### Compute the dice for each organ
        out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad = True)
    #     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
    #     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)

        intersect = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
        union = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
        
        for organID in range(0, numOfCategories):

            target = targets_one_hot[:,organID,...].contiguous().view(-1,1).squeeze(1) #for 2D or 3D
            result = results_one_hot[:,organID,...].contiguous().view(-1,1).squeeze(1) #for 2D or 3D
    #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
    #         print 'torch.sum(target): ',torch.sum(target)
            if torch.sum(target).cpu().data[0]==0:
                organWeight = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
            else:
                organWeight = 1/((torch.sum(target))**2+eps) # this is necessary, otherwise, union can be too big due to too big organ weight if some organ doesnot appear
#             print 'sum: %d'%torch.sum(target),' organWeight: %f'%organWeight
    #         intersect = torch.dot(result, target)
            intersect_vec = result * target
            # print 'organID: ',organID, ' intersect: ',torch.sum(intersect_vec).data[0],' organWeight: ',organWeight.data[0]
            intersect = intersect + organWeight*torch.sum(intersect_vec)
    #         print type(intersect)
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result)
    #         print type(result_sum)
            target_sum = torch.sum(target)
            # print 'organID: ', organID, ' union: ', (result_sum + target_sum).data[0], ' organWeight: ', organWeight.data[0]
            union = union + organWeight*(result_sum + target_sum) + (two*eps)
    
            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0
            
        IoU = intersect / union
        # print 'IoU is ', IoU.data[0],' intersect is ',intersect.data[0],' union is ',union.data[0]
        # print 'IoU is ', IoU.data[0]
#             out = torch.add(out, IoU.data*2)
        out =  one - two*IoU
        # print 'dice is ', out.data[0]
    #     print type(out)
        return out        

    


'''
    This is dice loss for one organ, which means you can only compute dice for one organ at a time, 
    but it only adapts to 0/1 (foreground,background problem)
    Note: we need to implement backward manually
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
   
output:
    loss: scalar
    
    so far, not succeed
'''
   
class WeightedDiceLoss4Organs(Function):
    def __init__(self, *args, **kwargs):
        self.numOfCategories = 2
#         pass

    def forward(self, inputs, targets, save=True):
        if save:
            self.save_for_backward(inputs, targets)
        eps = 0.000001
        
        inputSZ = inputs.size() #it should be sth like NxCxHxW
        
#         print 'line 336: inputs size: ',inputSZ
#         print 'line 337: input: ', inputs[1,1,1,1]
        
        _, results_ = inputs.max(1)
#         print 'line 339: results_ size: ',results_.size()
        results_ = torch.squeeze(results_) #NxHxW
#         print 'line 338: unique(results): ',unique(results_)
        self.inputs = torch.cuda.FloatTensor(inputs.size())
        self.inputs.copy_(inputs)
        
#         if inputs.is_cuda:
#             results = torch.cuda.FloatTensor(results_.size())
#             self.targets_ = torch.cuda.FloatTensor(targets.size())
#         else:
#             results = torch.FloatTensor(results_.size())
#             self.targets_ = torch.FloatTensor(targets.size())
            
        results = torch.cuda.LongTensor(results_.size())
        self.targets_ = torch.cuda.LongTensor(targets.size())
        results.copy_(results_) #NxHxW
        self.targets_.copy_(targets) 
        targets = self.targets_ #NxHxW
        
        self.numOfCategories = inputSZ[1]
        
        ####### Convert categorical to one-hot format
        targetSZ = results.size() #NxHxW
        numOfDims = len(targetSZ)
        ## We consider NxHxW 3D tensor
        result1 = torch.unsqueeze(results, numOfDims) #NxHxWx1
        self.results_one_hot = torch.cuda.FloatTensor(inputSZ).zero_() #NxCxHxW
        self.results_one_hot = self.results_one_hot.permute(0,2,3,1) #NxHxWxC
        self.results_one_hot.scatter_(numOfDims,result1,1) #scatter along the 'numOfDims' dimension
#         print 'line 361: one_hot size: ',self.results_one_hot.size()
        
        target1 = torch.unsqueeze(targets,numOfDims) #NxHxWx1
        self.targets_one_hot = torch.cuda.FloatTensor(inputSZ).zero_() #NxCxHxW
        self.targets_one_hot = self.targets_one_hot.permute(0,2,3,1) #NxHxWxC
        self.targets_one_hot.scatter_(numOfDims,target1,1) #scatter along the 'numOfDims' dimension
#         print 'line 367: one_hot size: ',self.targets_one_hot.size()
        
        
        ###### Now the prediction and target has become one-hot format
        ###### Compute the dice for each organ
#         out = torch.FloatTensor(self.numOfCategories).zero_()
        sumOut = 0.0
        self.intersect = torch.FloatTensor(self.numOfCategories).zero_()
        self.union = torch.FloatTensor(self.numOfCategories).zero_()
        for organID in range(0, self.numOfCategories):
            target = self.targets_one_hot[...,organID].contiguous().view(-1,1).squeeze(1)
            result = self.results_one_hot[...,organID].contiguous().view(-1,1).squeeze(1)
#             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
            
            intersect = torch.dot(result, target)
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result)
            target_sum = torch.sum(target)
            union = result_sum + target_sum + (2*eps)

            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0
            IoU = intersect / union
#             print('organID: {} union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#                 organID, union, intersect, target_sum, result_sum, 2*IoU))
#             out = torch.FloatTensor(1).fill_(2*IoU)
#             out[organID].fill_(2*IoU)
            sumOut = sumOut + IoU * 2
            self.intersect[organID], self.union[organID] = intersect, union
        out = torch.cuda.FloatTensor(1).fill_(sumOut)
        return out

    def backward(self, grad_output):
        inputs, _ = self.saved_tensors # we need probabilities for input
 
        targets = self.targets_one_hot # we need binary for targets
        intersects, unions = self.intersect, self.union
#         print 'targets size: ',targets.size(),'unions size: ',unions.size(),'intersects size: ',intersects.size()
        for i in range(0,self.numOfCategories):
            input = inputs[:,i,...]
            target = targets[...,i]
            union = unions[i]
            intersect = intersects[i]
            gt = torch.div(target, union)
            IoU2 = intersect/(union*union)
            print 'line 419: IoU2: ',IoU2

#             pred = torch.mul(input[:, 1], IoU2) #input[:,1] is equal to input[:,1,...]
            pred = torch.mul(input, IoU2) #input[:,1] is equal to input[:,1,...]
            print 'line 423: input: ',input.cpu()[1,1,1,...]
            print 'line 423: pred: ',pred.cpu()[1,1,1,...]
#             print 'gt size: ',gt.size(),' pred size: ',pred.size()
            print 'line 423: gt: ', gt.cpu()[1,1,1]
            dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
            if i==0:
                prev = torch.mul(dDice, grad_output[0])
            else:
                curr = torch.mul(dDice, -grad_output[0])
                grad_input = torch.cat((prev,curr), 0)
                prev = curr
        print 'line 429: grad_output: ',grad_output.cpu()        
        print 'line 430: grad_input: ', grad_input.cpu()[1,1,1,...]
        return grad_input , None


def unique(tensor1d):
    t, idx = np.unique(tensor1d.cpu().numpy(), return_inverse=True)
    return t


'''
    This is dice loss for organs, which means you can compute dice for more than one organ at a time
    Warning: The backward cannot done automatically, and this kind of dice expansion to multi-class is not correct
    Thus, I rewrite it.
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
    
output:
    loss: scalar
    
    so far, not succeed
'''
class DiceLoss4Organs(Function):
    def __init__(self,  organIDs = [1], organWeights=[1], size_average=True):   
        super(DiceLoss4Organs,self).__init__()
        self.organIDs = organIDs
        self.organWeights = organWeights
        self.size_average = size_average
        
    def forward(self, inputs, targets):
#         loss = 0.0
#         loss = torch.autograd.Variable(torch.Tensor(1), requires_grad=True)
  
        eps = 0.000001
        totalWeight = np.sum(self.organWeights)
        
        _,results = inputs.max(1) #maximize the channels
        results = torch.squeeze(results)


        sz =targets.size()
        singleSZ = targets[0,...].size()
#         print type(sz),singleSZ
        diceLoss = 0.0
        loss = np.zeros(sz[0])
#         print '..... target size: ',targets.size()
#         print '..... results size: ',results.size()
#         print type(inputs)
        for k in range(0,sz[0]):
            result_ = torch.FloatTensor(singleSZ)
            target_ = torch.FloatTensor(singleSZ)
            result_.copy_(results.data[k])
            target_.copy_(targets.data[k])
            for ind in range(0,len(self.organIDs)):
                id = self.organIDs[ind]
                #get the single organ representation
    #             indI = np.where(inputs==id)
    #             indI = inputs:eq(id)
                result = torch.zeros(result_.size())
                result[result_==id] = 1
                
    #             indT = np.where(targets==id)
    #             indT = targets:eq(id)
                target = torch.zeros(target_.size())
                target[target_==id] = 1
                
    #             iflat = input.view(-1)
    #             tflat = target.view(-1)
                intersect = torch.dot(result,target) #no need to flatern input or target
                intersect = np.max([eps, intersect]) #scalar value
    #             print '......intersect is: ', intersect
                input_sum = torch.sum(result)
                target_sum = torch.sum(target)
                union = input_sum + target_sum + 2*eps #not the real union
    #             print '.......union is: ', union
    #             print '..........dice is: ', 1.0*self.organWeights[ind]/totalWeight * (2.0 * intersect / union)
                loss[k] += 1.0 * self.organWeights[ind]/totalWeight*(1.0 - (2.0 * intersect / union))
            
#         loss = loss/len(self.organIDs)
#         print type(loss)
#         print 'loss shape: ', loss.shape,' loss is ',loss
        if self.size_average:
            diceLoss = loss.mean()
        else:
            diceLoss = loss.sum()

        diceLoss = torch.cuda.FloatTensor(1).fill_(1*diceLoss)
        diceLoss = Variable(diceLoss,requires_grad=True)
#         print 'haha2',type(loss)
        
        return diceLoss
    
    def backward(self, grad_ouput):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input , None



'''
    topK_Loss for regression problems.
    We only calculate the top K largest losses for all the elements 
'''


class topK_RegLoss(nn.Module):
    def __init__(self, topK, size_average=True):
        super(topK_RegLoss, self).__init__()
        self.size_average = size_average
        self.topK = topK

    def forward(self, preds, targets):
        """
            Args:
                inputs:(n, h, w, d)
                targets:(n, h, w, d)
        """
        assert not targets.requires_grad
        assert preds.shape == targets.shape, 'dim of preds and targets are different'

        K = torch.abs(preds - targets).view(-1)
        #
        #         base = targets.view(-1)
        #
        #         percent = K/base
        #         base[percent>thresold]

        # top 30% of 384x384 of 4 channels, be careful! K also has batches (used 8)! So actually it is (44236*4.0)/(384*384*2.0*8) = 0.075
        # 448*448*2.0*self.opt.batchSize*0.075
        if len(preds.shape) == 4:
            V, I = torch.topk(K, int(preds.size(0) * preds.size(1) * preds.size(2) * preds.size(3) * self.topK),
                              largest=True, sorted=True)
        else:
            V, I = torch.topk(K, int(preds.size(0) * preds.size(1) * preds.size(2) * self.topK), largest=True,
                              sorted=True)

        loss = torch.mean(V)

        return loss


'''
    topK_Loss for regression problems.
    We only calculate those losses larger than the specified threshold for all the elements 
'''
class RelativeThreshold_RegLoss(nn.Module):
    def __init__(self, threshold, size_average=True):
        super(RelativeThreshold_RegLoss, self).__init__()
        self.size_average = size_average
        self.eps = 1e-7
        self.threshold = threshold

    def forward(self, preds, targets):
        """
            Args:
                inputs:(n, h, w, d)
                targets:(n, h, w, d)
        """
        assert not targets.requires_grad
        assert preds.shape == targets.shape, 'dim of preds and targets are different'

        dist = torch.abs(preds - targets).view(-1)
        #
        baseV = targets.view(-1)

        baseV = torch.abs(baseV + self.eps)

        relativeDist = torch.div(dist, baseV)

        mask = relativeDist.ge(self.threshold)

        largerLossVec = torch.masked_select(dist, mask)

        loss = torch.mean(largerLossVec)

        return loss

'''
GDL loss for the image reconstruction
Note, this copy of gdl loss is successful. It may not be able to pass the gradcheck, that's becuase of the float/double issue, but it is fine.
If we change the filter to torch.DoubleTensor, and also cast the pred and gt as double tensor, it will pass the grad check. However, float is fine 
with the real application.
By Dong Nie
'''


class gdl_loss(nn.Module):
    def __init__(self, pNorm=2):
        super(gdl_loss, self).__init__()
        self.convX = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=(0, 1), bias=False)
        self.convY = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=(1, 0), bias=False)

        filterX = torch.FloatTensor([[[[-1, 1]]]])  # 1x2
        filterY = torch.FloatTensor([[[[1], [-1]]]])  # 2x1

        self.convX.weight = torch.nn.Parameter(filterX, requires_grad=False)
        self.convY.weight = torch.nn.Parameter(filterY, requires_grad=False)
        self.pNorm = pNorm

    def forward(self, pred, gt):
        assert not gt.requires_grad
        assert pred.dim() == 4
        assert gt.dim() == 4
        assert pred.size() == gt.size(), "{0} vs {1} ".format(pred.size(), gt.size())

        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))
        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))

        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)

        mat_loss_x = grad_diff_x ** self.pNorm

        mat_loss_y = grad_diff_y ** self.pNorm  # Batch x Channel x width x height

        shape = gt.shape

        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y)) / (shape[0] * shape[1] * shape[2] * shape[3])

        return mean_loss


'''
 semi-supervised/supervised loss functions for self-ensembling models for semi-supervised classification
'''


def temporal_loss_cla(out1, out2, w, labels):
    # MSE between current and temporal outputs
    def mse_loss(out1, out2):
        quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
        return quad_diff / out1.data.nelement()

    def masked_crossentropy(out, labels):
        cond = (labels >= 0)
        nnz = torch.nonzero(cond)
        nbsup = len(nnz)
        # check if labeled samples in batch, return 0 if none
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
            masked_labels = labels[cond]
            loss = F.cross_entropy(masked_outputs, masked_labels)
            return loss, nbsup
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0

    sup_loss, nbsup = masked_crossentropy(out1, labels)
    unsup_loss = mse_loss(out1, out2)
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup


'''
 semi-supervised/supervised loss functions for self-ensembling models for semi-supervised segmentation
  Dong Nie
  Sep. 2018 
'''


def temporal_loss_seg(out1, out2, w, labels, nd=3):
    # MSE between current and temporal outputs
    def mse_loss(out1, out2):
        quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
        return quad_diff / out1.data.nelement()

    sup_loss_dice = defGeneDiceLoss4Organs(out1, labels)
    if nd == 2:
        sup_loss_ce = defCrossEntropy2d(out1, labels)
    elif nd == 3:
        sup_loss_ce = defCrossEntropy3d(out1, labels)
    else:
        sup_loss_ce = 0

    sup_loss = sup_loss_dice + sup_loss_ce

    unsup_loss = mse_loss(out1, out2)
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss


'''
    This is generalized dice loss for organs, which means you can compute dice for more than one organ at a time
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
    actually, this copy of code can work for both 2D and 3D images
output:
    loss: scalar

    April, 2018, let's test, succeed
'''


def defGeneDiceLoss4Organs(inputs, targets):
    """
        Args:
            inputs:(n, c, h, w, d)
            targets:(n, h, w, d): 0,1,...,C-1
    """
    assert not targets.requires_grad
    assert inputs.dim() == 5, inputs.shape
    assert targets.dim() == 4, targets.shape
    assert inputs.size(0) == targets.size(0), "{0} vs {1} ".format(inputs.size(0), targets.size(0))
    assert inputs.size(2) == targets.size(1), "{0} vs {1} ".format(inputs.size(2), targets.size(1))
    assert inputs.size(3) == targets.size(2), "{0} vs {1} ".format(inputs.size(3), targets.size(2))
    assert inputs.size(4) == targets.size(3), "{0} vs {1} ".format(inputs.size(4), targets.size(3))

    eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
    one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
    two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))

    inputSZ = inputs.size()  # it should be sth like NxCxHxW

    inputs = F.softmax(inputs, dim=1)

    numOfCategories = inputSZ[1]
    #     assert numOfCategories==len(self.organIDs), 'organ weights is not matched with organs (bg should be included)'
    ####### Convert categorical to one-hot format

    results_one_hot = inputs

    target1 = Variable(torch.unsqueeze(targets.data, 1))  # Nx1xHxW
    targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_())  # NxCxHxW
    targets_one_hot.scatter_(1, target1, 1)  # scatter along the 'numOfDims' dimension

    ###### Now the prediction and target has become one-hot format
    ###### Compute the dice for each organ
    out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad=True)
    #     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
    #     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)

    intersect = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
    union = Variable(torch.cuda.FloatTensor(1).fill_(0.0))

    for organID in range(0, numOfCategories):

        target = targets_one_hot[:, organID, ...].contiguous().view(-1, 1).squeeze(1)  # for 2D or 3D
        result = results_one_hot[:, organID, ...].contiguous().view(-1, 1).squeeze(1)  # for 2D or 3D
        #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
        if torch.sum(target) == 0:
            organWeight = 0
        else:
            organWeight = 1 / ((torch.sum(target)) ** 2 + eps)
        #             print 'sum: %d'%torch.sum(target),' organWeight: %f'%organWeight
        #         intersect = torch.dot(result, target)
        intersect_vec = result * target
        intersect = intersect + organWeight * torch.sum(intersect_vec)
        #         print type(intersect)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        #         print type(result_sum)
        target_sum = torch.sum(target)
        union = union + organWeight * (result_sum + target_sum) + (two * eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0

    IoU = intersect / union
    #             out = torch.add(out, IoU.data*2)
    out = one - two * IoU

    #     print type(out)
    return out


'''
    The cross entropy loss for 2D data, usually used in pixel wise segmentation.
input:
    predict: 4D tensor, even Variable
    target: 3D tensor, even Variable
output:
    loss

    By Dong Nie
'''


def defCrossEntropy2d(predict, target, weight=None, ignore_label=None):
    """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w): 0,1,...,C-1
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
    assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
    assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != ignore_label)
    target = target[target_mask]
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, weight=weight)
    #         print 'cross-entropy-loss: ',type(loss)
    return loss


'''
    The cross entropy loss for 3D data, usually used in voxel wise segmentation.
input:
    predict: 5D tensor, even Variable: NxCxHXWXD
    target: 4D tensor, even Variable: NxHxWxD
output:
    loss

    By Dong Nie
'''


def defCrossEntropy3d(predict, target, weight=None, ignore_label=None):
    """
        Args:
            predict:(n, c, h, w, d)
            target:(n, h, w, d): 0,1,...,C-1       
    """
    assert not target.requires_grad
    assert predict.dim() == 5
    assert target.dim() == 4
    assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
    assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
    assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
    assert predict.size(4) == target.size(3), "{0} vs {1} ".format(predict.size(4), target.size(3))
    n, c, h, w, d = predict.size()
    target_mask = (target >= 0) * (target != ignore_label)  # actually, it doesn't convert to one-hot format
    target = target[target_mask]  # N*1
    predict = predict.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous()  # n, h, w, d, c
    predict = predict[target_mask.view(n, h, w, d, 1).repeat(1, 1, 1, 1, c)].view(-1, c)  # N*C
    loss = F.cross_entropy(predict, target, weight=weight)
    #         print 'cross-entropy-loss: ',type(loss)
    return loss



'''
Wasserstein loss
'''
def Wasserstein_Distance(D_real, D_fake):
    Wasserstein_Dist = D_real - D_fake

    return Wasserstein_Dist


'''
calculate gradient penalty for wgan
'''


def calc_gradient_penalty(netD, real_data, fake_data):
    # print real_data.size()
    batch_size = real_data.shape[0]
    alpha = torch.randn(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    # alpha = alpha.cuda(gpu) if use_cuda else alpha
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    #     if use_cuda:
    #         interpolates = interpolates.cuda(gpu)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


'''
compute the attention weights: 
    for prob belongs to [0.3,0.7] we use positive weights; 
    and we use negative weights belonging to [0,0.3) and (0.7,1]
Inputs:
    prob: probability
Outputs:
    res: attention we compute
'''
def computeAttentionWeight(prob):
    if prob>0.5:
        prob = 1 - prob
    res = 10.0/3*prob*prob + 7.0/3*prob -1
    if prob<0.3:
        res = 0
    return res


'''
compute the attention weights: 
    for prob belongs to [0.3,0.7] we use negative weights (easy samples for the segmenter, hard samples for the discriminator); 
    and we use positive weights belonging to [0,0.3) and (0.7,1] (hard samples for the segmenter, easy samples for the discriminator).
Inputs:
    prob: probability
Outputs:
    res: attention we compute
'''
def computeSampleAttentionWeight(prob):
    res = 12.5*(prob-0.5)*(prob-0.5)-0.5
    return res

'''
compute the attention weights: 
    difficulty-aware or confidence-aware voxel-wise loss
    here is sth like focal loss, for the high confidence region, we should retain the model, 
    if it is low confidence region, then we should pay more attention on these regions to train the model
Inputs:
    prob: probability = 1 - confidence probability
Outputs:
    res: attention we compute
'''
def computeVoxelAttentionWeight(prob):
    res = prob*prob
    return res

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=8):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)   
    
    
'''
    Sets the learning rate to the initial LR decayed by 10 every 10 epochs
'''
def adjust_learning_rate(optimizer, lr):
#     lr = opt.lr * (0.1 ** (epoch // opt.step))
    for param_group in optimizer.param_groups:
        print "current lr is ", param_group["lr"]
        if param_group["lr"] > lr:
            param_group["lr"] = lr  
    return lr
