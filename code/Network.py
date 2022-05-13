### YOUR CODE HERE
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""This script defines the network.

    we follow paper parameter setting. 
    For detail, we can refer to https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py

"""
# we will apply wrn for this project.
# Following HW2 code structure. We define standard_wide_residual for MyNetwork
class std_wresnet(nn.Module):
    """ 
        Create a standard wide residual network for MyNetwork.
        
    """

    def __init__(self, filters_in, filters_out, stride, drop_rate=0.0):
        super(std_wresnet, self).__init__()
        # based on architechure above, we need to define BN, ReLu, Conv, dropout, and shortcut
        self.bn1 = nn.BatchNorm2d(num_features=filters_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=filters_out)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=filters_out, out_channels=filters_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=drop_rate)
        # Define shortcut. If filters_in == filters_out, we can add x to output directly. Otherwise, we need a 1x1 conv layer.
        if filters_in == filters_out:
            self.projection_shortcut = None
        else:
            self.projection_shortcut = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=1, stride=stride, padding=0, bias=False)
    
    def forward(self, x):
        if self.projection_shortcut == None:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
            out = self.dropout(out)
            out = self.conv2(out)
            return torch.add(x, out)

        else:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))
            out = self.dropout(out)
            out = self.conv2(out)
            return torch.add(self.projection_shortcut(x), out)

class stack_block(nn.Module):
    def __init__(self, num_layers, filters_in, filters_out, stride, drop_rate):
        super(stack_block, self).__init__()

        stacked_layers = []
        for i in range(num_layers):
            if i == 0:
                stacked_layers.append(std_wresnet(filters_in,  filters_out, stride, drop_rate))
            else:
                stacked_layers.append(std_wresnet(filters_out, filters_out, 1     , drop_rate))

        self.stacked_layers = nn.Sequential(*stacked_layers)

    def forward(self, x):
        return self.stacked_layers(x)

class MyNetwork(nn.Module):

    def init_weights(self):
        for module in self.modules():
            print(module)
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()

    def __init__(self, config):
        super(MyNetwork, self).__init__()

        depth = config["depth"]
        num_classes = config["num_classes"]
        width = config["width"]
        drop_rate = config["drop_rate"]

        widths = [16 * width, 32 * width, 64 * width]
        n = (depth - 4) // 6 # depth is 6n + 4 so n = (depth - 4) // 6
        # we let our features be 10x times after each block.
        self.starting_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = stack_block(n, 16       , widths[0], 1, drop_rate)
        self.block2 = stack_block(n, widths[0], widths[1], 2, drop_rate)
        self.block3 = stack_block(n, widths[1], widths[2], 2, drop_rate)

        self.bn1 = nn.BatchNorm2d(widths[2])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(widths[2], num_classes)
        self.num_features = widths[2]
        self.init_weights()

    def forward(self, x):
        out = self.starting_layer(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8, 1, 0)
        out = out.view(-1, self.num_features) # change tensor shape
        out = self.fc(out)
        return out
### END CODE HERE