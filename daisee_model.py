import torch
import itertools

import torch.nn as nn
import torchvision.transforms.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
        x = self.fc(x)

        return x


def resnet18(pretrained_weights=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained_weights:
        model.load_state_dict(pretrained_weights)
    return model


class RegionLayer(nn.Module):

    def __init__(self, n_rows, n_cols, img_h, img_w):
        super(RegionLayer, self).__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.img_h = img_h
        self.img_w = img_w
        self.region_height = self.img_h // self.n_rows
        self.region_width = self.img_w // self.n_cols

        self.bn2d = nn.BatchNorm2d(32)
        self.prelu = nn.PReLU()
        self.conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        # assert x.shape == (1, 32, 160, 160)
        regions = [None] * self.n_rows
        for i in range(self.n_rows):
            regions[i] = [None] * self.n_cols
            
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                i_begin = i * self.region_height
                j_begin = j * self.region_width
                
                regions[i][j] = x[:, :, i_begin:i_begin+self.region_height, j_begin:j_begin+self.region_width]
                regions[i][j] = self.bn2d(regions[i][j])
                regions[i][j] = self.prelu(regions[i][j])
                regions[i][j] = self.conv(regions[i][j])

                  
        region_row = []
        for i in range(self.n_rows):
            t = torch.cat([regions[i][j] for j in range(self.n_cols)], dim=3)
            
            region_row.append(t)

        t = torch.cat([region_row[i] for i in range(self.n_rows)], dim=2)

        return t


class DRMLLayer(nn.Module):

    def __init__(self):
        super(DRMLLayer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(11, 11))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=(8, 8))
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(8, 8), stride=2)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=(6, 6))
        self.conv7 = nn.Conv2d(16, 16, kernel_size=(5, 5))
        self.conv8 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=3)
        self.fc = nn.Linear(1024, 512)
        self.region_layer = RegionLayer(4, 4, 160, 160).cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.region_layer(x)
        # assert x.shape == (1, 32, 160, 160),print(x.shape)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(-1)
        x = self.fc(x)
        return x


class DAiSEEModel(nn.Module):

    def __init__(self, subsample_rate, pretrained_weights=None):
        super(DAiSEEModel, self).__init__()
        self.bbox_resnet_1 = resnet18(pretrained_weights=pretrained_weights).cuda()
        #self.bbox_resnet_2 = resnet18(pretrained_weights=pretrained_weights)
        self.drml_layer = DRMLLayer().cuda()

        self.egaze_lin = nn.Linear(4, 10)
        self.lmarks_lin = nn.Linear(188, 30)
        self.bbox_lin = nn.Linear(1024, 60)
        self.fc = nn.Linear(100, 4)

        self.lstm = nn.LSTM(100, 100).cuda()
        self.h0 = torch.randn(1, 1, 100).cuda()
        self.c0 = torch.randn(1, 1, 100).cuda()

        self.smax = nn.Softmax(dim=1)
        
        self.subsampled_indices = self.subsampling_rate2_indices(subsample_rate)

    def subsampling_rate2_indices(self, rate, start_index=1, n_frames=300):

        return [i for i in range(start_index, n_frames+1, rate)]
        
    def forward(self, x):
    
        self.h0.detach_()
        self.c0.detach_()
        for i in self.subsampled_indices:
            
            x_bb = torch.cat([self.drml_layer(x[0][i][0]).unsqueeze_(0), self.bbox_resnet_1(x[1][i]["bbox_img"])], dim=1)
            x_bb = self.bbox_lin(x_bb)

            x_lm = torch.cat([x[0][i][1], x[1][i]["lmarks"]], dim=1)
            x_lm = self.lmarks_lin(x_lm)
                        
            x_eg = torch.cat([x[0][i][2], x[1][i]["egaze"]], dim=1)
            x_eg = self.egaze_lin(x_eg)    
            #return x_eg[:,0:4]
            lstm_outputs, (self.h0, self.c0) = self.lstm(
                    torch.cat([x_bb, x_lm, x_eg], dim=1).view(1,1,100), (self.h0, self.c0))
        
        assert lstm_outputs.shape == (1, 1, 100)
        lstm_outputs = lstm_outputs.view(1, 100)
        lstm_outputs = self.fc(lstm_outputs)
        lstm_outputs = self.smax(lstm_outputs)

        # TODO: 4 lstms???

        return lstm_outputs
