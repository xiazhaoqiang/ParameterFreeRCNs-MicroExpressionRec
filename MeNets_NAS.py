import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['MeNet_A', 'MeNet_D', 'MeNet_W', 'MeNet_H', 'MeNet_C', 'MeNet_E']

class ConvBlock(nn.Module):
    """convolutional layer blocks for sequtial convolution operations"""
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class RclBlock(nn.Module):
    """recurrent convolutional blocks"""
    def __init__(self, inplanes, planes):
        super(RclBlock, self).__init__()
        self.ffconv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.rrconv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )

    def forward(self, x):
        y = self.ffconv(x)
        y = self.rrconv(x + y)
        y = self.rrconv(x + y)
        out = self.downsample (y)
        return out

class DenseBlock(nn.Module):
    """densely connected convolutional blocks"""
    def __init__(self, inplanes, planes):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )

    def forward(self, x):
        y = self.conv1(x)
        z = self.conv2(x + y)
        # out = self.conv2(x + y + z)
        e = self.conv2(x + y + z)
        out = self.conv2(x + y + z + e)
        out = self.downsample (out)
        return out

class EmbeddingBlock(nn.Module):
    """densely connected convolutional blocks for embedding"""
    def __init__(self, inplanes, planes):
        super(EmbeddingBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=3, padding=0),
            nn.Dropout()
        )

    def forward(self, x, w, pool_size, classes):
        y = self.conv1(x)
        y1 = self.attenmap(F.adaptive_avg_pool2d(x, (pool_size, pool_size)), w, classes)
        y = torch.mul(F.interpolate(y1, (y.shape[2], y.shape[3])), y)
        z = self.conv2(x+y)
        e = self.conv2(x + y + z)
        out = self.conv2(x + y + z + e)
        out = self.downsample (out)
        return out

class EmbeddingBlock2(nn.Module):
    """densely connected convolutional blocks for embedding"""
    def __init__(self, inplanes, planes):
        super(EmbeddingBlock2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=3, padding=0),
            nn.Dropout()
        )

    def forward(self, x, w, pool_size, classes):
        y = self.conv1(x)
        #y1 = self.attenmap(F.adaptive_avg_pool2d(x, (pool_size, pool_size)), w, classes)
        #y = torch.mul(F.interpolate(y1, (y.shape[2], y.shape[3])), y)
        z = self.conv2(y)
        e = self.conv2(y + z)
        out = self.conv2(y + z + e)
        out = self.downsample (out)
        return out        
        
class SpatialAttentionBlock_A(nn.Module):
    """linear attention block for any layers"""
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttentionBlock_A, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l):
        N, C, W, H = l.size()
        c = self.op(l) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        return g

class SpatialAttentionBlock_P(nn.Module):
    """linear attention block for any layers"""
    def __init__(self, normalize_attn=True):
        super(SpatialAttentionBlock_P, self).__init__()
        self.normalize_attn = normalize_attn

    def forward(self, l, w, classes):
        output_cam = []
        for idx in range(0,classes):
            weights = w[idx,:].reshape((l.shape[1], l.shape[2], l.shape[3]))
            cam = weights * l
            cam = cam.mean(dim=1,keepdim=True)
            cam = cam - torch.min(torch.min(cam,3,True)[0],2,True)[0]
            cam = cam / torch.max(torch.max(cam,3,True)[0],2,True)[0]
            output_cam.append(cam)
        output = torch.cat(output_cam, dim=1)
        output = output.mean(dim=1,keepdim=True)
        return output

class SpatialAttentionBlock_F(nn.Module):
    """linear attention block for first layer"""
    def __init__(self, normalize_attn=True):
        super(SpatialAttentionBlock_F, self).__init__()
        self.normalize_attn = normalize_attn

    def forward(self, l, w, classes):
        output_cam = []
        for idx in range(0,classes):
            weights = w[idx,:].reshape((-1, l.shape[2], l.shape[3]))
            weights = weights.mean(dim=0,keepdim=True)
            cam = weights * l
            cam = cam.mean(dim=1,keepdim=True)
            cam = cam - torch.min(torch.min(cam,3,True)[0],2,True)[0]
            cam = cam / torch.max(torch.max(cam,3,True)[0],2,True)[0]
            output_cam.append(cam)
        output = torch.cat(output_cam, dim=1)
        output = output.mean(dim=1,keepdim=True)
        return output

def MakeLayer(block, planes, blocks):
    layers = []
    for _ in range(0, blocks):
        layers.append(block(planes, planes))
    return nn.Sequential(*layers)

class MeNet_A(nn.Module):
    """menet networks with adding attention unit
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5, model_version=3):
        super(MeNet_A, self).__init__()
        self.version = model_version
        self.classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, featuremaps, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),
        )
        self.rcls = MakeLayer(RclBlock, featuremaps, num_layers)
        self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.downsampling = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.version == 1:
            x = self.conv1(x)
            x = self.attenmap(x)
            x = self.rcls(x)
            x = self.avgpool(x)
        if self.version == 2:
            x = self.conv1(x)
            x = self.attenmap(x)
            x = self.rcls(x)
            x = self.avgpool(x)
        elif self.version == 3:
            x = self.conv1(x)
            y = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
            x = self.rcls(x)
            x = self.avgpool(x)
            x = x * y
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MeNet_D(nn.Module):
    """menet networks with dense connection
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(MeNet_D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, featuremaps, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),
        )
        self.dbl = MakeLayer(DenseBlock, featuremaps, num_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dbl(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MeNet_W(nn.Module):
    """menet networks with wide expansion
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(MeNet_W, self).__init__()
        num_channels = int(featuremaps/2)
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=5, stride=3, padding=2),
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=2, dilation=2),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream3 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=5, stride=3, padding=2),
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=3, dilation=3),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.rcls = MakeLayer(RclBlock, featuremaps, num_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x3 = self.stream3(x)
        x = torch.cat((x1,x2,x3),1)
        x = self.rcls(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MeNet_H(nn.Module):
    """menet networks with hybrid modules
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(MeNet_H, self).__init__()
        self.classes = num_classes
        num_channels = int(featuremaps/2)
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=1), # 1->3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=3, dilation=3), # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.dbl = MakeLayer(DenseBlock, featuremaps, num_layers)
        self.rcls = MakeLayer(RclBlock, featuremaps, num_layers)
        self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.downsampling = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x = torch.cat((x1,x2),1)
        y = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        x = self.dbl(x)
        x = self.avgpool(x)
        x = x * y
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

  

class MeNet_CS(nn.Module):
    """menet networks with cascaded modules with searching
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(MeNet_CS, self).__init__()
        self.classes = num_classes
        num_channels = int(featuremaps/2)
        self.archi = nn.Parameter(torch.randn(2,2))
        # self.stream1 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=1), # 1->3
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            # nn.Dropout(),
        # )
        # self.stream2 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=5, stride=1, padding=2), # 5,2/ 1,0
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            # nn.Dropout(),
        # )
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1), # 1->3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=3, dilation=3), # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(featuremaps, featuremaps, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(featuremaps, featuremaps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )


        self.softmax = nn.Softmax(0)
        self.attenmap = SpatialAttentionBlock_F(normalize_attn=True)
        self.downsampling = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        
        nn.init.constant(self.archi, 0.5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        W = self.softmax(self.archi)
        #W = self.archi
        #M for attention mask
        M1 = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x = torch.cat((x1,x2),1)
                
        x1 = torch.mul(F.interpolate(M1,(x.shape[2],x.shape[3])), x)
        
        #x = W[0][0]*x+W[0][1]*x1
        x = x+W[0][1]*x1
        #Second Ateention
        M2 = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        y = self.conv1(x)
        y1 = torch.mul(F.interpolate(M2,(y.shape[2],y.shape[3])), y)
        
        #y = W[1][0]*y + W[1][1]*y1
        y = y + W[1][1]*y1
        
        #Third Ateention
        M3 = self.attenmap(self.downsampling(y), self.classifier.weight, self.classes)
        z = self.conv2(x+y)
        z1 = torch.mul(F.interpolate(M3,(z.shape[2],z.shape[3])), z)
        
        #z = W[2][0]*z + W[2][1]*z1
        z = z #+ W[2][1]*z1
        #Forth Ateention
        M4 = self.attenmap(self.downsampling(z), self.classifier.weight, self.classes)
        e = self.conv2(x+y+z)
        e1 = torch.mul(F.interpolate(M4,(e.shape[2],e.shape[3])), e)
        
        e = e #+W[3][1]*e1
        #e = W[3][0]*e+W[3][1]*e1
        
        #Fiveth Ateention
        M5 = self.attenmap(self.downsampling(e), self.classifier.weight, self.classes)        
        out = self.conv2(x+y+z+e)
        out1 = torch.mul(F.interpolate(M5,(out.shape[2],out.shape[3])), out)
        
        #out = W[4][0]*out+W[4][1]*out1
        out = out #+W[4][1]*out1
        
        out = self.downsample(out)

        x = self.avgpool(out)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MeNet_CS2(nn.Module):
    """menet networks with cascaded modules with searching
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(MeNet_CS2, self).__init__()
        self.classes = num_classes
        num_channels = int(featuremaps/2)
        self.archi = nn.Parameter(torch.randn(4,2))
        # self.stream1 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=1), # 1->3
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            # nn.Dropout(),
        # )
        # self.stream2 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=5, stride=1, padding=2), # 5,2/ 1,0
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            # nn.Dropout(),
        # )
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1), # 1->3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=3, dilation=3), # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(featuremaps, featuremaps, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(featuremaps, featuremaps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )


        self.softmax = nn.Softmax(-1)
        self.attenmap = SpatialAttentionBlock_F(normalize_attn=True)
        self.downsampling = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        
        nn.init.constant(self.archi, 0.5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        W = self.softmax(self.archi)
        #W = self.archi
        #M for attention mask
        M1 = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x = torch.cat((x1,x2),1)

        #Second Ateention
        M2 = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        y = self.conv1(x)
        y1 = torch.mul(F.interpolate(M1,(y.shape[2],y.shape[3])), y)# Here we use M1
        
        y = W[0][0]*y + W[0][1]*y1

        #Third Ateention
        M3 = self.attenmap(self.downsampling(y), self.classifier.weight, self.classes)
        z = self.conv2(x+y)
        z1 = torch.mul(F.interpolate(M2,(z.shape[2],z.shape[3])), z)
        
        z = W[1][0]*z + W[1][1]*z1
        #z = z #+ W[2][1]*z1
        #Forth Ateention
        M4 = self.attenmap(self.downsampling(z), self.classifier.weight, self.classes)
        e = self.conv2(x+y+z)
        e1 = torch.mul(F.interpolate(M3,(e.shape[2],e.shape[3])), e)

        e = W[2][0]*e+W[2][1]*e1
        
        #Fiveth Ateention
        #M5 = self.attenmap(self.downsampling(e), self.classifier.weight, self.classes)        
        out = self.conv2(x+y+z+e)
        out1 = torch.mul(F.interpolate(M4,(out.shape[2],out.shape[3])), out)
        
        out = W[3][0]*out+W[3][1]*out1
        #out = out #+W[4][1]*out1
        
        out = self.downsample(out)

        x = self.avgpool(out)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MeNet_CS3(nn.Module):
    """menet networks with cascaded modules with searching
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(MeNet_CS3, self).__init__()
        self.classes = num_classes
        num_channels = int(featuremaps/2)
        self.archi = nn.Parameter(torch.randn(3,2))
        # self.stream1 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=1), # 1->3
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            # nn.Dropout(),
        # )
        # self.stream2 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=5, stride=1, padding=2), # 5,2/ 1,0
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            # nn.Dropout(),
        # )
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1), # 1->3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=3, dilation=3), # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(featuremaps, featuremaps, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(featuremaps, featuremaps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )


        self.softmax = nn.Softmax(-1)
        self.attenmap = SpatialAttentionBlock_F(normalize_attn=True)
        self.downsampling = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        
        nn.init.constant(self.archi, 0.5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        W = self.softmax(self.archi)
        #W = self.archi
        #M for attention mask
        M1 = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x = torch.cat((x1,x2),1)

        #Second Ateention
        M2 = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        y = self.conv1(x)

        #Third Ateention
        M3 = self.attenmap(self.downsampling(y), self.classifier.weight, self.classes)
        z = self.conv2(x+y)
        z1 = torch.mul(F.interpolate(M1,(z.shape[2],z.shape[3])), z)
        
        z = W[0][0]*z + W[0][1]*z1
        #z = z #+ W[2][1]*z1
        #Forth Ateention
        #M4 = self.attenmap(self.downsampling(z), self.classifier.weight, self.classes)
        e = self.conv2(x+y+z)
        e1 = torch.mul(F.interpolate(M2,(e.shape[2],e.shape[3])), e)

        e = W[1][0]*e+W[1][1]*e1
        
        #Fiveth Ateention
        #M5 = self.attenmap(self.downsampling(e), self.classifier.weight, self.classes)        
        out = self.conv2(x+y+z+e)
        out1 = torch.mul(F.interpolate(M3,(out.shape[2],out.shape[3])), out)
        
        out = W[2][0]*out+W[2][1]*out1
        #out = out #+W[4][1]*out1
        
        out = self.downsample(out)

        x = self.avgpool(out)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
class MeNet_C(nn.Module):
    """menet networks with cascaded modules
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(MeNet_C, self).__init__()
        self.classes = num_classes
        num_channels = int(featuremaps/2)
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=1), # 1->3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=5, stride=1, padding=2), # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.dbl = MakeLayer(DenseBlock, featuremaps, num_layers)
        # self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.attenmap = SpatialAttentionBlock_F(normalize_attn=True)
        self.downsampling = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x = torch.cat((x1,x2),1)
        # y = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        x = torch.mul(F.interpolate(y,(x.shape[2],x.shape[3])), x)
        x = self.dbl(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MeNet_E(nn.Module):
    """menet networks with embedded modules
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(MeNet_E, self).__init__()
        self.classes = num_classes
        self.poolsize = pool_size
        num_channels = int(featuremaps/2)
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1), # 1->3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=3, dilation=3), # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.ebl = EmbeddingBlock(featuremaps, featuremaps)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x = torch.cat((x1,x2),1)
        x = self.ebl(x, self.classifier.weight, self.poolsize, self.classes)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
        
# class MeNet_ES(nn.Module):
    # """menet networks with embedded modules by searching
    # """
    # def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        # super(MeNet_ES, self).__init__()
        # self.classes = num_classes
        # self.poolsize = pool_size
        # num_channels = featuremaps
        # self.archi = nn.Parameter(torch.randn(3))
        # self.stream1 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1), # 1->3
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_channels),
            # # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            # nn.Dropout(),
        # )
        # self.stream2 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1, dilation=2), # 5,2/ 1,0
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_channels),
            # # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            # nn.Dropout(),
        # )
        # self.stream3 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=3, dilation=3), # 5,2/ 1,0
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_channels),
            # # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            # nn.Dropout(),
        # )
        
        # self.bn = nn.BatchNorm2d(num_channels)
        # self.softmax = nn.Softmax(0)
        # self.ebl = EmbeddingBlock(featuremaps, featuremaps)
        # self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        # self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        # nn.init.constant(self.archi, 0.333)
        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    # def forward(self, x):
        # x1 = self.stream1(x)
        # x2 = self.stream2(x)
        # x3 = self.stream3(x)
        # #x = torch.cat((x1,x2,x3),1)
        # W = self.softmax(self.archi)
        # #print(W)
        # x = W[0]*x1+ W[1]*x2+ W[2]*x3
        # x = self.bn(x)
        # x = self.ebl(x, self.classifier.weight, self.poolsize, self.classes)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x

class MeNet_ES(nn.Module):
    """menet networks with embedded modules by searching
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5):
        super(MeNet_ES, self).__init__()
        self.classes = num_classes
        self.poolsize = pool_size
        num_channels = int(featuremaps/2)
        self.archi = nn.Parameter(torch.randn(2))
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1), # 1->3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=3, dilation=3), # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        
        self.stream21 = nn.Sequential(
            nn.Conv2d(featuremaps, num_channels, kernel_size=3, stride=3, padding=1), # 1->3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream22 = nn.Sequential(
            nn.Conv2d(featuremaps, num_channels, kernel_size=3, stride=3, padding=3, dilation=3), # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        
        #self.bn = nn.BatchNorm2d(num_channels)
        self.softmax = nn.Softmax(0)
        self.ebl = EmbeddingBlock2(num_input, featuremaps)
        self.ebl2 = EmbeddingBlock(featuremaps, featuremaps)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        nn.init.constant(self.archi, 0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        
        x = self.ebl(x, self.classifier.weight, self.poolsize, self.classes)
        x1 = torch.cat((x1,x2),1)
        
        W = self.softmax(self.archi)
        x = W[0]*x+ W[1]*x1
        
        
        #
        y = self.ebl2(x, self.classifier.weight, self.poolsize, self.classes)
        y1 = self.stream21(x)
        y2 = self.stream22(x)
        y1 = torch.cat((y1,y2),1)
        
        x = W[1]*y+ W[0]*y1
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        print(W)
        return x             