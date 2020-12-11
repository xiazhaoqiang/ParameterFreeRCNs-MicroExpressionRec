import torch
import torch.nn as nn

class RecConv(nn.Module):

    def __init__(self, inplanes, planes):
        super(RecConv, self).__init__()
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
        # y = self.rrconv(x + y)
        out = self.downsample (y)

        return out


class RecNet(nn.Module):
    """Recurent networks with single output
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers):
        super(RecNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, featuremaps, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),
        )
        self.rcls = self._make_layer(RecConv, featuremaps, num_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Linear(5*5*featuremaps, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.rcls(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

class RecNet_v2(nn.Module):
    """Recurent networks with shallower architecture
        """
    def __init__(self, num_input, featuremaps, num_classes):
        super(RecNet_v2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, featuremaps, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Linear(5*5*featuremaps, num_classes)

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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

class RecNet_v3(nn.Module):
    """Recurent networks with multiple outputs
        """
    def __init__(self, num_input, featuremaps, num_classes, num_layers):
        super(RecNet_v3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, featuremaps, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),
        )
        self.rcls = self._make_layer(RecConv, featuremaps, num_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Linear(5*5*featuremaps, num_classes[0])
        self.classifier_aux1 = nn.Linear(5*5*featuremaps, num_classes[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.rcls(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        y_1 = self.classifier_aux1(x)

        return y, y_1
