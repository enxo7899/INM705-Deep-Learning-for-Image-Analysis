import torch
import torchvision
import torch.hub
import torch.nn as nn

class ResNet34Wrapper:

    def __init__(self, out_classes_len, checkpoint=None):
        self.checkpoint_file = checkpoint
        self.model = torchvision.models.resnet34(weights=None)
        self.model_image_net = torchvision.models.resnet34(weights=None)
        self.model.fc.out_features = out_classes_len
        self.model.fc.weight = torch.nn.Parameter(
            torch.randn(self.model.fc.out_features, self.model.fc.in_features),
            requires_grad=True)
        # this must be a nn.Parameter
        self.model.fc.bias = torch.nn.Parameter(torch.randn(self.model.fc.out_features), requires_grad=True)  # same with the bias

class VGG16Wrapper:

    def __init__(self, out_classes_len, checkpoint=None):
        self.checkpoint_file = checkpoint
        self.model = torchvision.models.vgg16(weights=None)
        self.model_image_net = torchvision.models.vgg16(weights=None)
        self.model.fc.out_features = out_classes_len
        self.model.fc.weight = torch.nn.Parameter(
            torch.randn(self.model.fc.out_features, self.model.fc.in_features),
            requires_grad=True)
        # this must be a nn.Parameter
        self.model.fc.bias = torch.nn.Parameter(torch.randn(self.model.fc.out_features), requires_grad=True)  # same with the bias

# Custom ResNet Model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

# # Waste Classifier Model
class WasteClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WasteClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 56 * 56, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def new_method(self, num_classes):
        return num_classes

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLO(nn.Module):
    def __init__(self, num_classes=6):
        super(YOLO, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(48, 32, kernel_size=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv8 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv10 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv12 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv14 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv16 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv17 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv18 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv19 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv20 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv24 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv25 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv26 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv27 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv28 = nn.Conv2d(256, 1024, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.maxpool4(x)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x = F.relu(self.conv20(x))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        x = F.relu(self.conv24(x))
        x = F.relu(self.conv25(x))
        x = F.relu(self.conv26(x))
        x = F.relu(self.conv27(x))
        x = F.relu(self.conv28(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = YOLO()
print(model)