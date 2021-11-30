from torch import nn, Tensor
import torchvision


class Models(nn.Module):
    def __init__(self, arch: str, num_classes: int = 26 * 4):
        self.arch = arch
        super(Models, self).__init__()
        model_map = {
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152,
            'squeezenet1_0': torchvision.models.squeezenet1_0,
            'squeezenet1_1': torchvision.models.squeezenet1_1,
            'mobilenet': torchvision.models.mobilenet_v2,
            'cnn': CNN,
            'cnnv2': CNNv2,
        }
        model = model_map[arch]()
        if arch.startswith('resnet'):
            model.inplanes = 64
            model.conv1 = nn.Conv2d(1, model.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=3,
                                    bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        elif arch.startswith('squeezenet'):
            if arch.endswith('1_0'):
                model.features._modules['0'] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))
            elif arch.endswith('1_1'):
                model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))

            final_conv = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
            model.classifier._modules['1'] = final_conv

            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    if m is final_conv:
                        nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    else:
                        nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif arch.startswith('cnn'):
            pass
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)


class CNN(nn.Module):
    def __init__(self, num_class=26, num_char=4):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # batch*3*80*25
            nn.Conv2d(1, 16, (3, 3), padding=(1, 1)),  # batch*3*80*25  (3, 3)
            nn.MaxPool2d(2, 2),  # batch*3*40*13
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # batch*16*40*12
            nn.Conv2d(16, 64, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # batch*64*20*6
            nn.Conv2d(64, 512, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # batch*512*10*3
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # batch*512*5*1
        )
        # self.fc = nn.Linear(512 * 5 * 1, self.num_class * self.num_char)
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Linear(512 * 5 * 1, self.num_class * self.num_char),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512 * 5 * 1)
        # x = self.fc(x)
        x = self.classifier(x)
        return x


class CNNv2(nn.Module):
    def __init__(self, num_class=26, num_char=4):
        super(CNNv2, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            # batch*3*80*25
            nn.Conv2d(1, 16, (3, 3), padding=(1, 1)),  # batch*3*80*25  (3, 3)
            nn.MaxPool2d(2, 2),  # batch*3*40*13
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # batch*16*40*12
            nn.Conv2d(16, 64, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # batch*64*20*6
            nn.Conv2d(64, 512, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # batch*512*10*3
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # batch*512*5*1
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Linear(512 * 5 * 1, self.num_class * self.num_char),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512 * 5 * 1)
        x = self.classifier(x)
        return x
