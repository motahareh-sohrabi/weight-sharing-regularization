import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim, gamma=0.0):
        super(LinearRegression, self).__init__()
        self.gamma = gamma
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        output = self.linear(x)
        # if self.gamma > 0:
        # regularization_term = self.gamma * torch.norm(self.linear.weight, dim=1)
        # return output - regularization_term
        return output


class FullyConnectedMNIST(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512], output_size=10):
        super(FullyConnectedMNIST, self).__init__()

        layers = [
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], output_size),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape  # must be a list

    def __repr__(self):
        return "Reshape({})".format(self.shape)

    def forward(self, x):
        self.bs = x.size(0)
        return x.view(self.bs, *self.shape)


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode="fan_out")
        module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.uniform_()
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class S_Conv(nn.Module):
    def __init__(self, input_size=[3, 32, 32], ch=150, num_classes=10):
        super(S_Conv, self).__init__()
        self.ch = ch

        self.input_channels = input_size[0]
        self.input_height = input_size[1]
        self.input_width = input_size[2]

        H_out = (self.input_height + 2 * 4 - 9) // 2 + 1
        W_out = (self.input_width + 2 * 4 - 9) // 2 + 1

        self.features = nn.Sequential(
            nn.Conv2d(
                self.input_channels, ch, kernel_size=9, stride=2, padding=4, bias=True
            ),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            Reshape([ch * H_out * W_out]),
            nn.Linear(ch * H_out * W_out, 24 * ch, bias=True),
            nn.BatchNorm1d(24 * ch),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(nn.Linear(24 * ch, num_classes, bias=True))

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def S_FC(input_size=[3, 32, 32], num_classes=10, ch=150):
    layers = []

    input_features = input_size[0] * input_size[1] * input_size[2]

    layers.append(Reshape([input_features]))
    layers.append(nn.Linear(input_features, ch * 16 * 16, bias=True))
    layers.append(nn.BatchNorm1d(ch * 16 * 16))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(ch * 16 * 16, ch * 24, bias=True))
    layers.append(nn.BatchNorm1d(ch * 24))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(ch * 24, num_classes, bias=True))

    model = nn.Sequential(*layers)
    model.apply(initialize_weights)
    return model


class Net_2CNN(nn.Module):
    def __init__(self, alpha1=16, alpha2=32, input_size=[1, 14, 14]):
        super(Net_2CNN, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.input_channels = input_size[0]
        self.input_height = input_size[1]
        self.input_width = input_size[2]

        self.conv1 = nn.Conv2d(
            1, self.alpha1, kernel_size=3, stride=1, padding=1, padding_mode="circular"
        )  # output size = (14-3+2*1)/1 + 1 = 14
        self.bn1 = nn.BatchNorm2d(self.alpha1)

        self.conv2 = nn.Conv2d(
            self.alpha1,
            self.alpha2,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )  # output size = (14-3+2*1)/1 + 1 = 14
        self.bn2 = nn.BatchNorm2d(self.alpha2)

        self.fc1 = nn.Linear(self.alpha2, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # global pooling
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, self.alpha2)
        x = self.fc1(x)
        return x


class Net_2FC(nn.Module):
    def __init__(self, alpha1=16, alpha2=32, input_size=[1, 14, 14]):
        super(Net_2FC, self).__init__()
        self.input_shape = input_size[0] * input_size[1] * input_size[2]
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.fc1 = nn.Linear(self.input_shape, self.input_shape * self.alpha1)
        self.bn1 = nn.BatchNorm1d(self.input_shape * self.alpha1)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(
            self.input_shape * self.alpha1, self.input_shape * self.alpha2
        )
        self.bn2 = nn.BatchNorm1d(self.input_shape * self.alpha2)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(self.alpha2, 10)

    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = x.view(-1, self.alpha2, self.input_shape)
        x = torch.mean(x, dim=-1)
        x = self.fc3(x)

        return x
