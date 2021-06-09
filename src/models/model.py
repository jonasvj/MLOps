import torch.nn.functional as F
from torch import nn


def conv_dim(input_height, input_width, kernel_size, stride, padding):
    new_height = int((input_height - kernel_size[0] + 2 * padding[0]) / stride[0] + 1)
    new_width = int((input_width - kernel_size[1] + 2 * padding[1]) / stride[1] + 1)

    return (new_height, new_width)


class ImageClassifier(nn.Module):
    def __init__(self, height=28, width=28, channels=1, classes=10, dropout=0.25):
        super(ImageClassifier, self).__init__()
        self.kwargs = {
            "height": height,
            "width": width,
            "channels": channels,
            "classes": classes,
            "dropout": dropout,
        }
        self.width = self.kwargs["width"]
        self.height = self.kwargs["height"]
        self.channels = self.kwargs["channels"]
        self.classes = self.kwargs["classes"]
        self.dropout_rate = self.kwargs["dropout"]

        self.conv_1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=16,
            kernel_size=8,
            stride=1,
            padding=0,
        )
        self.conv_1_dim = conv_dim(
            self.height,
            self.width,
            self.conv_1.kernel_size,
            self.conv_1.stride,
            self.conv_1.padding,
        )

        self.conv_2 = nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.conv_2_dim = conv_dim(
            self.conv_1_dim[0],
            self.conv_1_dim[1],
            self.conv_2.kernel_size,
            self.conv_2.stride,
            self.conv_2.padding,
        )

        self.max_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=1,
            padding=0)
        self.max_pool_dim = conv_dim(
            self.conv_2_dim[0],
            self.conv_2_dim[1],
            (self.max_pool.kernel_size, self.max_pool.kernel_size),
            (self.max_pool.stride, self.max_pool.stride),
            (self.max_pool.padding, self.max_pool.padding),
        )

        self.linear = nn.Linear(
            in_features=self.conv_2.out_channels
            * self.max_pool_dim[0]
            * self.max_pool_dim[1],
            out_features=self.classes,
        )

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.embeddings = None

    def forward(self, x):
        if x.ndim not in [3, 4]:
            raise ValueError('Expected input to be a 3D or 4D tensor')

        x = self.dropout(F.relu(self.conv_1(x)))
        x = self.dropout(F.relu(self.conv_2(x)))
        x = self.dropout(self.max_pool(x))

        self.embeddings = x.view(-1, self.linear.in_features)

        return F.log_softmax(self.linear(self.embeddings), dim=1)
