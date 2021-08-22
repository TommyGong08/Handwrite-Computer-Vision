from torch import nn
import torch


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 11, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(11, 22, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(22, 32, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
        )

        # 二分类标签
        self.label_layer = nn.Sequential(
            nn.Conv2d(128, 1, 19),
            nn.ReLU()
        )

        # 回归检测框
        self.position_layer = nn.Sequential(
            nn.Conv2d(128, 4, 19),
            nn.ReLU()
        )

        # 多分类回归
        self.class_layer = nn.Sequential(
            nn.Conv2d(128, 20, 19),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layers(x)
        label_ = self.label_layer(out)
        label_ = torch.squeeze(label_, dim=2)
        label_ = torch.squeeze(label_, dim=2)
        label_ = torch.squeeze(label_, dim=1)
        position_ = self.position_layer(out)
        position_ = torch.squeeze(position_, dim=2)
        position_ = torch.squeeze(position_, dim=2)
        class_ = self.class_layer(out)
        class_ = torch.squeeze(class_, dim=2)
        class_ = torch.squeeze(class_, dim=2)
        return label_, position_, class_


if __name__ == '__main__':
    net = MyNet()
    x = torch.randn(3, 3, 300, 300)   # 第一个数是batch，3通道, 300*300的图片
    print(net(x)[0].shape)
    print(net(x)[1].shape)
    print(net(x)[2].shape)