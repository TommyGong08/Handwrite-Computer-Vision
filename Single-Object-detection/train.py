import os
from datetime import datetime

from net import MyNet
from data import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import torch


class Train:
    def __init__(self, root, weight_path):
        self.summaryWriter = SummaryWriter('logs')

        self.train_dataset = MyDataset(root=root, is_train=True)
        self.test_dataset = MyDataset(root=root, is_Train=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=50, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=50, shuffle=True)

        self.net = MyNet().to('cuda')
        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
        self.opt = optim.Adam(self.net.parameters())
        self.label_loss = nn.BCEWithLogitsLoss()
        self.position_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

        self.train = True
        self.test = True

    def __call__(self):
        index1 = 0
        index2 = 0
        for epoch in range(100):
            if self.train:
                for i, (img, label, position, class_) in enumerate(self.train_loader):
                    print((img, label, position, class_))
                    self.net.train()
                    img, label, position, class_ = img.to('cuda'), label.to('cuda'), position.to('cuda'), class_.to(
                        'cuda')  # 数据放进cuda

                    out_label, out_position, out_class = self.net(img)

                    label_loss = self.label_loss(out_label, label)
                    position_loss = self.position_loss(out_position, position)
                    class_ = class_[torch.where(class_ >= 0)]
                    out_class = class_[torch.where(out_class >= 0)]
                    class_loss = self.class_loss(out_class, class_)

                    train_loss = label_loss + position_loss + class_loss

                    self.opt.zero_grad()
                    train_loss.backward()
                    self.opt.step()

                    if i%10 == 0:
                        print(f'train_loss {i} ====> ', train_loss.item())
                        self.summaryWriter.add_scalar('train_loss', train_loss,index1)
                        index1 +=1

                    date_time =str(datetime.now()).replace(':', '-').replace('.', '-').replace(':', '-')
                    torch.save(self.net.state_dict(), f'param/{date_time}-{epoch}.pt')

            if self.test:
                sum_label_acc = 0
                sum_class_acc = 0
                for i, (img, label, position, class_) in enumerate(self.train_loader):
                    print((img, label, position, class_))
                    self.net.train()
                    img, label, position, class_ = img.to('cuda'), label.to('cuda'), position.to('cuda'), class_.to(
                        'cuda')  # 数据放进cuda

                    out_label, out_position, out_class = self.net(img)

                    label_loss = self.label_loss(out_label, label)
                    position_loss = self.position_loss(out_position, position)
                    class_ = class_[torch.where(class_ >= 0)]
                    out_class = class_[torch.where(out_class >= 0)]
                    class_loss = self.class_loss(out_class, class_)

                    test_loss = label_loss + position_loss + class_loss

                    out_label = torch.tensor(torch.sigmoid(out_label))
                    out_label[torch.where(out_label >= 0.5)] = 1
                    out_label[torch.where(out_label < 0.5)] = 0

                    out_class = torch.argmax(torch.softmax(out_class, dim=1))

                    label_acc = torch.mean(torch.eq(out_label, label).float())
                    class_acc = torch.mean(torch.eq(out_class, class_).float())

                    sum_label_acc += label_acc
                    sum_class_acc += class_acc

                    self.opt.zero_grad()
                    test_loss.backward()
                    self.opt.step()

                    if i % 10 == 0:
                        print(f'train_loss {i} ====> ', test_loss.item())
                        self.summaryWriter.add_scalar('train_loss', test_loss,index1)
                        index2 += 1

                avg_class_acc = sum_class_acc / i
                avg_label_acc = sum_label_acc / i
                print(f'avg_class_acc {epoch}==>>', avg_class_acc)
                print(f'avg_label_acc {epoch}==>>', avg_label_acc)
                self.summaryWriter.add_scalar('avg_sort_acc', avg_class_acc, epoch)
                self.summaryWriter.add_scalar('avg_label_acc', avg_label_acc, epoch)


if __name__ == '__main__':
    train = Train('path/to/your/train_dataset', 'path/to/your/weight.pt')
    train()
