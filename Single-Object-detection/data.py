from torch.utils.data import Dataset
import os
import cv2 as cv
import torch


class MyDataset(Dataset):
    def __init__(self, root, is_train = True):
        """
        只是添加路径，当需要用到图片时再用getitem取数据
        """
        self.dateset = []
        dir = 'train' if is_train else 'test'
        sub_dir = os.path.join(root, dir)
        img_list = os.listdir(sub_dir)
        for i in img_list:
            img_dir = os.path.join(sub_dir, i)
            self.dateset.append(img_dir)

    def __len__(self):
        return len(self.dateset)

    def __getitem__(self, index):
        data = self.dateset[index]
        img = cv.imread(data)
        print(img.shape)

        # new_img = np.transpose(img, (2,0,1))
        new_img = torch.tensor(img).permute(2, 0, 1)  # 转换成torch类型

        '''
        下面添加label, position和class
        根据你自己的标签格式而定
        '''
        label = 'path/to/your/label'
        position = 'path/to/the/position'
        class_ = 'path/to/the/class'


if __name__ == '__main__':
    MyDataset('Path/to/your/dataset', is_train=True)