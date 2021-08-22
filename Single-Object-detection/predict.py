from net import MyNet
import cv2 as cv
import torch
import os

if __name__ == '__main__':
    img_name = os.listdir(r'path/to/your/img_file')
    for i in img_name:
        img_dir = os.path.join(r'path/to/your/img_file', i)
        img = cv.imread(img_dir)
        print(i.split('.')[2:6])

        cv2.imshow()
