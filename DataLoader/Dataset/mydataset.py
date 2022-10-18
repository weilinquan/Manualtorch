from dataset import Dataset
import numpy as np


class MyDataset(Dataset):
    # 重写dataset类
    # 实现了 __getitem__ 方法
    def __init__(self, address):
        # 读取数据集
        self.my_img = np.loadtxt(address, delimiter=',')

    def __getitem__(self, index):
        # 根据标签返回图片和标签信息
        return self.my_img[index]

    def __len__(self):
        # 获取数据集长度
        return self.my_img.shape[0]

