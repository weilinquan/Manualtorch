from .dataset import Dataset
import numpy as np


class SimpleV1Dataset(Dataset):
    def __init__(self):
        # 伪造数据
        self.imgs = np.arange(0, 16).reshape(8, 2)

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return self.imgs.shape[0]
