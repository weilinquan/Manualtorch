class Dataset(object):
    # 实现 __getitem__ 方法
    def __getitem__(self, index):
        raise NotImplementedError
    # 获取数据集长度
    def __len__(self):
        raise NotImplementedError