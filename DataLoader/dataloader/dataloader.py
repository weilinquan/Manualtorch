from .collate import default_collate
from .sampler import BatchSampler, SequentialSampler, RandomSampler


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, collate_fn=None, drop_last=False):
        self.dataset = dataset

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # 一旦设置了batch_sampler，那么batch_size、shuffle、sampler和drop_last四个参数就不能传入
            # 因为这4个参数功能和batch_sampler功能冲突了
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False

        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        # 未设置batch_sampler则采用默认类
        if batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = iter(batch_sampler)

        if collate_fn is None:
            collate_fn = default_collate
        self.collate_fn = collate_fn

    def __next__(self):
        index = next(self.batch_sampler)
        data = [self.dataset[idx] for idx in index]
        data = self.collate_fn(data)
        return data

    def __iter__(self):
        return self
