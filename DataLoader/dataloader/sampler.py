import numpy as np

class Sampler(object):
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


# 顺序输出索引
class SequentialSampler(Sampler):

    def __init__(self, data_source):
        super(SequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        # 返回迭代器，不然无法 for .. in ..
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


# 随机输出索引
class RandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None):
        super(RandomSampler, self).__init__(data_source)
        # 数据集
        self.data_source = data_source
        # 是否可以重复采样
        self.replacement = replacement
        # 采样长度，一般等于 data_source 长度
        self._num_samples = num_samples

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        # 通过 yield 关键字返回迭代器对象
        if self.replacement:
            # 有放回抽样
            # 之所以按照每次生成32个，可能是因为想减少重复抽样概率 ?
            for _ in range(self.num_samples // 32):
                yield from np.random.randint(low=0, high=n, size=(32,)).tolist()
            yield from np.random.randint(low=0, high=n, size=(self.num_samples % 32,)).tolist()
        else:
            # 无放回抽样
            ans = np.array([i for i in range(n)])
            for _ in range(self.num_samples // n):
                np.random.shuffle(ans)
                yield from ans.tolist()
            np.random.shuffle(ans)
            yield from ans.tolist()[:self.num_samples % n]

    def __len__(self):
        return self.num_samples


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        # 调用 sampler 内部的迭代器对象
        for idx in self.sampler:
            batch.append(idx)
            # 如果已经得到了 batch 个 索引，则可以通过 yield 关键字生成生成器返回，得到迭代器对象
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            # 如果最后的索引数不够一个 batch，则抛弃
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
