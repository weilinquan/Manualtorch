import numpy as np


def default_collate(batch):
    real_batch = np.array(batch)
    return real_batch
    # elem = batch[0]
    # elem_type = type(elem)
    # if isinstance(elem, torch.Tensor):
    #     return torch.stack(batch, 0)
    # elif elem_type.__module__ == 'numpy':
    #     return default_collate([torch.as_tensor(b) for b in batch])
    # else:
    #     raise NotImplementedError
