import sys

import numpy as np
from DataLoader.Dataset.DealDataset import DealDataset

sys.path.append("..")
# from DataLoader import collate
from DataLoader.dataloader.dataloader import DataLoader
from DataLoader.Dataset import mydataset

trainDataset = DealDataset('../DataLoader/MNIST_data/', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
testDataset = DealDataset('../DataLoader/MNIST_data/', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
dataloader = DataLoader(trainDataset, batch_size=100, shuffle=False)

# test_loader = dataloader.DataLoader(
#     dataset=testDataset,
#     batch_size=100,
#     shuffle=False,
# )
for data in dataloader:
    print(data)
