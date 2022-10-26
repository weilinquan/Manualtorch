import sys

sys.path.append("..")
# from DataLoader import collate
from DataLoader.dataloader.dataloader import DataLoader
from DataLoader.Dataset import mydataset

test1 = mydataset.MyDataset('../DataLoader/Dataset/mnist_test.csv')
dataloader = DataLoader(test1, batch_size=50, shuffle=True, drop_last=True)
for data in dataloader:
    print(data)
