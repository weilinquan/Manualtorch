import mydataset

# 测试MyDataset类
# 从mnist_test.csv中读取图片和标签信息
test1 = mydataset.MyDataset('mnist_test.csv')
# 测试读取效果
print(test1.my_img)
# 测试getitem
print(test1.__getitem__(0))
# 测试len
print(test1.__len__())
