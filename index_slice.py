import torch

a = torch.rand(4, 3, 28, 28)

# print(a[0])
print(a[0].shape)

print(a[:2].shape)  #从第1张图到2张图片
print(a[:, :, :])     #可以任意的切，序号从0开始,前闭区间后开区间

print(a[:,:,::2,:]) #如果两个冒号，第二个冒号后面是steps
print(a.index_select(1, torch.tensor([0,2]))) #选择 第一个0指的是选择维度，这里选择的是第一个维度,序号为0，第二个列表指的是你要选择的具体序号，比如这里指的是第一个维度的0号和2号。输入格式为torch.tensor

print(a[...].shape)   #...表示数据维度太多的时候，可以用...省略所有

x = torch.randn(3, 4)
print(x)

mask = x.ge(0.5)    #选择元素大于等于0.5的值，
print(mask)
print(torch.masked_select(x, mask))     #得到矩阵后可以用mask的矩阵再把大的值调出来，变成一个一维tensor
