import torch

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)

#合并两个dim：
a3 = torch.cat([a1, a2], dim=0)
print(a3.shape)

#除了要合并的dim，其他维度数据量不一样会报错

a2 = torch.rand(4,3,32,32)
a4 = torch.stack([a1,a2], dim=0)
print(a4.shape)
#把两个维度数据相同的元素堆叠到一起，形成一个新的，增加了一个维度的张量

b1 = torch.rand(32,8)
b2 = torch.rand(32, 8)
c = torch.stack([b1, b2], dim=0)

c = torch.rand(4, 32, 8)
print(c.shape)
split_1, split_2 = c.split([1, 3], dim=0)
print(split_1.shape)
print(split_2.shape)
#对向量的拆分split    [1,1]表示拆分两个，前面的数量为1，后面的数量为1.也可以拆分多个，[1，2，3]表示拆分成三份，第一份的数量为1，第二份数量为2，第三份数量为3

split_1, split_2, split_3, split_4 = c.split(1, dim=0)
print(split_1.shape)
print(split_2.shape)
#这种语法，split中前面的2表示的是每个单元内拆完后保留的数量。这里第0号dim中共有4个，拆分成4个，每个单元含有1份数据

split_1, split_2 = c.chunk(2, dim=0)
print(split_1.shape)
#chunk中前面数字则是指定拆分完后有多少单元格。这里拆分完后有两个tensor


