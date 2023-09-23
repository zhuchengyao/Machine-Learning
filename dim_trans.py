import torch

a = torch.rand(4,1,28,28)
print(a.shape)

b = a.view(4, 1*28*28)
print(b.shape)

c = a.view(4*1*28,28)
print(c.shape)

d = a.view(4,1*28,28)
print(d.shape) #view函数可以合并任意的维度，但是合并维度需要有意义

e = a.unsqueeze(-1)
print(e.shape)  #在最后的位置插入一个维度还可以用其他数字，比如-3，-5

f = a.unsqueeze(0)
print(f.shape)  #在第一个位置插入一个维度位置，可以任选

g = a.squeeze()
print(g.shape) #删除所有为1的维度

bb = torch.rand(1, 32, 1, 1)
print(bb.shape)
h = bb.expand(4, 32, 14, 14)    #复制原来为1的dimension，只有原来为1的dim才能够复制,这里复制32会报错
print(h)
print(h.shape)

i = bb.expand(2, -1, -1, -1)
print(i.shape)  #保持原来的维度不变，用-1即可

j = bb.repeat(4, 32, 1, 1)
print(j.shape)  #把各个对应位置的维度重复对应填入数据的次数，第一个数据本身1，重复4次，变成4，以此类推

cc = torch.rand(3,4)
cc = cc.t()
print(cc.shape)     #二维矩阵，可以用.t()方法来做转置

k = bb.transpose(1,3)
print(k.shape)  #交换1和3的维度信息
