import torch
import numpy as np


# 标量：[]，  一维向量：[2] 这里的2是数据，指的是里面的数据量， 二维向量 [2,3] 是2×3的二维向量

a = np.array([2, 3.3])
a = torch.from_numpy(a)
print(a)

b = np.ones([2, 3])
b = torch.from_numpy(b)
print(b)

c = torch.tensor([2.2, 3.2])
print(c)                                    #创建tensor

d = torch.tensor([[2.2, 3.3], [5.1, 4.4]])
print(d)

e = torch.Tensor(2,3,1)
print(e)    #大写的时候，输出的是维度，也就是2行3列1纵深

f = torch.rand(3, 3)
print(f)        #生成3×3矩阵，数据从0到1的随机数，不会报错

e = torch.rand_like(e)
print(e)        #把e读出来，再创建一个e的格式类型的tensor，赋值给e覆盖

g = torch.randint(1, 10, [2, 3])
print(g)        #自定义最大最小值，然后放到一个[2×3]tensor中去

h = torch.full([2, 3], 10)
print(h)    #所有的数据全部赋值为同一个值

i = torch.arange(0, 10, 2)
print(i)    #生成0-10，以2递增的tensor，数据量由上述信息计算出来

j = torch.linspace(0, 10, steps=4)
print(j)    #生成0-10的4等分数据

k = torch.logspace(0, -1, steps=10)
print(k)    #生成10的0次方到10的-1次方的10等分数据

l = torch.eye(3, 4)
print(l)    #生成对角为1，其余全为0的矩阵，可以不是方阵，如果只输入一个参数，那么就为方阵

m = torch.full([2, 2, 2], 5)
print(m)    #创建一个2×2×2的，数据全是5的tensor