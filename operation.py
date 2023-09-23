import torch

a = torch.rand(3,4)
b = torch.rand(4)
print(a)
print(b)

d = torch.add(a, b)
c = a+b
print(torch.all(torch.eq(c, d))) #all表示检查所有位置
print(c) #加法是把元素每一个维度都加上子维度的数据,有两种写法，一种直接加a+b。另一种add

#四则运算 + - * / 都是对应位置的计算。要与之区别的是矩阵的乘法, a@b, torch.mm, torch.matmul都是相同的. 在计算维度大于2的情况，只能用matmul，且它的计算仅仅只算后面两个维度的矩阵相乘
a1 = torch.rand(3,4)
a2 = torch.rand(4,3)
a3 = torch.mm(a1, a2)
a4 = torch.matmul(a1, a2)
a5 = a1@a2
print(torch.all(torch.eq(a3, a4)))
print(torch.all(torch.eq(a3, a5)))
print(a3)


b1 = torch.full([2, 2], 3)
b2 = b1**2  #平方
print(b2)
b3 = b2**0.5    #开方
print(b3)
b4 = b2.sqrt()
print(b4)   #开方
b5 = b2.rsqrt() #开放取倒数
print(b5)


c1 = torch.full([2, 2],1)
c2 = torch.exp(c1)  #以e为底取次方
print(c2)
c3 = torch.log(c2)  #以e为底取log
print(c3)

d1 = torch.tensor(3.14)
d2 = torch.floor(d1)  #取不超过的最大整数
print(d2)
d3 = torch.ceil(d1) # 取超过它的最小整数
print(d3)
d4 = torch.trunc(d1)    #取整数部分
d5 = torch.frac(d1)     #取小数部分

#clamp裁减
grad = torch.rand(2, 3) * 15
print(grad.max())
print(grad.median())
print(grad.clamp(10))   #grad的值小于10的全部裁剪为10
print(grad.clamp(1,10)) #grad的值超不在1到10之间的全部裁剪到1和10

