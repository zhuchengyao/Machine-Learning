import torch

a = torch.rand([8])
b = a.view(2, 4).float()
c = a.view(2, 2, 2).float()
print(b)
print(c)

# d = a.norm()
# print(d)
# print(c.norm(2))
# print(c.norm(2, dim=1))       #目前这块有问题

e = torch.randn(4, 10)
print(e.max())
print(e.mean())
print(e.argmax(dim=1)) #在维度为1的地方寻找最大值的下标

a1 = b.max(dim=1)
print(a1)
a2 = b.max(dim=1, keepdim=True)
print(a2)
a3 = b.topk(2, dim=1)
print(a3)   #返回概率最大的k个值,指定维度. 如果 largest = False，这样设置就是找到概率最小的k个