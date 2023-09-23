import torch
from torch.nn import functional as F

a = torch.linspace(-1,1,10)

b = torch.sigmoid(a)
print(b)
c = torch.tanh(a)
print(c)

d = torch.relu(a)
print(d)

x = torch.ones(1)
w = torch.full([1], 2.0)
mse = F.mse_loss(torch.ones(1), x*w)
print(mse)
w.requires_grad_()
mse = F.mse_loss(torch.ones(1), x*w)
print(mse.backward())
print(w.grad)

#torch.autograd.grad(loss, [w1, w2, w3]) #这里是手动求解loss对w1偏微分，loss对w2偏微分...
#loss.backward() #也是求解loss对w1,w2等的偏微分，但是是附在w1...后面，调用：w1.grad

a = torch.rand(3)
a.requires_grad_()
print(a)

p = F.softmax(a, dim=0)
print(p)