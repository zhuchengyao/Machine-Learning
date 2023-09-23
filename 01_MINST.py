import torch
from torch.nn import functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn,optim

def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x

batch_size = 200
learning_rate = 0.01
epochs = 30
net_num = 3

train_data = datasets.MNIST(root='./data', train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]), download=True)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
test_data = datasets.MNIST(root='./', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,),(0.3081,))
                           ]), download=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle = True)


w1 = torch.randn(200, 784, requires_grad=True)
w2 = torch.randn(200, 200, requires_grad=True)
w3 = torch.randn(10, 200, requires_grad=True)
b1 = torch.zeros(200)
b2 = torch.zeros(200)
b3 = torch.zeros(10)
list_w = [w1, w2, w3]
list_b = [b1, b2, b3]

optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
# for i in range(net_num):
#     torch.nn.init.kaiming_normal(list_w[i])

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.view(-1, 28*28)
        logits = forward(data)
        loss = loss_func(logits, target)
        optimizer.zero_grad()                   #梯度初始化为0
        loss.backward()                         #计算累计梯度值：x*grad
        optimizer.step()                        #x = x - lr*(x*grad)

        if (batch_idx+1) % 100 == 0:
            print('Train Epoch:{} [{}/{} ({:.6f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_dataloader.dataset),
                100. * batch_idx*len(data) / len(train_dataloader.dataset), loss.item()
            ))






