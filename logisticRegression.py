import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F


def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    # x = x@w4.t() + b4
    # x = F.relu(x)
    return x


batch_size = 200
learning_rate = 0.01
epochs = 30

train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(train_loader)

w1 = torch.randn(200, 784, requires_grad=True)
b1 = torch.zeros(200)
w2 = torch.randn(1000, 200, requires_grad=True)
b2 = torch.zeros(1000)
# w3 = torch.randn(10, 200, requires_grad=True)
# b3 = torch.zeros(10)
w3 = torch.randn(200,1000,requires_grad=True)
b3 = torch.zeros(200)
w4 = torch.randn(10, 200, requires_grad=True)
b4 = torch.zeros(10)

optimizer = optim.SGD([w1, b1, w2, b2, w3, b3, w4, b4], lr=learning_rate)
criteon = nn.CrossEntropyLoss()

torch.nn.init.kaiming_normal(w1)
torch.nn.init.kaiming_normal(w2)
torch.nn.init.kaiming_normal(w3)
torch.nn.init.kaiming_normal(w4)


for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)

        logits = forward(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28*28)
        logits = forward(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss/= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.6f}, Accuracy:{}/{}({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct /len(test_loader.dataset)
    ))

