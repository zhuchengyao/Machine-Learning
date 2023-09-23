import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
from visdom import Visdom

batch_size = 200
learning_rate = 0.01
epochs = 10
viz = Visdom()

train_data = datasets.MNIST(root='./', train=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))
                            ]), download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST(root='./', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,),(0.3081,))
                           ]),
                           download=True)
test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.modul = nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.modul(x)
        return x

device = torch.device('cuda:0')

net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        viz.images(data.view(-1, 1, 28, 28), win='x')
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {},\tProgress: {}/{},\t in percent: {:.6f}%,\t Loss: {:.6f}'.format(
                epoch,
                batch_idx*len(data), len(train_loader.dataset),
                100*len(data)*batch_idx/len(train_loader.dataset),
                loss.item()
            ))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28*28)

        data, target = data.to(device), target.cuda()
        logits = net(data)                      #测试集数据放入forward计算结果
        test_loss += criteon(logits, target).item()

        pred = logits.argmax(dim=1)             #把预测结果弄出来
        # print(target.data)
        label = target.data
        # correct = torch.eq(pred, label).sum()
        correct += pred.eq(target).float().sum().item()     #预测结果和真值做对比，为true则是1，false则是0.把为1的全部相加得到预测正确值的数量. float是把数据转化为浮点数，sum是把这些数字相加，item()是把前面计算的元素取出来.
        # print(correct)

        # correct += pred.eq(target.data).sum()               #正确数据存放在target.data里

    test_loss/= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.6f}, Accuracy:{}/{}({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct /len(test_loader.dataset)
    ))







