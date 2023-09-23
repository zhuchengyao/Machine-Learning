import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5,0.5],std=[0.5,0.5])])  #define the transform function

data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                           train=False)               #get MINST dataset

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                                shuffle=True)        #load the MINST dataset


# the project says we need use knn to classify the data. I would like to use CNN to do it first.




model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

model.to(device=)

torch.save(model.state_dict(), "model_parameter.pkl")



