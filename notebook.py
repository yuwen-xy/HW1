#!/usr/bin/env python
# coding:utf-8
from abc import ABCMeta

import torch
from torch import nn, optim, utils
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import os

batch_size = 128
learning_rate = 0.001
num_epoches = 100
DOWNLOAD_MNIST = False  # dataset download

# MNIST digits dataset
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

# train data
train_dataset = datasets.MNIST(root='./mnist',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=DOWNLOAD_MNIST
                               )
# test data
test_dataset = datasets.MNIST(root='./mnist',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=DOWNLOAD_MNIST
                              )

train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Network(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = Network(28 * 28, 256, 256, 10)  # in_dim, n_hidden_1, n_hidden_2, out_dim

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    # train
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)

        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        #forward
        out = model(img)
        loss = criterion(out, label)
        #backward
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t_loss = running_loss / (len(train_dataset))
    t_acc = running_acc / len(train_dataset)
    if epoch + 1 == 1:
        train_loss = [t_loss]
        train_acc = [t_acc]
    else:
        train_loss.append(t_loss)
        train_acc.append(t_acc)
    print('Finish {} epoch, Loss:{:.6f}, Acc:{:.6f}'.format(epoch + 1, t_loss, t_acc))

# test
model.eval()
eval_loss = 0.
eval_acc = 0.
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)

    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()

e_loss = eval_loss / (len(test_dataset))
e_acc = eval_acc / len(test_dataset)
print('Test, Loss:{:.6f}, Acc:{:.6f}'.format(e_loss, e_acc))

plt.plot(range(1,num_epoches+1), train_loss, 'b', label='Train_Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.plot()
plt.show()
