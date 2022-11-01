from numpy import vstack
from numpy import argmax
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch
from tqdm import tqdm
import os

from model import CNN
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cuda:0'
print("model.to", device)


 
# prepare the dataset
def prepare_data(path):
    # define standardization
    trans = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    # load dataset
    train = MNIST(path, train=True, download=True, transform=trans)
    test = MNIST(path, train=False, download=True, transform=trans)
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=64, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl
 

class Model():
    def __init__(self):
        self.build()
        self.ckpt_dir = "../checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # Writer will output to ./runs/ directory by default
        self.writer = SummaryWriter()

    def build(self):
        self.model = CNN(1)
        # model.summary()
        self.model.to(device)

        # define the optimization
        self.criterion = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)


    def train_one_epoch(self, train_dl):
        mean_loss = 0.0
        desc = f"mean_loss={mean_loss:.5}"
        p = tqdm(train_dl, desc=desc)
        for i, (inputs, targets) in enumerate(p):
            # clear the gradients
            self.optimizer.zero_grad()
            # compute the model output
            inputs = inputs.to(device)
            targets = targets.to(device)
            yhat = self.model(inputs)
            # calculate loss
            loss = self.criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            self.optimizer.step()

            mean_loss += loss.item()
            desc = f"mean_loss={mean_loss / (i+1):.5}"
            p.set_description(desc=desc)
        return mean_loss / len(train_dl)

    # train the model
    def train(self, train_dl):
        # enumerate epochs
        for epoch in range(10):
            # enumerate mini batches
            loss_value = self.train_one_epoch(train_dl)
            self.writer.add_scalar('Loss/train', loss_value, epoch)


    # evaluate the model
    def evaluate(self, test_dl):
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(test_dl):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # evaluate the model on the test set
            yhat = self.model(inputs)
            # retrieve numpy array
            yhat = yhat.cpu().detach().numpy()
            actual = targets.cpu().numpy()
            # convert to class labels
            yhat = argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "model.pt"))

# prepare the data
path = '../MNIST_dataset'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
# # train the model
model = Model()
# train the model
model.train(train_dl)
# evaluate the model
acc = model.evaluate(test_dl)
model.save()

print('Accuracy: %.3f' % acc)