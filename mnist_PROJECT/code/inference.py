# import imp
from statistics import mode
import numpy as np
from scipy.special import softmax
import cv2
import matplotlib.pyplot as plt
import argparse
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from model import CNN

PATH = '../MNIST_dataset'
MEAN = 0.1307
STD = 0.3081
def main(arg):
    model = CNN(1)
    model.load_state_dict(torch.load(arg.model_path, map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load(arg.model_path))
    trans = Compose([ToTensor(), Normalize((MEAN,), (STD,))])
    # load dataset
    test = MNIST(PATH, train=False, download=True, transform=trans)
    # prepare data loaders
    test_dl = DataLoader(test, batch_size=6, shuffle=True)
    inputs, targets = next(iter(test_dl))
    yhat = model(inputs)
    yhat = yhat.cpu().detach().numpy()
    yhat = np.argmax(yhat, axis=1)
    inputs = (inputs * STD) + MEAN

    print("yhat", yhat)
    print("targets", targets)

    for indxyhat, inputs in enumerate(inputs):        
        plt.subplot(2, 3, indxyhat+1)
        plt.tight_layout(h_pad=3 ,w_pad=3)
        plt.title("pred = " + str(yhat[indxyhat]))
        inputs = torch.reshape(inputs, (28, 28, 1))        
        plt.imshow(inputs)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='configuration path')
    args = parser.parse_args()
    main(args)