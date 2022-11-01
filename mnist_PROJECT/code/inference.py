import imp
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
    print(f"model_path: {arg.model_path}")

    model = CNN(1)
    model.load_state_dict(torch.load(arg.model_path))

    trans = Compose([ToTensor(), Normalize((MEAN,), (STD,))])
    # load dataset
    test = MNIST(PATH, train=False, download=True, transform=trans)
    # prepare data loaders
    test_dl = DataLoader(test, batch_size=6, shuffle=True)

    # for i, (inputs, targets) in enumerate(test_dl):
    inputs, targets = next(iter(test_dl))
    yhat = model(inputs)
    yhat = yhat.cpu().detach().numpy()
    yhat = np.argmax(yhat, axis=1)
    print("yhat", yhat)
    print("targets", targets)

    inputs = (inputs * STD) + MEAN
    print("inputs", inputs.shape)
    plt.imshow(inputs[0,0,:,:])
    plt.show()
    # cv2.imshow("a", inputs.numpy()[0,0,:,:])
    # cv2.waitKey(0)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='configuration path')
    # parser.add_argument('--img_path', type=int, help='Number of data loading workers', default=4)
    args = parser.parse_args()

    main(args)