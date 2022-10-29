import numpy as np
import matplotlib.pyplot as plt
import glob
from inference import chek_digit
from mlxtend.data import loadlocal_mnist


def call_inference(img_test, label_test, model_path, n_classes):
    X_test,_ = loadlocal_mnist(img_test, label_test)
    images = [X_test[np.random.randint(0, len(X_test))] for i in range(9)]
    for index, img in enumerate(images):
        varable = chek_digit(model_path, img, n_classes,len(images))
        varable.gess_digit()
        plt.subplot(3, 3, index+1)
        plt.tight_layout(h_pad=1 ,w_pad=1)
        plt.title("pred = " + str(varable.final_pred))
        plt.imshow(img.reshape(28,28))
    plt.show()

img_test = "../MNIST_dataset/mnist/t10k-images-idx3-ubyte"
label_test = "../MNIST_dataset/mnist/t10k-labels-idx1-ubyte"
model_path = "../MNIST_dataset/best_model/finall_winner_model.npz"
call_inference(img_test, label_test, model_path, 10)