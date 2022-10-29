import numpy as np
from scipy.special import softmax
import cv2
import matplotlib.pyplot as plt

class chek_digit():
    def __init__(self, variables_path, img_path, n_classes,index):
        self.values = np.load(variables_path)
        self.image = img_path
        self.n_classes = n_classes
        self.index = index

    def make_onehot(self, *args, **kwargs):
        n_classes = self.n_classes
        temp = np.zeros((n_classes, ))
        temp[kwargs['pred']]=1
        one_hot = temp[::-1]
        final_pred  = np.argmax(one_hot)
        return final_pred

    def gess_digit(self):
        image = self.image
        my_model = self.values["name3"]
        my_bias = self.values["name4"]
        logit = np.matmul(image.reshape(1, image.shape[0]),my_model) + my_bias
        pred = np.argmax(logit)
        self.final_pred = self.make_onehot(pred=pred)
        return True
 
