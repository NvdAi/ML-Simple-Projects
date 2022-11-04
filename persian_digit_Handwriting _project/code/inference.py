import numpy as np
from scipy.special import softmax
import cv2
import matplotlib.pyplot as plt

class chek_digit():
    def __init__(self, variables_path, img_path, n_classes,index):
        self.values = np.load(variables_path)
        self.image = cv2.imread(img_path)
        self.n_classes = n_classes
        self.index = index
        
    
    def resize_image(self):
        src_image = self.image
        src_image = self.image[:,:,0]
        dst_image_height=32
        dst_image_width = 32
        src_image_height = src_image.shape[0]
        src_image_width = src_image.shape[1]
        if src_image_height > dst_image_height or src_image_width > dst_image_width:
            height_scale = dst_image_height / src_image_height
            width_scale = dst_image_width / src_image_width
            scale = min(height_scale, width_scale)
            img = cv2.resize(src=src_image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            img = src_image

        img_height = img.shape[0]
        img_width = img.shape[1]
        dst_image = np.zeros(shape=[dst_image_height, dst_image_width], dtype=np.uint8)
        y_offset = (dst_image_height - img_height) // 2
        x_offset = (dst_image_width - img_width) // 2
        dst_image[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = img
        image = dst_image / 255
        image = np.where(image >= 0.5, 1, 0)
        return image

    def make_onehot(self, *args, **kwargs):
        n_classes = self.n_classes
        temp = np.zeros((n_classes, ))
        temp[kwargs['pred']]=1
        one_hot = temp[::-1]
        final_pred  = np.argmax(one_hot)
        return final_pred

    def gess_digit(self):
        image = self.resize_image()
        my_model = self.values["name3"]
        my_bias = self.values["name4"]
        logit = np.matmul(image.reshape(1, image.shape[0]*image.shape[1]),my_model) + my_bias
        pred = np.argmax(logit)
        self.final_pred = self.make_onehot(pred=pred)
        return True
 
