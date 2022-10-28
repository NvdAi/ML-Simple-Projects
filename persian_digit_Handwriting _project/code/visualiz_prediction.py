import numpy as np
import matplotlib.pyplot as plt
import glob
from inference import chek_digit


def call_inference(img_folder_path, model_path,n_classes):
    all_path = glob.glob(img_folder_path)
    img_path_list = [all_path[np.random.randint(0, len(all_path))] for i in range(9)]
    for index, img in enumerate(img_path_list):
        varable = chek_digit(model_path, img, n_classes,len(img_path_list))
        varable.gess_digit()
        plt.subplot(3, 3, index+1)
        plt.tight_layout(h_pad=1 ,w_pad=1)
        plt.title("pred = " + str(varable.final_pred))
        image = plt.imread(img)
        plt.imshow(image)
    plt.show()

call_inference("../Hoda_dataset/hoda_images/test_images/*", "../Hoda_dataset/best_model/finall_winner_model.npz", 10)