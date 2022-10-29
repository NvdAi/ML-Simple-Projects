import os
import argparse
import matplotlib.pyplot as plt 
import math
import numpy as np
import random
import cv2 
import glob 
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from scipy.special import softmax
from mlxtend.data import loadlocal_mnist
# from inference import chek_digit
# from variables import load_variables

def make_onehot(o_n, n_classes):
    temp = np.zeros(( n_classes, ))
    temp[int(o_n)]=1
    one_hot = temp[::-1]
    return one_hot

def calculate_mean_loss(model, X, y, bias, n_classes,n_feature):
    model_accuracy = 0
    for label,sample in enumerate(X):
        logit = np.matmul(sample.reshape(1,n_feature ), model) + bias
        prediction = softmax(logit, axis=1)
        o_n = y[label]
        one_hot = make_onehot(o_n, n_classes)
        
        if np.argmax(one_hot) == np.argmax(prediction):
            model_accuracy+=1
        else:
            pass
    model_accuracy = (model_accuracy/X.shape[0])*100
    return model_accuracy
    

def calculate_mean_loss_all_models(models, X, y, biases, n_classes, n_feature):  
    mean_acc_all = []
    for b,mod in enumerate(models):
        bias = biases[b]
        model_accuracy = calculate_mean_loss(mod, X, y, bias, n_classes,n_feature,) 
        mean_acc_all.append(model_accuracy)
    return mean_acc_all                     
 
def get_winner_model(models, biases, mean_acc_all):
    indx_winner = mean_acc_all .index(max(mean_acc_all))
    max_acc = max(mean_acc_all)
    winner_model = models[indx_winner]
    winner_bias = biases[indx_winner]
    return winner_model, winner_bias, max_acc

def create_new_generation(winner_model, winner_bias, n_agent,n_classes, n_feature, first_sigma, sigma_gen):
    models = []
    biases =[]
    t =int(n_agent*20/100)
    s = n_agent-t
    for i in range(s-1):
        temp_m, _ = make_blobs(n_samples=1, centers=winner_model.reshape(1, winner_model.shape[0]*n_classes), cluster_std=sigma_gen)
        models.append(temp_m.reshape(winner_model.shape[0], n_classes))
        temp_l, _ = make_blobs(n_samples=1, centers=winner_bias, cluster_std=sigma_gen)
        biases.append(temp_l)
    for i in range(t):
        model,_ = make_blobs(n_samples=1,centers =np.zeros((1,n_classes*n_feature)) ,cluster_std=first_sigma)
        models.append(model.reshape(n_feature,n_classes))
        bias,_ = make_blobs(n_samples = 1, centers=np.zeros((1,n_classes)),cluster_std=first_sigma)
        biases.append(bias.reshape(1,n_classes))
    models.append(winner_model)
    biases.append(winner_bias)
    return models, biases


def load_pars_data(train_images,train_labels,test_images,test_labels,n_pars):
    X_train, y_train = loadlocal_mnist(train_images, train_labels)
    X_test, y_test = loadlocal_mnist(test_images, test_labels)
    X_train = X_train/256
    X_test  = X_test/256
    n_classes = len(np.unique(y_train))
    n_feature = X_train.shape[1]
    # shuffle
    test_data_shufel = np.hstack((X_test, y_test.reshape(y_test.shape[0],1)))
    test_shuffled = shuffle(test_data_shufel)
    y_test = test_shuffled[:,test_shuffled.shape[1]-1:test_shuffled.shape[1]]
    X_test = test_shuffled[:,0:test_shuffled.shape[1]-1]
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1])
    #pars
    n_pars_train = int(X_train.shape[0]*n_pars/100)
    X_train=X_train[:n_pars_train]
    y_train=y_train[:n_pars_train]
    n_pars_test = int(X_test.shape[0]*n_pars/100)
    X_test = X_test[:n_pars_test,:]
    y_test=y_test[:n_pars_test]
    return X_train, y_train, X_test, y_test, n_classes, n_feature


def main(args):
    
    train_images = "../MNIST_dataset/mnist/train-images-idx3-ubyte"
    train_labels = "../MNIST_dataset/mnist/train-labels-idx1-ubyte"
    test_images = "../MNIST_dataset/mnist/t10k-images-idx3-ubyte"
    test_labels = "../MNIST_dataset/mnist/t10k-labels-idx1-ubyte"

    X_train, y_train, X_test, y_test, n_classes, n_feature = load_pars_data(train_images,train_labels,test_images,test_labels,args.npars)

    n_models = args.nagent
    generation =args.ngen
    first_sigma = 0.5
    sigma_temp = 0.15
    models = []
    biases =[]
    for i in range(n_models):
        model,_ = make_blobs(n_samples=1,centers =np.zeros((1,n_feature*n_classes)) ,cluster_std=first_sigma)
        models.append(model.reshape(n_feature,n_classes))
        bias,_ = make_blobs(n_samples = 1, centers=np.zeros((1,n_classes)),cluster_std=first_sigma)
        biases.append(bias.reshape(1,n_classes))
   
    
    acc_train = []
    acc_test = []
    for i in range(generation):
        print("====================================================================")
        print("Analyzing generation %d ..." % i)
        mean_acc_all= calculate_mean_loss_all_models(models, X_train, y_train, biases, n_classes, n_feature)
        winner_model, winner_bias, max_acc = get_winner_model(models, biases, mean_acc_all)
        acc_train.append(max_acc)
        print("max acc of each genneration train =",max_acc)
        accuracy_test = calculate_mean_loss(winner_model,X_test,y_test,winner_bias,n_classes,n_feature)
        print("wineer modell acc of each genneration test =",accuracy_test)
        acc_test.append(accuracy_test)
        models, biases = create_new_generation(winner_model,winner_bias, n_models,n_classes,n_feature,first_sigma,sigma_temp)

        if i==generation-1:
            model_folder = "../MNIST_dataset/best_model"
            try:
                os.makedirs(model_folder)
            except:
                pass
            full_name_model = os.path.join(model_folder, "finall_winner_model.npz")
            np.savez(full_name_model, name1=acc_train,name2=acc_test,name3=winner_model,name4=winner_bias)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngen', default=10, type=int, help='number of generations')
    parser.add_argument('--nagent', required=True, type=int, help='number of agents in each generation')
    parser.add_argument('--npars', required=True, type=int, help='tran on n% of data')
    args = parser.parse_args()
    main(args)




