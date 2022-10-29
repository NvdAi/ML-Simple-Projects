import os
import numpy as np
import matplotlib.pyplot as plt

class load_variables():
    def __init__(self, variables_path):
        self.values = np.load(variables_path) 
        self.load = True

    def section_data(self):
        if self.load:
            train_acc = self.values["name1"]
            test_acc = self.values["name2"]
        return train_acc,test_acc

    def disply (self):
        train_acc, test_acc = self.section_data()
        plt.plot(train_acc, "-o", label="train_acc")
        plt.plot(test_acc, "-o", label="test_acc")
        plt.ylabel("accuracy")
        plt.xlabel("generation")
        plt.xticks(rotation=70)
        plt.xticks(np.arange(0, len(train_acc), 1.0))
        plt.legend()
        plt.show()


if __name__ == "__main__":    
    model_folder = "../MNIST_dataset/best_model"
    full_name_model = os.path.join(model_folder, "finall_winner_model.npz")
    show_diagram = load_variables(full_name_model)
    show_diagram.disply()
