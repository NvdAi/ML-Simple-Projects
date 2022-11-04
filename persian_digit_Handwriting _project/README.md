# Persian_Digit_Handwriting_Detection

## Dateset:
I work on HODA dataset. Hoda is a large dataset of persian digit handwriting images. This data consist of handwriting of different people in different fonts.

## Code:
1- Run "read_hoda_dataset" script to preparation from "cdb" files to "npz" and also set suitable size for it's images.

```shell
python read_hoda_dataset.py
```

2.1- Run "hoda_model" to train and save the model called "finall_winner_model" in "Hoda_dataset/best_model" directory and also calculate accuracy.

```shell
python hoda_model.py
```

2.2- The "hoda_model" code is a single perceptron model, For each generation in train step i mean loss of each model is calculated on all data and save the best model and for next itreation I generated new model around best model in last generation. I have used the softmax loss function, and finally I got accuracy arond 80%. The best model will be store in best_model directory.


### Some images
train and test accuracy diagram
<img src="Hoda_dataset/Figures/Figure_2.png" width="1800" height="600"> 

outputs
<img src="Hoda_dataset/Figures/Figure_1.png" width="1800" height="600">

<img src="Hoda_dataset/Figures/Figure_3.png" width="1800" height="600">

<img src="Hoda_dataset/Figures/Figure_4.png" width="1800" height="600">

<img src="Hoda_dataset/Figures/Figure_5.png" width="1800" height="600">