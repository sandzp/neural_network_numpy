# Numpy-only 2-layer Neural Network for Binary Classification

A 2-layer neural network for binary classification built using Numpy only. 

The neural_net file contains a neural net class with a range of different inbuilt function allowing you to define how many nodes each layer has. Can specify learning rate and number of epochs. 

Uses hard-coded ReLU and sigmoid functions. 

Performs random weight initialization, forward propagation, backwards propagation and cross-entropy loss calculation.

## Performance

The model was tested on 3 datasets with different numbers or features and with binary outcomes:

### 1. UCI Heart Diease Dataset

- Contains data from 270 patients pertaining to 13 clinical variables (age, sex, chest pain, etc.) and weather the patient has heart disease or not.</br> 
- Using a 80/20 train/test split, 500 epochs and a learning rate of 0.001 it achieved train accuracy of 94% and a test accuracy or 68%.

![](https://github.com/sandzp/neural_network_numpy/blob/main/Images/heart_disease_model1.png)

### 2. Banknote Authenticity Dataset

- Contains data from 1372 examples and 4 parameters (variance, skewness, curtosis, entropy) and whether the banknote is real or fake.</br> 
- Using a 80/20 train/test split, 500 epochs and a learning rate of 0.001 it achieved train accuracy of 99% and a test accuracy of 97%.

![](https://github.com/sandzp/neural_network_numpy/blob/main/Images/bank_note_model1.png)

### 3. Sonar Dataset

- Contains data from 208 samples, each containing 59 variables and whether the sonar signal is a rock or a mine.</br>
- Using a 70/30 train/test split, 500 epochs and a learning rate of 0.001 it achieved a train accuracy of 98% and a test accuracy of 85%.

![](https://github.com/sandzp/neural_network_numpy/blob/main/Images/sonar_dataset_model1.png)

Next goals are to expand it to allow for multi-class prediction, batching, different activation functions and  more layers. 
