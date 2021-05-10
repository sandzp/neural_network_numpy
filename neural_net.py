import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

class NeuralNet(object):
    '''
    Two layer neural network
    '''

    def __init__(self, layers = [13, 8, 1], learning_rate = 0.001, epochs = 500):
        self.params = {}
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None

    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1) # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1])
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2])

    def relu(self, Z):
        '''
        ReLU activation function, threshold operation where values < 0 = 0
        '''
        return np.maximum(0,Z)

    def sigmoid(self, Z):
        '''
        Sigmoid activation function
        '''
        return 1/(1 + np.exp(-Z))

    def eta(self, x):
        '''
        Function to handle 0 values
        '''
        ETA = 0.0000000001
        return np.maximum(x, ETA)

    def cross_entropy(self, y, yhat):
        '''
        Cross entropy loss function guaranteed to converge
        '''
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        yinv = 1.0 - y
        yhat = self.eta(yhat) # clips values to avoid NaN's in log function
        yhat_inv = self.eta(yhat_inv)
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((yinv), np.log(yhat_inv))))
        return loss

    def forward_propagation(self):
        '''
        1. Compute weighted sum between input and first layer's weights and then adds the bias: Z1 = (W1*X) + b
        2. Pass result through ReLU activation function: A1 = ReLU(Z1)
        3. Compute output function by passing the result through the sigmoid function: A2 = sigmoid(Z2)
        4. Compute the loss between the predicted output and the true labels: loss(A2, Y)
        '''
        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.cross_entropy(self.y, yhat)

        # Save calculated parameters:
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1
        return yhat, loss

    def drelu(self, Z):
        '''
        Derivative of Relu function
        '''
        Z[Z <= 0] = 0
        Z[Z > 0] = 1
        return Z

    def backward_propagation(self, yhat):
        '''
        Calculates the derivatives of respective functions and updates the weights
        '''
        y_inv = 1 - self.y
        yhat_inv = 1 - yhat

        dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(self.y, self.eta(yhat))
        dl_wrt_sig = yhat * yhat_inv
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis = 0, keepdims = True)

        dl_wrt_z1 = dl_wrt_A1 * self.drelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis = 0, keepdims = True)

        #update the weights and biases
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

        return

    def fit(self, X, y):
        '''
        Trains the neural network using specified data + labels
        '''
        self.X = X
        self.y = y
        self.init_weights() # initialize weights and biases

        #Train neural network using certain number of epochs
        for i in range(self.epochs):
            yhat, loss = self.forward_propagation()
            self.backward_propagation(yhat)
            self.loss.append(loss)
            print("Training epoch {}, calculated loss: {}".format(i+1, loss))

        return

    def predict(self, X):
        '''
        Makes predictions on test dataset
        '''
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)

    def accuracy(self, y, yhat):
        '''
        Calculates the accuracy between predicted value and true labels
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self, imagepath = None):
        '''
        Plots a loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve for Training")

        if imagepath != None:
            plt.savefig(fname=imagepath)

        plt.show()

        return
