import numpy as np
from math import exp


class LogisticRegression():

    def __init__(self, b0, b1, learn_rate, epochs):
        """Init the model parameters
           b0,b1 = weights
           learning rate = rate of variation of the weights between epochs
           epochs = number of iterations to train the model
        """
        self.b0 = b0
        self.b1 = b1
        self.learning_rate = learn_rate
        self.epochs = epochs

    def normalize(x):
        """Normalize the data between 0 and 1
        """
        return X - X.mean()

    def predict(self, data, b0, b1):
        """Function that represents the log of probabilities. 
           The logistic function (also called the sigmoid) is use.
        """
        return np.array([1/(1+exp(-1*b0+(-1*b1*x))) for x in data])
