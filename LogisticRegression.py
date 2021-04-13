import numpy as np
from math import exp


class LogisticRegression():

    def __init__(self, b0, b1, learn_rate, epochs, X, Y):
        """Init the model parameters
           b0,b1 = weights
           learning rate = rate of variation of the weights between epochs
           epochs = number of iterations to train the model
        """
        self.b0 = b0
        self.b1 = b1
        self.learning_rate = learn_rate
        self.epochs = epochs
        self.accuracy = 0
        self.GD_method(X, Y)

    def normalize(self, X):
        """Normalize the data between 0 and 1
        """
        return X - X.mean()

    def logistic_function(self, data, b0, b1):
        """Function that represents the log of probabilities. 
           The logistic function (also called the sigmoid) is use.
        """
        return np.array([1/(1+exp(-1*b0+(-1*b1*x))) for x in data])

    def cost_derivative(self, y_pred, Y, X):
        """Cost Function partial derivatives to find the minimum values 
        for the L2 loss function
        """
        # Derivative of loss wrt b0
        D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))
        # Derivative of loss wrt b1
        D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))
        return D_b0, D_b1

    def GD_method(self, X, Y):
        """Function that contains the algorithm to train and update
        the wights for the logistic regression model. Where X contains the 
        dataset samples and Y contains it's correspondent values.
        In each epoch the weights will be updated to improve the model performance
        """
        X = self.normalize(X)
        for epoch in range(self.epochs):
            y_pred = self.logistic_function(X, self.b0, self.b1)
            D_b0, D_b1 = self.cost_derivative(y_pred, Y, X)
            # Update Weights
            self.b0 = self.b0 - self.learning_rate * D_b0
            self.b1 = self.b1 - self.learning_rate * D_b1
        y_pred = self.predict(X)
        self.evaluate(y_pred, Y)

    def predict(self, x_test):
        """Predict values after the model is trained.
            Return a vector with a binary value for each prediction
        """
        x_test_norm = self.normalize(x_test)
        y_pred = self.logistic_function(x_test_norm, self.b0, self.b1)
        return np.array([1 if p >= 0.5 else 0 for p in y_pred])

    def evaluate(self, y_pred, y_test):
        """Evalute the model accuracy
        """
        for i in range(len(y_pred)):
            if y_pred[i] == y_test.iloc[i]:
                self.accuracy += 1
        self.accuracy = (self.accuracy/len(y_pred))
