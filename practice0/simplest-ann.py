# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
import matplotlib.pyplot as plt 

class SimplestAnnPossible:
    """
    This contains the simplest ever possible ANN
    Input(x) ----w----> Output (yp)
    yp = Y predicted
    Notice:
        - There is no activation function, just simple linear output
        - There is no bias
        - There are only 2 "nodes"
        - Only 1 weight
        - Learning rate is fixed
        - There are no batches, or batch = 1
        - etc.
        
        It uses the typical Gradient descent method to update weights.
    """
    
    def __init__(self, iterations):
        self.iterations = iterations
        w_combinations = 1
        self.w = np.random.rand(w_combinations)
        self.learning_rate = 0.01
        
    def set_input_training_values(self, x):
        self.x = np.array(x)
        
    def set_output_training_values(self, y):
        self.y = np.array(y)
        
    def register_training_info(self, i, yp, c):
        self.training_info[0].append(self.x[i][0])
        self.training_info[1].append(self.y[i][0])
        self.training_info[2].append(yp[0])
        self.training_info[3].append(c[0])
        
    def train(self):
        self.training_info = [[],[],[],[]]
        for j in range(self.iterations):
            print ("Iteration {}".format(j))
            for i in range(self.x.shape[0]):
                yp = self.w[0]*self.x[i] #y predicted
                c = np.exp2(yp - self.y[i]) #cost
                dcdyp = 2*(yp - self.y[i]) #derivative cost with respect y predicted
                dypdw = self.x[i] #derivative y pred with respect w
                dcdw = dcdyp * dypdw
                print("y = {0} , y_pred = {1}, error = {2}, x = {3}, w = {4}, dcdyp = {5}, dypdw = {6}, dcdw = {7}"
                      .format(self.y[i], yp, c, self.x[i], self.w[0], dcdyp,
                              dypdw, dcdw))
                self.register_training_info(i, yp, c)
                self.w[0] = self.w[0] - (self.learning_rate * dcdw) #redefining the weight
                
    def predict(self, input_data):
        return input_data * self.w[0]
    
    def plot(self):
        plt.plot(self.training_info[0], self.training_info[3], label= "error")

        plt.legend()
        plt.show()
        
        plt.plot(self.training_info[0], self.training_info[2], label= "predicted")
        plt.plot(self.x, self.y, label= "real")
        
        plt.legend()
        plt.show()
        

model = SimplestAnnPossible(10)
model.set_input_training_values([[1], [3], [5]])
model.set_output_training_values([[2], [6], [10]])

model.train()

model.predict(124)