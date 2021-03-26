from typing import Callable, Tuple
from typing import List
import numpy as np
from numpy import ndarray

np.random.seed(0)

Function = Callable[[ndarray], ndarray]
ChainFunction = List[Function]


def derivative(func: Function, x: ndarray, delta: float = 0.001) -> ndarray:
    return func(x + delta) - func(x - delta) / (2 * delta)


def square(x: ndarray):
    return np.power(x, 2)


def sigmoid(x: ndarray):
    return 1 / (1 + np.exp(-x))


der = derivative(square, np.array([1, 2, 3]))
print(der)


def nested_functions(functions: ChainFunction, x: ndarray) -> ndarray:
    if len(functions) == 0:
        return np.empty(0)
    if len(functions) == 1:
        return functions[0](x)
    else:
        # f2 (f1(x)) -> [f1, f2], thus f2 (nested (f1))
        return functions[-1](nested_functions(functions[0:-1], x))


chain: ChainFunction = [square, sigmoid]

print(nested_functions(chain, np.array([1, 2, 3])))

print(nested_functions(chain, np.array([[1, 2, 3], [2, 3, 4]])))


class Neuron:
    def __init__(self, inputs: ndarray, expected: ndarray) -> None:
        self.inputs = inputs
        self.weights = np.random.randn(inputs.shape[1])
        self.output = None
        self.output_sum = None
        self.expected = expected

    def add_inputs(self, inputs: ndarray) -> None:
        self.inputs.put(-1, inputs)

    def forward(self) -> None:
        self.output_sum = np.dot(self.inputs, self.weights.T)
        self.output = sigmoid(self.output_sum)

    def backward(self) -> None:
        self.weights = self.weights - 0.001 * np.mean(
            self.inputs.T * derivative(sigmoid, self.output_sum) * 2 * (self.expected - self.output), axis=1)

    def predict(self, inputs: ndarray) -> ndarray:
        output_sum = np.dot(inputs, self.weights.T)
        output = sigmoid(output_sum)
        return output


neuron = Neuron(np.array([[0.5, 0.7, -0.9], [-0.1, -0.4, 1]]), np.array([0, 1]))
print("Neuron working")
print(neuron.inputs)
print(neuron.weights)

for i in range(100):
    print("Output {}: {} W{}".format(i, neuron.output, neuron.weights))
    neuron.forward()
    neuron.backward()

# For now a very simple Perceptron, that gives back 1 if first two negative numbers, last one positive, and viceversa
print(neuron.predict(np.array([-0.5, -0.3, 0.8])))
print(neuron.predict(np.array([0.5, 0.3, -0.8])))


