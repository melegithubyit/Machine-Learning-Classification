import math
import random



#  SOME TIME MATH OVERFLOW IS HAPPENING THEREFORE WE TRIED TO HANDLE THE SIGN OF ACTIVATION VALUE FOR NEGATIVE INPUT AND POSITIVE INPUT
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return 1 / (1 + math.exp(x))


def train_logistic_regression(train_data, learning_rate):
    num_features = len(train_data[0]) - 1
    weights = []
    for i in range(num_features):
        rand_weight = random.uniform(-0.5, 0.5)
        weights.append(rand_weight)
    bias = random.uniform(-0.5, 0.5)

#  YOU CAN CHANGE THE NUMNER OF ITERATIONS AND HOE THE ALGORITHM ACTS
    num_iterations = 1000
    for _ in range(num_iterations):
        for row in train_data:
            inputs = row[:-1]
            label = row[-1]
            z = bias
            for i in range(num_features):
                z += weights[i] * inputs[i]
            predicted = sigmoid(z)

            error = label - predicted
            bias += learning_rate * error

            for i in range(num_features):
                weights[i] += learning_rate * error * inputs[i]

    return weights, bias



def predict(row, weights, bias):
    z = bias
    for i in range(len(row)):
        z += weights[i] * row[i]     
    predicted = sigmoid(z)
    return predicted
