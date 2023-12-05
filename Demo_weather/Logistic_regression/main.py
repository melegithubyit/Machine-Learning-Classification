import os
import matplotlib.pyplot as plt
from utilities import train_logistic_regression
from dataLoader import load_data, clear_data
from evaluators import evaluate


# IDENTIFYING THE PATH FOR THE FILS AND PASS FOR THE DATA LOADER
file_path_1 = os.path.join(os.path.dirname(__file__), 'train.csv')
file_path_2 = os.path.join(os.path.dirname(__file__), 'test.csv')
file_path_3 = os.path.join(os.path.dirname(__file__), 'validate.csv')

train_data = clear_data(load_data(file_path_1))
test_data = clear_data(load_data(file_path_2))
validation_data = clear_data(load_data(file_path_3))



# _______SET THE LEARNIG RATES AND THE SMOOTHING VALUES HERE__________
learning_rates = [0.001, 0.01, 0.1]
accuracies = []

for learning_rate in learning_rates:
    weights, bias = train_logistic_regression(train_data, learning_rate)
    accuracy = evaluate(validation_data, weights, bias)
    accuracies.append((learning_rate, accuracy))

# SORTING THE ACCURACY VALUES IN DESCENDING ORDER TO MAKE THE HIGHER ACCURACY AT THE TOP(OPTIONAL)
accuracies.sort(key=lambda x: x[1], reverse=True)


#  PRINTING THE ACCURACIES
for accuracy in accuracies:
    learning_rate, acc = accuracy
    print("Learning Rate -->",learning_rate, "accuracy:",acc,"%")

# SELECTING THE HIGHEST ACCURAACY VALUE
best_learning_rate, best_accuracy = accuracies[0]


#  WE HAVE TRAINED THE MODEL USING THE BEST HPERPARAMETERS ON THE ENTIRE TRAINING DATA
weights, bias = train_logistic_regression(train_data + validation_data, best_learning_rate)

# EVALUATE THE MODEL ON THE TEST DATA
test_accuracy = evaluate(test_data, weights, bias)
print("\nTest Accuracy: ",test_accuracy,"%")

# PLOTTING THE ACCURACY ON DIFFERENT RATE
x = [accuracy[0] for accuracy in accuracies]
y = [accuracy[1] for accuracy in accuracies]

