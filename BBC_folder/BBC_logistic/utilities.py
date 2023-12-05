from collections import defaultdict
import math
from random import shuffle
from matplotlib import pyplot as plt
from feature_Extractor import calculate_tf



#  USE THE CLEAR DATA METHOD TO GET RID OF THE PUNCTUATIONS WHICH HAS NO CONTRIBUTION IN TRAINING OUR ALGORITHM
def clear_data(dataset):
    processed_data = []
    for sentence, category in dataset:
        sent = ''
        for word in list(sentence):
            if word not in ['.',',','!','?']:
                sent += word
        processed_data.append((sent, category))
    return processed_data
    

def sigmoid(z):
    return 1 / (1 + math.exp(-z))




def train_logistic_regression(dataset, learning_rate):
    # SIMPLE LABEL TO NUMBER CONVERSION. MAPPING THE LABELS TO NUMERICAL VALUES
    label_num = {}
    num_dataset = []
    categories = []
    for sentence, category in dataset:
        if category not in label_num:
            label_num[category] = len(label_num)
            categories.append(category)
        encoded_category = label_num[category]
        num_dataset.append((sentence, encoded_category))

    # INITALIZE WEIGHTS
    num_categories = len(categories)
    weights = {i: 0 for i in range(num_categories)}
    feature_weights = {}

    # ITERATIONS FOR TRAINING 
    num_iterations = 100
    shuffle(num_dataset)
    for _ in range(num_iterations):
        for sentence, category in num_dataset:
            # CALCULATING THE ACTIVATION SCORE
            
            activation_score = weights[category]

            features = calculate_tf(sentence)
            for feature, value in features.items():
                if feature not in feature_weights:
                    feature_weights[feature] = {i: 0 for i in range(num_categories)}
                activation_score += feature_weights[feature][category] * value

            # CALCULATE THE PREDICTED CATEGORY
            predicted_category = sigmoid(activation_score)

            # UPDATE THE WEIGHTS USING GRADIENT ASCEND
            weights[category] += learning_rate * (1 - predicted_category)

            for feature, value in features.items():
                if feature in feature_weights:
                    feature_weights[feature][category] += learning_rate * (1 - predicted_category) * value

    return weights, label_num, categories



def plotter(learning_rates, accuracy_results):
    plt.figure(figsize=(5, 3))
    for i in range(len(learning_rates)):
        plt.plot(accuracy_results[i], marker='*', label=f'LEARNING-RATE = {learning_rates[i]}')
    plt.xlabel('Laplace Smoothing')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.show()