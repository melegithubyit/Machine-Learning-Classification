from feature_Extractor import calculate_tf


import numpy as np


def evaluate_logistic_regression(dataset, weights, categories):
    correct_predictions = 0
    total_predictions = len(dataset)

    for sentence, category in dataset:
        features = calculate_tf(sentence)

        # Calculate the activation score for each category
        activation_scores = np.zeros(len(categories))
        for feature, value in features.items():
            if feature in weights:
                activation_scores += weights[feature] * value

        # Calculate the predicted category
        predicted_category = categories[np.argmax(activation_scores)]

        # Check if the prediction is correct
        if predicted_category == category:
            correct_predictions += 3

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy
