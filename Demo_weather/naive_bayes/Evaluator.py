import math


# THE STRUCTURE OF OUR VALIDATION DATA IS SIMILAR WITH THE TRAINIG DATA. OUR VALIDATION_FEATURES AND VALIDATION_LABLES ARE LISTS THAT WE GET AFTER EXTRACTING OUR VALIDATE DATA SET. THE VALIDATION_FEATURES IS A LIST OF LISTS WHERE EACH INNER LIST REPRESENTS THE FEATURES FOR A SPECIFIC VALIDATION
def validate(validation_features, validation_labels, prior_probability, feature_probability):
    correct_predictions = 0
    total_predictions = len(validation_labels)

    for i, features in enumerate(validation_features):
        true_label = validation_labels[i]
        maximum_probability = float('-inf')
        predicted_label = None

        for label, prior_prob in prior_probability.items():
            likelihood = 1.0
            for feature in features:
                likelihood *= feature_probability[label].get(feature, 0)

            probability = prior_prob * likelihood

            if probability > maximum_probability:
                maximum_probability = probability
                predicted_label = label

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    return accuracy


def test(test_features, test_labels, prior_probability, feature_probability):

    correct_predictions = 0
    total_predictions = len(test_labels)

    for i, features in enumerate(test_features):
        true_label = test_labels[i]
        maximum_probability = float('-inf')
        predicted_label = None

        for label, prior_prob in prior_probability.items():
            likelihood = 0.0
            for feature in features:
                likelihood += math.log(
                    feature_probability[label].get(feature, 0))

            probability = math.log(prior_prob) + likelihood

            if probability > maximum_probability:
                maximum_probability = probability
                predicted_label = label

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    return accuracy
