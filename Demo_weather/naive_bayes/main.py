from dataloader import load_data
from Utilities import featureize,train
from Evaluator import test, validate
import os


# FINIDING THE PATH FOR OUR TRAINING, TESTING AND VALIDATING FILES.
file_path_1 = os.path.join(os.path.dirname(__file__), 'train.csv')
file_path_2 = os.path.join(os.path.dirname(__file__), 'test.csv')
file_path_3 = os.path.join(os.path.dirname(__file__), 'validate.csv')


# FINDING THE RESPECTIVE FEATURES AND LABELS FROM THE FILES IN THE FORM OF LIST OF LISTS FOR THE FEATURES AND IN THE FORM OF LIST FOR THE LABELS
training_features, training_labels = load_data(file_path_1)
test_features, test_labels = load_data(file_path_2)
validation_features, validation_labels = load_data(file_path_3)


# FINDING THE TRANSFORMED FEATURES AND LABELS FOR THE TRAINING DATA AND APPEND THEM WITH THEIR RESPECTIVR LIST. WE SET IT AS FEATURE_TRAIN_X AND FEATURE_TRAIN_Y
feature_train_x, feature_train_y = [], []
for x, y in zip(training_features, training_labels):
    feature_x, label_y = featureize(x, y)
    feature_train_x.append(feature_x)
    feature_train_y.append(label_y)
    

# FINDING THE TRANSFORMED FEATURES AND LABELS FOR THE VALIDATING DATA AND APPEND THEM WITH THEIR RESPECTIVR LIST. WE SET IT AS FEATURE_VALID_X AND FEATURE_VALID_Y
feature_valid_x, feature_valid_y = [], []
for x, y in zip(validation_features, validation_labels):
    feature_x, label_y = featureize(x, y)
    feature_valid_x.append(feature_x)
    feature_valid_y.append(label_y)


# FINDING THE TRANSFORMED FEATURES AND LABELS FOR THE TESTING DATA AND APPEND THEM WITH THEIR RESPECTIVR LIST. WE SET IT AS FEATURE_TEST_X AND FEATURE_TEST_Y
feature_test_x, feature_test_y = [], []
for x, y in zip(test_features, test_labels):
    feature_x, label_y = featureize(x, y)
    feature_test_x.append(feature_x)
    feature_test_y.append(label_y)


# LIST OF LAPLACE SMOOTHING THAT WE CAN CHECK TH PERFORMANCE OF THE TRAING FOR DIFFERENT VALUES
laplace_smoothing = [0.1, 0.5, 1.0]
best_accuracy = 0.0
best_model = None


for smoothing in laplace_smoothing:
    prior_probability, feature_probability = train(feature_train_x, feature_train_y, smoothing)
    validation_accuracy = validate(feature_valid_x, feature_valid_y, prior_probability, feature_probability)


if validation_accuracy > best_accuracy:
    best_accuracy = validation_accuracy
    best_model = (prior_probability, feature_probability)

test_accuracy = test(feature_test_x, feature_test_y, best_model[0], best_model[1])
print('Validation Accuracy :', int(best_accuracy), '%')
print('Test Accuracy: ', int(test_accuracy), '%')
