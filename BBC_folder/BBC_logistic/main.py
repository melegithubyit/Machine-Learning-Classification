import os
from Data_loader import data_loader
from utilities import clear_data, train_logistic_regression, plotter
from Evaluator import evaluate_logistic_regression


if __name__ == '__main__':
    # Step 1: READ AND LOAD THE DATA SET
    file_path = os.path.join(os.path.dirname(__file__), 'bbc-text.csv')
    data = data_loader(file_path)


    # MAKING THE DATA GET RID OF PUNCTUATIONS
    print("computing Logical Regression Prediction wait for seconds...")
    dataset = clear_data(data)

    # THE STAGE WHERE WE EVALUATE THE MODEL WITH DIFFERENT TRAINING RATES AND LAPLACE SMOOTHING
    learning_rates = [0.01, 0.1, 10]
    accuracy_results = []

    for learning_rate in learning_rates:
        learning_rate_accuracy = []
        
            # TRINING  OUR MODEL AND PREDICT
        weights, label_encoding, categories = train_logistic_regression(dataset, learning_rate)

            # EVALUATION STAGE OF THE MODEL
        accuracy = evaluate_logistic_regression(dataset, weights, categories)
        learning_rate_accuracy.append(accuracy)

        accuracy_results.append(learning_rate_accuracy)



    # PLOTTING AND PRINTING THE RESULT OF THE PREDICTION
    print("Your accuracy value is: ", accuracy,'%')
    plotter(learning_rates, accuracy_results)

