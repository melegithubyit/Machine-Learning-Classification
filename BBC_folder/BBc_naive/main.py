import os
import timeit
import matplotlib.pyplot as plt
from Naive_Bayes import NaiveBayesClassifier
from Data_loader import load_data
from Feature_extractors import bag_of_words
from Feature_extractors import TF_IDF
from Feature_extractors import word_embeddings


laplace_smoothings_list = [0.5, 1.0, 10, 100]
file_path = os.path.join(os.path.dirname(__file__), 'bbc-text.csv')
split_ratio = 0.8
training_data, testing_data = load_data(file_path, split_ratio)
accuracy_list = []



for laplace_smoothing in laplace_smoothings_list:
        classifier = NaiveBayesClassifier(word_embeddings, laplace_smoothing)
        # classifier = NaiveBayesClassifier(TF_IDF, laplace_smoothing)
        # classifier = NaiveBayesClassifier(word_embeddings ,laplace_smoothing)
        classifier.train(training_data)
        accuracy = classifier.calculate_accuracy(testing_data)
        accuracy_list.append(accuracy)
        
        print(f"for the laplace smoothing values {laplace_smoothing}, Accuracy for prediction is: {accuracy}%\n")
        
        
plt.plot(laplace_smoothings_list, accuracy_list, marker='o')
plt.xlabel('smoothing values')
plt.ylabel('Accuracy')
plt.title('Accuracy vs smoothing values')
plt.grid(True)
plt.show()

