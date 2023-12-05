import numpy as np
from mnist import MNIST
from random import sample


class NaiveBayes:

    def __init__(self, smoothing=0.01, offset=5000):
        self.smoothing = smoothing
        self.varOffset = offset*smoothing

    def fit2(self, X, y):
        
        samples = len(X)
        self._classes = np.unique(y)
        classes = len(self._classes)
        self.pmf = np.zeros(classes)
        self._priors = np.zeros(classes)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self.pmf[idx] = np.mean(
                [x.size - np.count_nonzero(x) for x in X_c]) / X_c.shape[1]
            self._priors[idx] = len(X_c) + self.smoothing / float(samples) + self.smoothing*classes

    def fit(self, X, y):
        samples, features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        
        self._mean = np.zeros((n_classes, features))
        
        self._var = np.zeros((n_classes, features))
        
        self._priors = np.zeros(n_classes)

        for idx, c in enumerate(self._classes):
            
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(
                axis=0) + np.array([self.smoothing]*X_c.shape[1])
            self._var[idx, :] = X_c.var(
                axis=0) + np.array([self.varOffset]*X_c.shape[1])
            self._priors[idx] = len(X_c) / float(samples)

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.predict(x))
        return prediction

    def predict2(self, X):
        prediction = []
        for x in X:
            prediction.append(self._predict2(x))
        return prediction


    def _predict(self, x):
        posteriors = []

        
        for i in range(len(self._classes)):
            prior = np.log(self._priors[i])
            posterior = np.sum(np.log(self._pdf(i, x)))
            posterior += prior
            posteriors.append(posterior)

        
        result =  self._classes[np.argmax(posteriors)]

        return result 

    def _predict2(self, x):
        posteriors = []
        
        for i in range(len(self._classes)):
            prior = np.log(self._priors[i])
            numOfZeros = x.size - np.count_nonzero(x) / x.size
            posterior = np.log(1/abs(self.pmf[i] - numOfZeros))
            posterior += prior

            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, idx, x):
        mean = self._mean[idx]
        var = self._var[idx]
        

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        numerator[numerator == 0] = 1
        denominator = np.sqrt(2 * np.pi * var)

        result = (numerator) / (denominator)
        return result 


def accuracy(y, prediction):
    sum = 0
    for i in range(len(prediction)):
        if y[i] == prediction[i]:
            sum += 1

    result =  sum*100/len(prediction)

    return result 


# Testing
LAPLACE = sample([0.1, 0.5, 1, 10, 100], 1)[0]
LAPLACE = 100
mndata = MNIST('./')
mndata.gz = True
trainingImages, trainingLabels = map(np.array, mndata.load_training())
testingImages, testingLabels = map(np.array, mndata.load_testing())

nb = NaiveBayes(LAPLACE)
print("Training our model wait...")
nb.fit(trainingImages, trainingLabels)
print("Predicting...")
yPred = nb.predict(testingImages)
score = accuracy(testingLabels, yPred)
print(f"Accuracy with laplace smoothing of {LAPLACE} is {score}")
with open('outputNaive_bayes.txt', 'a') as file:
    file.writelines([f"1, {LAPLACE}, {score}", "\n"])
