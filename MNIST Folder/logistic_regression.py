import numpy as np
from mnist import MNIST


def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=10):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        samples, features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self.weights = np.zeros((n_classes, features))
        correctLabel = np.zeros((samples, n_classes))
        for i, right in enumerate(y):
            correctLabel[i, right] = 1

        self.bias = np.zeros(n_classes)

        for i, c in enumerate(self._classes):
            for _ in range(self.n_iters):
                
                linear_pred = np.array(
                    [self.weights.dot(x) + self.bias for x in X])
                prediction = sigmoid(linear_pred)

                weight = (1/samples) * np.dot(X.T, (prediction - correctLabel))
                bias = (1/samples) * np.sum(prediction-correctLabel)

                
                self.weights -= self.lr*weight.T
                self.bias = self.bias - self.lr*bias.T

    def predict(self, X):
        prediction = np.array([self.weights.dot(x) + self.bias for x in X])
        predict = sigmoid(prediction)
        value = predict.argmax(axis=1)
        return value


LEARNINGRATE = 1.5
print("Minst...")

mndata = MNIST('./')
mndata.gz = True
trainingImages, trainingLabels = map(np.array, mndata.load_training())
testingImages, testingLabels = map(np.array, mndata.load_testing())


trainingImages = trainingImages[:5000]
trainingLabels = trainingLabels[:5000]


print("Fitting...")
logR = LogisticRegression(lr=LEARNINGRATE)
logR.fit(trainingImages, trainingLabels)

print("Predicting...")
predict = logR.predict(testingImages)


def accuracy(predict, test):
    result = 0
    for i in range(len(predict)):
        if predict[i] == test[i]:
            result += 1

    value = result*100/len(predict)

    return value


acc = accuracy(predict, testingLabels)
print(f"Accuracy is {acc}%")

with open('outputLogistic-regression.txt', 'a') as file:
    file.writelines([f"1, {LEARNINGRATE}, {acc}", "\n"])
