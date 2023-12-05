import numpy as np
import matplotlib.pyplot as plt
naiveOrLog = input("Do you want naiveBayes or logisitcRegression? (n/l)")
if naiveOrLog.lower() == 'n':

    with open("outputNaive-bayes.txt") as file:
        lines = file.readlines()
        lines = [list(map(eval, line.split(','))) for line in lines]
        feature1 = []
        for line in lines:
            if line[0] == 1:
                feature1.append(line)

        feature1 = np.array(sorted(feature1, key=lambda x: x[1]))

        x = feature1[:, 1]
        print(x)
        y = feature1[:, 2]

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Accuracy with laplace smoothing")
        plt.xlabel('Laplace Value')
        plt.ylabel('Accuracy(%)')
        ax.grid()
        fig.savefig('NaiveBayes.png')
        plt.show()

else:
    with open("outputLogistic-regression.txt") as file:
        lines = file.readlines()
        lines = [list(map(eval, line.split(','))) for line in lines]
        feature1 = []
        for line in lines:
            if line[0] == 1:
                feature1.append(line)

        feature1 = np.array(sorted(feature1, key=lambda x: x[1]))

        x = feature1[:, 1]
        print(x)
        y = feature1[:, 2]
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Accuracy with learning rate")
        plt.xlabel('Learning rate')
        plt.ylabel('Accuracy(%)')
        ax.grid()
        fig.savefig('LogisticRegression.png')
        plt.show()
