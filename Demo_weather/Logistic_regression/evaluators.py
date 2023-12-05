from utilities import predict

def evaluate(test_data, weights, bias):
    num_correct = 0
    for row in test_data:
        inputs = row[:-1]
        label = row[-1]
        predicted = predict(inputs, weights, bias)
        if predicted >= 0.5 and label == 1:
            num_correct += 1
        elif predicted < 0.5 and label == 0:
            num_correct += 1
    accuracy = num_correct / len(test_data) * 100
    return int(accuracy)
