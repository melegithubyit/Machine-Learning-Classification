import csv

def data_loader(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # SKIP THE HEADER
        for row in csv_reader:
            category = row[0]
            sentence = row[1]
            dataset.append((sentence, category))
    return dataset
