import csv
import random
import re


def load_data(filename, split_ratio):
    training_data = []
    testing_data = []
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        dataset = list(lines)
        random.shuffle(dataset) 
        index_split = int(len(dataset) * split_ratio)
        training_data = dataset[:index_split]
        testing_data = dataset[index_split:]
    return training_data, testing_data


def clear_data(text):  # not training_data
    text = re.sub(r'[^\w\s]', '', text)
    each_word_array = text.lower().split()
    return each_word_array
