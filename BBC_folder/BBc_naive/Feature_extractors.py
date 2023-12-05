import math
from Data_loader import clear_data
from collections import defaultdict

def word_embeddings(dataset, embedding_dim=100):
    word_counts = defaultdict(int)
    word_embeddings = {}
    for _, text in dataset:
        words=clear_data(text)
        for word in words:
            word_counts[word] += 1
    for word, count in word_counts.items():
        vector = [count] * embedding_dim
        word_embeddings[word] = vector

    return word_embeddings

def TF_IDF(dataset):
    features = {}
    doc_count = len(dataset)
    for _, text in dataset:
        words = clear_data(text)
        for word in words:
            if word not in features:
                features[word] = 1
            else:
                features[word] += 1
    return {word: math.log(doc_count / features[word]) for word in features}


def bag_of_words(dataset):
    features = []
    for _, text in dataset:
        words = clear_data(text)
        features.extend(words)
    return list(set(features))  

