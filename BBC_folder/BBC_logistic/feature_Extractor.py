from collections import defaultdict


def calculate_tf(sentence):
    words = sentence.split()
    term_frequency = defaultdict(int)
    for word in words:
        term_frequency[word] += 1
    return term_frequency


