import math
import re
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self, feature_extractor, smoothing_factor):
        self.extract = feature_extractor
        self.smoothing = smoothing_factor
        self.category = []
        self.feature_probs = defaultdict(dict)
        self.category_probs = {}
    
    def train(self, dataset):
        self.val = self.smoothing
        self.category = list(set([cat for cat, _ in dataset]))
        total_size = len(dataset)
        count = 0
        for cat in self.category:
            for c, _ in dataset:
                if c == cat:
                    count += 1
            self.category_probs[cat] = count / total_size

        features = self.extract(dataset)
        preprocessed_texts = self.cleared_texts(dataset)
        
        for cat in self.category:
            texts = [t for c, t in dataset if c == cat]
            text_count = len(texts)
            word_counts = defaultdict(int)
            
            for text in texts:
                words = preprocessed_texts[text]
                for word in words:
                    word_counts[word] += 1
            
            for feature in features:
                count = word_counts[feature]
                self.feature_probs[cat][feature] = (count + self.val) / (text_count + len(features) * self.val)
    
    def cleared_texts(self, dataset):
        preprocessed_texts = {}
        for _, text in dataset:
            preprocessed_texts[text] = self.extract([(None, text)])  # Extract features from text
        return preprocessed_texts
    
    def predict(self, text):
        features = self.extract([(None, text)])  # Extract features from text
        probs = {cat: math.log(self.probability_of_category(cat)) for cat in self.category}
        
        for cat in self.category:
            feat_probs = [self.calculate_feature_prob(feature, cat) for feature in features]
            feat_probs = [p for p in feat_probs if p > 0.0]
            probs[cat] += sum(math.log(p) for p in feat_probs)
        
        return max(probs, key=probs.get)
    
    def probability_of_category(self, category):
        return self.category_probs[category]
    
    def calculate_feature_prob(self, feature, category):
        if feature in self.feature_probs[category]:
            return self.feature_probs[category][feature]
        else:
            return 0.0
    
    def calculate_accuracy(self, test_set):
        correct = 0
        total = len(test_set)
        for cat, text in test_set:
            predicted_cat = self.predict(text)
            if predicted_cat == cat:
                correct += 1
        return correct / total * 100