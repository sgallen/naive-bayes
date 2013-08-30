#! /usr/bin/python
""" Naive Bayes Classifier - Utilizes 'bag of words' and laplacian smoothing
    assumes training data is classified per file, with one example per line 
    of file. """

__appname__ = "Naive Bayes Classifier"
__author__ = "Scott G. Allen"
__version__ = "0.0pre0"

import argparse
import numpy as np
import string
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize

class NBClassifier(object):
    def __init__(self, data):
        self.training_data = data
        self.labels = data.keys()
        # http://ohuiginn.net/mt/2010/07/nested_dictionaries_in_python.html
        self.bag_of_words = defaultdict(lambda: defaultdict(int))
        self.bag_of_word_probabilities = defaultdict(lambda: defaultdict(int))

    def create_bag_of_words(self):
        for class_key, class_data in self.training_data.items():
            token_list = self.create_token_list(class_data)
            for token in token_list:
                self.bag_of_words[token][class_key] += 1 

    def create_token_list(self, data_list):
        token_list = []
        for entry in data_list:
            token_list.extend(self.tokenize_entry(entry))
        return token_list

    def tokenize_entry(self, entry):
        entry_tokens = []
        sentences = sent_tokenize(entry)
        for sentence in sentences:
            entry_tokens.extend(word_tokenize(sentence))
        lower_tokens = [word.lower() for word in entry_tokens if word not in string.punctuation]
        return lower_tokens

    def get_bag_of_words(self):
        return self.bag_of_words

    def get_num_of_distinct_words(self):
        return float(len(self.bag_of_words.keys()))

    def get_class_probability(self, feature_class, k=1):
        # Using Laplacian Smoothing, default k = 1
        if feature_class in self.labels:
            return ((len(self.training_data[feature_class]) + k) / (self.get_num_training_examples() + k*(self.get_num_classes())))
        else:
            return ((0 + k) / (self.get_num_training_examples() + k*(self.get_num_classes)))

    def get_num_training_examples(self):
        num = 0
        for feature_class, data in self.training_data.items():
            num += len(data)
        return num

    def get_num_classes(self):
        return float(len(self.labels))

    def get_feature_probability(self, feature, feature_class, k=1):
        # Using Laplacian Smoothing, default k = 1
        if feature in self.bag_of_words and feature_class in self.bag_of_words[feature]:
            return ((self.bag_of_words[feature][feature_class] + k) / (self.get_num_words_in_class(feature_class) + (k*self.get_num_distinct_words())))
        else:
            return ((0 + k) / (self.get_num_words_in_class(feature_class) + (k*self.get_num_distinct_words())))

    def get_num_words_in_class(self, feature_class):
        num_words = 0
        for word, value_dict in self.bag_of_words.items():
            if feature_class in value_dict:
                num_words += value_dict[feature_class]
        return float(num_words)

    def get_num_distinct_words(self):
        return len(self.bag_of_words.keys())

    def train(self):
        for word, values in self.bag_of_words.items():
            for feature_class, word_count in values.items():
                self.bag_of_word_probabilities[word][feature_class] = self.get_feature_probability(word, feature_class)

    def get_bag_of_word_probabilities(self):
        return self.bag_of_word_probabilities

    def calc_proportional_prob(self, input_data, feature_class):
        input_tokens = self.tokenize_entry(input_data)
        feature_class_probability = self.get_class_probability(feature_class)
        # probability = 1.0
        probability = math.log(1.0)
        for token in input_tokens:
            if token in self.bag_of_word_probabilities and feature_class in self.bag_of_word_probabilities[token]:
                probability += math.log(self.bag_of_word_probabilities[token][feature_class])    
                # probability *= self.bag_of_word_probabilities[token][feature_class]
            else:
                # probability *= self.get_feature_probability(token, feature_class)
                probability += math.log(self.get_feature_probability(token, feature_class))
        return (probability + math.log(feature_class_probability))
        # return (probability * feature_class_probability)

def load_training_data(data_files_list):
    data = {}
    for filename in data_files_list:
        label = filename.split('_')[1]
        with open(filename, 'r') as f:
            data[label] = f.readlines()
    return data

def main():
    # setup argument parsing
    parser = argparse.ArgumentParser(version="%%prog v{0}".format(__version__))
    parser.add_argument("data_files", help="File containing first training data set", nargs='+')
    args = parser.parse_args()
    data_dict = load_training_data(args.data_files)
    nb = NBClassifier(data_dict)
    nb.create_bag_of_words()
    
    print "Number of classes: {}".format(nb.get_num_classes())
    print "Total number of distinct words: {}".format(nb.get_num_distinct_words())

    nb.train()

    test = """Early Friday afternoon, the lead negotiators for the N.B.A. and the players union will hold a bargaining session in Beverly Hills - the latest attempt to break a 12-month stalemate on a new labor deal."""

    print "Prob. sports: {}".format(nb.calc_proportional_prob(test, "sports"))
    print "Prob. arts: {}".format(nb.calc_proportional_prob(test, "arts"))

if __name__ == "__main__":
    main()
