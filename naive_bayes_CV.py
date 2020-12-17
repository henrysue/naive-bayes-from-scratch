# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:17:22 2020

@author: Henru
"""

# Import Libraries
import string
import pandas as pd
from math import log

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv('.\\Data\\Sentiment140\\training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
test = pd.read_csv('.\\Data\\Sentiment140\\testdata.manual.2009.06.14.csv', header=None)

# Adding column indexes for training and test data
train.columns = ['Tweet Sentiment', 'Tweet ID', 'Tweet Date', 'Query', 'Tweet User', 'Tweet Text']
test.columns = ['Tweet Sentiment', 'Tweet ID', 'Tweet Date', 'Query', 'Tweet User', 'Tweet Text']

# Replace class labels (normalize 0:4 to 0:1)
train['Tweet Sentiment'] = train['Tweet Sentiment'].replace([0,4],[0,1])

# Take small subsample of test data for function validation
_, train_small = train_test_split(train, test_size = 0.0001)

# Count the frequency of each class
freq = train_small['Tweet Sentiment'].value_counts()

# Reset indices after split
training_text = train_small['Tweet Text'].reset_index(drop=True)
training_result = train_small['Tweet Sentiment'].reset_index(drop=True)

# Define a function to split data into a dictionary by class labels
def class_split(sparse_mat, label_col):
    """
    Splits DataFrame into a list of class labels and their respective data points
    Input: numpy array of word counts of each word for each text passage, column of data labels
    Output: A dictionary of class:data points
    """
    class_dict = {}
    
    for i in range(0, label_col.shape[0]):

        class_label = label_col[i]

        if class_label not in class_dict:
            class_dict[class_label] = []

        class_dict[class_label].append(sparse_mat[i].tolist())

    return(class_dict)

# Define a function to count the frequency of each feature
def count_features(class_dict):
    """
    Returns a dictionary of series of total counts of each feature occurence 
        for each class label
    Input: Dictionary of rows by class label
    Output: Dictionary of class:list of counts of each feature
    """
    counts = {}
    
    for class_label in class_dict.keys(): 
        
        n = len(class_dict[class_label][0])
        counts[class_label] = [0] * n
        
        for row in class_dict[class_label]:
            
            for idx, count in enumerate(row):
                
                counts[class_label][idx] += count
        
    return(counts)

# Define a function to train the classifier
def train_NB(sparse_mat, label_col):
    """
    Trains multinomial naive bayes classifier given a sparse dataframe of words (columns) 
        and counts (row information) and row label.
    Input: sparse_mat: Sparse matrix with word counts for each text passage, 
        label_col: column labels for each row
    Output: log_prior of trained classifier, log_likelihood of trained classifier,
        list of classes, list of words in vocab
    """ 
    # initialize an empty dictionary for probabilities of each class.
    log_prior = {}
    log_likelihood = {}
    label_counts = {}
    total_words_in_class = {}
    
    # Get our dictionary of data points per class
    class_dict = class_split(sparse_mat, label_col)

    # Counts number of times each class label occurs
    for label in class_dict.keys():
        
        count = 0
        
        for i in label_col:
            
            if i == label:
                count += 1
            
        label_counts[label] = count
    
    # Run 'count_features' function to count the number of times
    # each tokenized word occurs per class 
    counts = count_features(class_dict)
    
    # Find the total size of the vocab
    vocab_size = len(class_dict[0][0])
    
    for class_name in class_dict.keys():
        
        total_words_in_class[class_name] = 0
        
        for i in range(len(counts[class_name])):
            
            total_words_in_class[class_name] += counts[class_name][i]
    
    for class_name in class_dict:
        
        if class_name not in log_likelihood:
            log_likelihood[class_name] = {}
        
        log_prior[class_name] = log(len(class_dict[class_name]) / len(sparse_mat))
        
        # Calculate log-likelihood with Laplace Smoothing (add-1 Smoothing)
        for word_idx, count in enumerate(counts[class_name]):
            
            log_likelihood[class_name][word_idx] = log((counts[class_name][word_idx] + 1) / (total_words_in_class[class_name] + vocab_size))
        
    return(log_prior, log_likelihood)

# Define test function for trained Naive Bayes Classifier
def test_NB(document, log_prior, log_likelihood, vocab):
    """
    This function takes a document and predicts a class using argmax of each class probability
    Input: text document/passage, trained log_prior, trained log_likelihood, list of class labels,
        and vocab of the trained classifier.
    Output: The predicted class based or argmax(class probabilities)
    
    """
    prob_sum = {}
    
    stripped_doc = document.translate(str.maketrans('', '', string.punctuation))
    
    for class_label in log_prior.keys():
        
        prob_sum[class_label] = log_prior[class_label]
    
        for word in stripped_doc.split(' '):
            
            if word in vocab:
                
                for idx, vocab_word in enumerate(vocab):
                    if word == vocab_word:
                        word_idx = idx 
                        
                        prob_sum[class_label] += log_likelihood[class_label][word_idx]
        
    max_value = max(prob_sum.values())  # maximum value
    prediction = [key for key, val in prob_sum.items() if val == max_value][0]
        
    return(prediction)

# Call vectorizer to get Bag of Words for our corpus 
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(training_text)
y = training_result

# Cast our bag of words sparse matrix as an array to be used with our implementation
train_mat = X.toarray()

# Split our training data by class
class_dict_small = class_split(train_mat, y)

# Tally 
counts = count_features(class_dict_small)

prior, likelihood = train_NB(train_mat, y)

# Test Document / String for our model. It is clearly positive sentiment
test_doc = 'happy birthday to you! I love you'

# Test our classifier on our test text 'test_doc'
pred = test_NB(test_doc, prior, likelihood, vectorizer.get_feature_names())

# Uncomment below line to output our prediciton for the test doc (should be 1 or 'positive')
# print(pred)




