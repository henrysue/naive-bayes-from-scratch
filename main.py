# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:55:35 2020

main.py

@author: Henry
"""

## Naive Bayes Classifier From Scratch
## CS 235 Final Project
## UC Riverside - hsue002@ucr.edu

# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.dummy import DummyClassifier

from naive_bayes_CV import train_NB, test_NB

## Data cleaning for evaluation

# Read training and test data
train = pd.read_csv('.\\Data\\Sentiment140\\training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
test = pd.read_csv('.\\Data\\Sentiment140\\testdata.manual.2009.06.14.csv', header=None)

# Adding column indexes for training and test data
train.columns = ['Tweet Sentiment', 'Tweet ID', 'Tweet Date', 'Query', 'Tweet User', 'Tweet Text']
test.columns = ['Tweet Sentiment', 'Tweet ID', 'Tweet Date', 'Query', 'Tweet User', 'Tweet Text']

# Replace class labels
train['Tweet Sentiment'] = train['Tweet Sentiment'].replace([0,4],[0,1])
test['Tweet Sentiment'] = test['Tweet Sentiment'].replace([0,2,4],[0,0.5,1])

_, train_small = train_test_split(train, test_size = 0.02, random_state = 42, stratify = train['Tweet Sentiment'])

# Extract training data
training_text = train_small['Tweet Text'].reset_index(drop=True)
training_result = train_small['Tweet Sentiment'].reset_index(drop=True)

# Extract test data
test_text = test['Tweet Text']
test_answers = test['Tweet Sentiment']

# Drop indexes where there is neutral sentiment in test data - training data offers no neutral sentiment training data
neutral_index = test_answers.index[test_answers == 0.5].tolist()
test_text_binary = test['Tweet Text'].drop(neutral_index).reset_index(drop=True)
test_answers_binary = test['Tweet Sentiment'].drop(neutral_index).reset_index(drop=True)

## Baseline Model Performance Test using canned functions from sklearn
"""
Here we follow the training example in scikit-learn's documentation.
We fit two baseline models - sklearn's implementation of MultinomialNB and a Dummy Classifier.
The dummy classifier will predict based solely on the distribution of class labels.
If we were to stratified split the training data, the dummy classifier should guess 50/50 or 50% acc on test data.
"""
# Here we specify sklearn's CountVectorizer as the vectorizer for each word
vectorizer = CountVectorizer()

# Train Multinomial Naive Bayes
MultinomialNB_clf = MultinomialNB()
MultinomialNB_clf.fit(vectorizer.fit_transform(training_text), training_result)

# Train Bernoulli Naive Bayes
BernoulliNB_clf = BernoulliNB()
BernoulliNB_clf.fit(vectorizer.fit_transform(training_text), training_result)

# Train Dummy Classifier
dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(vectorizer.fit_transform(training_text), training_result)

# Define a function to analyze text using our classifier
def analyze_text(clf, vectorizer, test_text):
    """
    This helper function takes in a classifier, vectorizer and text and returns a prediction.
    Input: clf: Classifier Model, vectorizer: Vectorizer, test_text: a text passage 
    Output: Original test_text passage, the model's prediction
    """
    prediction = clf.predict(vectorizer.transform([test_text]))
    
    return(test_text, prediction)
    
# Define a duplicate function of analyze_text to use i/o for our own implementation of NB
def analyze_text_self_NB(vectorizer, test_text):
    """
    This helper function takes in a classifier, vectorizer and text and returns a prediction.
    Input: clf: Classifier Model, vectorizer: Vectorizer, test_text: a text passage 
    Output: Original test_text passage, the model's prediction
    """
    prediction = test_NB(test_text, prior, likelihood, vectorizer.get_feature_names())
    
    return(test_text, prediction)

# Define a function to evalute the accuracy of a model
def evaluate_model_accuracy(clf, test_text, test_result):
    """
    This function runs the "analyze_text" function over each row in a test_text column and compares the output to
        the test answers ("test_result" column)
    Input: clf:classifier, test_text: column of test texts, test_result: column of correct class answers
    Output: A formatted string for model accuracy (rounded to two decimal places) 
    """
    total = len(test_text)
    num_correct = 0
    
    for index in range(0, total):
        
        text, result = analyze_text(clf, vectorizer, test_text[index])
        
        if result[0] == test_result[index]:
            num_correct +=1 
    
    return_string = 'Model Info: ' + str(clf) + '\nModel Accuracy: ' + str(round(num_correct * 100 / total, 2)) + '%\n'
    
    print(return_string)
    
# Define a function to evalute the accuracy of our implementation of NB
def evaluate_model_accuracy_NB(test_text, test_result):
    """
    This function runs the "analyze_text" function over each row in a test_text column and compares the output to
        the test answers ("test_result" column)
    Input: clf:classifier, test_text: column of test texts, test_result: column of correct class answers
    Output: A formatted string for model accuracy (rounded to two decimal places) 
    """
    total = len(test_text)
    num_correct = 0
    
    for index in range(0, total):
        
        text, result = analyze_text_self_NB(vectorizer, test_text[index])
        
        if result == test_result[index]:
            num_correct +=1 
    
    return_string = 'Model Info: ' + 'Self-Implemented Naive Bayes \n' + 'Model Accuracy: ' + str(round(num_correct * 100 / total, 2)) + '%\n'
    
    print(return_string)
    
# Define a function to evalute the accuracy of our implementation of NB
def evaluate_model_stats_NB(test_text, test_result):
    """
    This function runs the "analyze_text" function over each row in a test_text column and compares the output to
        the test answers ("test_result" column)
    Input: clf:classifier, test_text: column of test texts, test_result: column of correct class answers
    Output: A formatted string for model accuracy (rounded to two decimal places) 
    """
    total = len(test_text)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for index in range(0, total):
        
        text, result = analyze_text_self_NB(vectorizer, test_text[index])
        
        if result == 1:
            
            if result == test_result[index]:
                tp += 1
            else:
                fp += 1
        else:
            if result == test_result[index]:
                tn += 1
            else:
                fn += 1

    confusion_matrix = pd.DataFrame(np.array([[tp,fp],[fn,tn]]), 
                                    columns = ['True Pos','True Neg'],
                                    index = ['Predict Pos','Predict Neg'])
    acc = ((tp + tn) / total)
    prec = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    spec = (tn / (tn + fp))
    f1 = (2 * prec * recall) / (prec + recall)
    
    print('Model Info: Self-Implemented Naive Bayes')
    print('Model Accuracy: ' + str(round((acc)*100,2)) + '%')
    print('Model Precision: ' + str(round((prec)*100,2)) + '%')
    print('Model Recall: ' + str(round((recall)*100,2)) + '%')
    print('Model Specificity: ' + str(round((spec)*100,2)) + '%')
    print('Model F1 Score: ' + str(round((f1)*100,2)) + '%\n')
    print('Model Confusion Matrix:')
    print(confusion_matrix)
    
# Evaluate self-built implementation of Naive Bayes
X = vectorizer.fit_transform(training_text)
y = training_result

# Getting vocab for vectorizer and csting as numpy array
vocab = vectorizer.get_feature_names()
count_matrix = X.toarray() 

# Train our Naive Bayes
prior, likelihood = train_NB(count_matrix, y)

# Evaluate the output of all of our models
print('--- Model Performance Comparison ---\n' )
evaluate_model_accuracy(dummy_clf, test_text_binary, test_answers_binary)
evaluate_model_accuracy(MultinomialNB_clf, test_text_binary, test_answers_binary)
evaluate_model_accuracy(BernoulliNB_clf, test_text_binary, test_answers_binary)
evaluate_model_accuracy_NB(test_text_binary, test_answers_binary)

print('--- Self Implemented Model Statistics ---\n')
evaluate_model_stats_NB(test_text_binary, test_answers_binary)









