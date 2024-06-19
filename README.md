# Sentiment Analysis - Coding Classification algorithms from scratch

This project is part of the "Machine Learning with Python - From Linear Models to Deep Learning" course offered by MIT.

## Introduction

This project implements several classification algorithms for machine learning tasks. The algorithms included are:

1. **Perceptron Algorithm**: This algorithm updates classification parameters based on a single step. It iterates through the dataset multiple times to converge to an optimal solution.

2. **Average Perceptron Algorithm**: Similar to the Perceptron algorithm but computes the average of the parameters over multiple iterations, leading to a more stable model.

3. **Pegasos Algorithm**: This algorithm optimizes linear classifiers using stochastic gradient descent. It incorporates a regularization term to prevent overfitting.

## Bag of Words (BoW)

Bag of Words (BoW) is a technique used in Natural Language Processing (NLP) to represent text data. It treats each document as a collection of words, ignoring grammar and word order. This representation is useful for tasks like text classification and sentiment analysis.

## Code Overview

### project1.py

This Python file contains implementations of the aforementioned algorithms along with helper functions for feature extraction and classification.

1. **hinge_loss_single**: Calculates the hinge loss for a single data point given classification parameters.
2. **hinge_loss_full**: Computes the average hinge loss over a dataset.
3. **perceptron**: Implements the Perceptron algorithm.
4. **average_perceptron**: Implements the Average Perceptron algorithm.
5. **pegasos_single_step_update**: Updates classification parameters using the Pegasos algorithm.
6. **pegasos**: Implements the Pegasos algorithm for optimization.
7. **classify**: Classifies data points using given parameters.
8. **classifier_accuracy**: Computes accuracy of a classifier on training and validation data.
9. **accuracy**: Computes the fraction of correct predictions.
10. **extract_words** and **bag_of_words**: Helper functions for BoW feature extraction.

### Usage

To use these algorithms, import the necessary functions from `project1.py` and pass the required inputs according to each function's documentation.

For example:

```python
from project1 import perceptron, classify, classifier_accuracy

# Load your data and feature matrices
# ...

# Train the perceptron algorithm
theta, theta_0 = perceptron(feature_matrix, labels, T=10)

# Classify new data points
predictions = classify(new_feature_matrix, theta, theta_0)

# Compute classifier accuracy
train_accuracy, val_accuracy = classifier_accuracy(perceptron, train_feature_matrix, val_feature_matrix,
                                                   train_labels, val_labels, T=10)
```
