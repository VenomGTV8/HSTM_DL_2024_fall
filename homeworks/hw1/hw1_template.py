import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(train_csv, val_csv, test_csv):
    ### YOUR CODE HERE
    return X_train, y_train, X_val, y_val, X_test, y_test


def init_model():
    ### YOUR CODE HERE
    return model, criterion, optimizer


def evaluate(model, X, y):
    ### YOUR CODE HERE
    return predictions, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs):
    ### YOUR CODE HERE: train the model and validate it every epoch on X_val, y_val
    return model


def main(args):
    ### YOUR CODE HERE

    # Load data

    # Initialize model

    # Train model

    # Predict on test set

    # dump predictions to 'submission.csv'
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='homeworks/hw1/data/train.csv')
    parser.add_argument('--val_csv', default='homeworks/hw1/data/val.csv')
    parser.add_argument('--test_csv', default='homeworks/hw1/data/test.csv')
    parser.add_argument('--out_csv', default='homeworks/hw1/data/submission.csv')
    parser.add_argument('--lr', default=0)
    parser.add_argument('--batch_size', default=0)
    parser.add_argument('--num_epoches', default=0)

    args = parser.parse_args()
    main(args)
