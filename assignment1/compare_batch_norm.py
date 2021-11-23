################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
from train_mlp_pytorch import train
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """

    dims = [[128], [256, 128], [512, 256, 128]]
    results = {}
    print(results_filename)
    for i in range(len(dims)):
        model, val_accuracies, test_accuracy, logging_info = train(dims[i], 0.1, False, 128, 20, 42, 'data/')
        norm_model, norm_val_accuracies, norm_test_accuracy, norm_logging_info = train(dims[i], 0.1, True, 128, 20, 42,
                                                                                       'data/')
        results[f'net_{i + 1}'] = {
            'non-normalized': {'val_accuracies': val_accuracies,
                               'test_accuracy': test_accuracy,
                               'logging_info': logging_info},
            'normalized': {'val_accuracies': norm_val_accuracies,
                           'test_accuracy': norm_test_accuracy,
                           'logging_info': norm_logging_info}}
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    tacc = np.load("mlp_train_accuracies.npy")
    tloss = np.load("mlp_train_losses.npy")
    vloss = np.load("mlp_validation_accuracy.npy")
    titles = ["Train loss", "Train accuracy", "Validation loss"]
    arrs = [tloss, tacc, vloss]
    for i in range(3):
        plt.plot(np.arange(0, 10), arrs[i])
        plt.xlabel("Epochs")
        plt.ylabel(titles[i])
        plt.title(f"{titles[i]} of Numpy MLP over 10 epochs")
        plt.savefig(f'q0_{i}')
        plt.show()
    results = None
    with open('results.json') as f:
        results = json.load(f)
    c = iter(['r', 'g', 'b'])
    x = np.arange(20)
    for key in results.keys():
        col = next(c)
        print(np.std(results[key]['non-normalized']['val_accuracies']))
        print(np.std(results[key]['normalized']['val_accuracies']))
        print()
        plt.plot(x, results[key]['non-normalized']['val_accuracies'], f'{col}-')
        plt.plot(x, results[key]['normalized']['val_accuracies'], f'{col}--')
        plt.ylabel("Validation accuracy")
        plt.xlabel("Epochs")
    #     plt.legend()
    line1, = plt.plot([], "k-", label="Non-normalized")
    line2, = plt.plot([], "k--", label="Normalized", )
    plt.legend(handles=[line1, line2])
    title = "Validation accuracy for neural networks of varying dimensions with a learning rate of 0.1 \n"
    caption = "Red: 1 hidden layer with 128 neurons\n"
    caption += "Green: 2 hidden layers with 256 and 128 neurons respectively\n"
    caption += "Blue: 3 hidden layers with 512, 256, and 128 neurons respectively\n"
    plt.xticks(np.arange(0, 21, 2))
    plt.figtext(0, -0.2, caption, wrap=True, horizontalalignment='left', fontsize=12)
    plt.title(title)
    # plt.show()
    plt.savefig("q4_acc_val", bbox_inches='tight')
    #     plt.plot(x, results[key]['non-normalized']['logging_info']['train_loss'], label="Non-normalized")
    #     plt.plot(x, results[key]['normalized']['logging_info']['train_loss'], label="Normalized")
    #     plt.legend()
    #     plt.ylabel("Train loss")
    #     plt.xlabel("Epochs")
    #     plt.show()
    #     plt.plot(x, results[key]['non-normalized']['val_accuracies'], label="Non-normalized")
    #     plt.plot(x, results[key]['normalized']['val_accuracies'],label="Normalized")
    #     plt.legend()
    #     plt.ylabel("Validation accuracy")
    #     plt.xlabel("Epochs")
    #     plt.show()
    a = []
    for i in range(3):
        #     z = [np.mean(results[f'net_{i+1}'][x]['logging_info']['train_accuracy'])for x in ['normalized', 'non-normalized']]
        #     z = [np.std(results[f'net_{i+1}'][x]['val_accuracies'])for x in ['normalized', 'non-normalized']]
        z = [(results[f'net_{i + 1}'][x]['test_accuracy']) for x in ['normalized', 'non-normalized']]
        print(z)
        a.append(z)
    # np.mean(results[key]['non-normalized']['val_accuracies'])
    for i in range(3):
        #     print(f'Standard deviation of validation accuracies with batch normalization {round(a[i][0],3)}')
        #     print(f'Standard deviation of validation accuracies without batch normalization {round(a[i][1],3)}')
        #     print()
        print(
            f'Improvement in test accuracy using batch normalization for network {i + 1} {round((a[i][0] - a[i][1]) * 100, 2)}%')
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    # FILENAME = 'results.json'
    # train_models(FILENAME)
    # plot_results(FILENAME)
