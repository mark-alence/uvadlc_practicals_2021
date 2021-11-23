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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    preds = predictions.argmax(1)
    accuracy = (preds == labels).sum() / preds.shape[0]
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy.item()


def evaluate_model(model, loader):
    acc = 0
    cifar10_iter = iter(loader)
    num_batches = 0
    N = len(loader.dataset)
    while data := next(cifar10_iter, None):
        num_batches += 1
        images, labels = data
        input_data = images.reshape(
            (images.shape[0], -1))
        outputs = model(input_data)
        acc += accuracy(outputs, labels) * (len(labels) / N)
    return acc


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    if hidden_dims:
        dnn_hidden_units = [int(hidden_dims)
                            for hidden_dims in hidden_dims]
    else:
        dnn_hidden_units = []

    n_classes = 10
    n_inputs = 3 * 32 * 32

    model = nn.DataParallel(MLP(n_inputs, dnn_hidden_units, n_classes, use_batch_norm=use_batch_norm)).to(device)
    best_model = None
    loss_module = nn.CrossEntropyLoss()

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    N = len(cifar10_loader['train'].dataset)
    for step in range(epochs):
        cifar10_train_iter = iter(cifar10_loader['train'])
        t_loss = 0
        t_accuracy = 0
        while data := next(cifar10_train_iter, None):
            images, labels = data
            input_data = images.reshape((batch_size, -1))
            outputs = model.forward(input_data)
            loss = loss_module(outputs, labels)
            t_loss += loss.item() * (len(labels) / N)
            t_accuracy += accuracy(outputs, labels) * (len(labels) / N)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(t_loss)
        train_accuracies.append(t_accuracy)
        val_acc = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_acc)
        if val_acc == max(val_accuracies): best_model = deepcopy(model)

    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    logging_info = {'train_loss': train_losses, 'train_accuracy': train_accuracies, 'use_batch_norm': use_batch_norm}
    np.save('mlp_train_accuracies_torch', train_accuracies)
    np.save('mlp_train_losses_torch', train_losses)
    np.save('mlp_validation_accuracy_torch', val_accuracies)
    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
