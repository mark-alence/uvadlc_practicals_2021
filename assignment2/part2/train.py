###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Adapted: 2021-11-11
###############################################################################
import sys
from datetime import datetime
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)
    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    device = args.device
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    data_loader = DataLoader(dataset, args.batch_size,
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)
    args.vocabulary_size = dataset.vocabulary_size
    model = TextGenerationModel(args)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_module = nn.CrossEntropyLoss()
    accuracies = []
    losses = []
    for epoch in range(args.num_epochs):
        true_preds, count = 0, 0
        l = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x).permute((1, 2, 0))
            loss = loss_module(preds, y.permute((1, 0)))
            l += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            true_preds += (preds.argmax(dim=1) == y.T).sum()
            count += np.prod(y.shape)
        train_acc = true_preds.item() / count
        accuracies.append(train_acc)
        losses.append(l / count)
        print(f'TRAIN ACC AT EPOCH {epoch + 1}: {train_acc}')
        if epoch + 1 in [1, 5, 10, 20]:
            torch.save(model.state_dict(), f'model_epoch_{epoch + 1}_{args.text}')
    np.save(f'accuracies_{args.text}', accuracies)
    np.save(f'losses_{args.text}', losses)
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    text = "book_EN_democracy_in_the_US"
    parser.add_argument('--txt_file', type=str, default=f'assets/{text}.txt',
                        help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=35, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')
    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    args.text = text
    train(args)

    dataset = TextDataset(args.txt_file, args.input_seq_length)
    args.vocabulary_size = dataset.vocabulary_size
    nl = dataset._char_to_ix['\n']
    space = dataset._char_to_ix[' ']
    for m in [1]:
        model = TextGenerationModel(args)
        model.load_state_dict(torch.load(f'model_epoch_{m}_{text}', map_location=torch.device('cpu')))
        model.eval()
        for t in [0.5, 1, 2]:
            samples = model.sample(temperature=t)
            for s in samples:
                sentence = [dataset._ix_to_char[int(i)] for i in s]
                sentence = [x if x != '\n' else ' ' for x in sentence]
                print(''.join(sentence))
            print()
