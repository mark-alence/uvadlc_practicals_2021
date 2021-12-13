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
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import *
from networks import *


def permute_indices(molecules: Batch) -> Batch:
    """permute the atoms within a molecule, but not across molecules

    Args:
        molecules: batch of molecules from pytorch geometric

    Returns:
        permuted molecules
    """
    # Permute the node indices within a molecule, but not across them.
    ranges = [
        (i, j) for i, j in zip(molecules.ptr.tolist(), molecules.ptr[1:].tolist())
    ]
    permu = torch.cat([torch.arange(i, j)[torch.randperm(j - i)] for i, j in ranges])

    n_nodes = molecules.x.size(0)
    inits = torch.arange(n_nodes)
    # For the edge_index to work, this must be an inverse permutation map.
    translation = {k: v for k, v in zip(permu.tolist(), inits.tolist())}

    permuted = deepcopy(molecules)
    permuted.x = permuted.x[permu]
    # Below is the identity transform, by construction of our permutation.
    permuted.batch = permuted.batch[permu]
    permuted.edge_index = (
        permuted.edge_index.cpu()
            .apply_(translation.get)
            .to(molecules.edge_index.device)
    )
    return permuted


def compute_loss(
        model: nn.Module, molecules: Batch, criterion: Callable
) -> torch.Tensor:
    """use the model to predict the target determined by molecules. loss computed by criterion.

    Args:
        model: trainable network
        molecules: batch of molecules from pytorch geometric
        criterion: callable which takes a prediction and the ground truth 

    Returns:
        loss

    TODO: 
    - conditionally compute loss based on model type
    - make sure there are no warnings / errors
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    is_gnn = model._get_name() == 'GNN'
    x = get_node_features(molecules) if is_gnn else get_mlp_features(molecules).to(model.device)
    y = get_labels(molecules).to(model.device)
    if is_gnn:
        preds = model(x, molecules.edge_index, torch.Tensor([int(torch.argmax(arr)) for arr in molecules.edge_attr]),
                      molecules.batch)
    else:
        preds = model(x)
    loss = criterion(preds.flatten(), y)
    #######################
    # END OF YOUR CODE    #
    #######################
    return loss.item()


def evaluate_model(
        model: nn.Module, data_loader: DataLoader, criterion: Callable, permute: bool
) -> float:
    """
    Performs the evaluation of the model on a given dataset.

    Args:
        model: trainable network
        data_loader: The data loader of the dataset to evaluate.
        criterion: loss module, i.e. torch.nn.MSELoss()
        permute: whether to permute the atoms within a molecule
    Returns:
        avg_loss: scalar float, the average loss of the model on the dataset.

    Hint: make sure to return the average loss of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    
    TODO: conditionally permute indices
          calculate loss
          average loss independent of batch sizes
          make sure the model is in the correct mode
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    avg_loss = 0
    N = len(data_loader.dataset)
    for b in data_loader:
        avg_loss += compute_loss(model, b if not permute else permute_indices(b), criterion) * len(b.y) / N
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_loss


def train(
        model: nn.Module, lr: float, batch_size: int, epochs: int, seed: int, data_dir: str
):
    """a full training cycle of an mlp / gnn on qm9.

    Args:
        model: a differentiable pytorch module which estimates the U0 quantity
        lr: learning rate of optimizer
        batch_size: batch size of molecules
        epochs: number of epochs to optimize over
        seed: random seed
        data_dir: where to place the qm9 data

    Returns:
        model: the trained model which performed best on the validation set
        test_loss: the loss over the test set
        permuted_test_loss: the loss over the test set where atomic indices have been permuted
        val_losses: the losses over the validation set at every epoch
        logging_info: general object with information for making plots or whatever you'd like to do with it

    TODO:
    - Implement the training of both the mlp and the gnn in the same function
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

    # Loading the dataset
    train, valid, test = get_qm9(data_dir, model.device)
    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        exclude_keys=["pos", "idx", "z", "name"],
    )
    valid_dataloader = DataLoader(
        valid, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )
    test_dataloader = DataLoader(
        test, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize loss module and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    val_losses = []
    train_losses = []
    best_model = None
    is_gnn = model._get_name() == 'GNN'
    N = len(train_dataloader.dataset)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for b in train_dataloader:
            b = b.to(model.device)
            x = get_node_features(b) if is_gnn else get_mlp_features(b).to(model.device)
            y = get_labels(b).to(model.device)
            optimizer.zero_grad()
            if is_gnn:
                preds = model(x, b.edge_index, torch.Tensor([int(torch.argmax(arr)) for arr in b.edge_attr]), b.batch)
            else:
                preds = model(x)
            loss = criterion(preds.flatten(), y)
            train_loss += loss.item() * b.y.shape[0] / N
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss)
        print(f'TRAIN LOSS AT EPOCH {epoch}: {train_loss}')
        model.eval()
        val_loss = evaluate_model(model, valid_dataloader, criterion, False)
        if not len(val_losses) or val_loss < min(val_losses):
            best_model = deepcopy(model)
        val_losses.append(val_loss)

    test_loss = evaluate_model(best_model, test_dataloader, criterion, False)
    permuted_test_loss = evaluate_model(best_model, test_dataloader, criterion, True)
    # torch.save(best_model.state_dict(), f'{best_model._get_name()}')
    # np.save(f'{best_model._get_name()}_train_loss', np.array(train_losses))
    np.save(f'{best_model._get_name()}_val_loss', np.array(val_losses))
    # # TODO: Training loop including validation, using evaluate_model
    # # TODO: Do optimization, we used adam with amsgrad. (many should work)
    # # TODO: Test best model
    # # TODO: Test best model against permuted indices
    # permuted_test_loss = ...
    # # TODO: Add any information you might want to save for plotting
    # logging_info = ...
    logging_info = {'train_losses': train_losses}

    return best_model, test_loss, permuted_test_loss, val_losses, logging_info

    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def test(model, data_dir, batch_size):
    criterion = nn.MSELoss()
    train, valid, test = get_qm9(data_dir, model.device)
    test_dataloader = DataLoader(
        test, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )
    return evaluate_model(model, test_dataloader, criterion, True)


def main(**kwargs):
    """main handles the arguments, instantiates the correct model, tracks the results, and saves them."""
    which_model = kwargs.pop("model")
    mlp_hidden_dims = kwargs.pop("mlp_hidden_dims")
    gnn_hidden_dims = kwargs.pop("gnn_hidden_dims")
    gnn_num_blocks = kwargs.pop("gnn_num_blocks")

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if which_model == "mlp":
        model = MLP(FLAT_INPUT_DIM, mlp_hidden_dims, 1)
    elif which_model == "gnn":
        model = GNN(
            n_node_features=Z_ONE_HOT_DIM,
            n_edge_features=EDGE_ATTR_DIM,
            n_hidden=gnn_hidden_dims,
            n_output=1,
            num_convolution_blocks=gnn_num_blocks,
        )
    else:
        raise NotImplementedError("only mlp and gnn are possible models.")

    model.to(device)
    model, test_loss, permuted_test_loss, val_losses, logging_info = train(
        model, **kwargs
    )
    # plot the loss curve, etc. below.
    epochs = kwargs.pop("epochs")
    x = np.arange(epochs)
    plt.plot(x, logging_info['train_losses'])
    plt.title(f'Train loss over epochs for GNN')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("gnn_train_loss.png")
    # plt.show()
    plt.figure()
    plt.plot(x, val_losses)
    plt.title(f'Validation loss over epochs for GNN')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("gnn_val_loss.png")
    # plt.show()
    print('Final average test loss', test_loss)
    print('Final average permuted test loss', permuted_test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model",
        default="gnn",
        type=str,
        choices=["mlp", "gnn"],
        help="Select between training an mlp or a gnn.",
    )

    parser.add_argument(
        "--mlp_hidden_dims",
        default=[128, 128, 128, 128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the mlp. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--gnn_hidden_dims",
        default=64,
        type=int,
        help="Hidden dimensionalities to use inside the mlp. The same number of hidden features are used at every layer.",
    )
    parser.add_argument(
        "--gnn_num_blocks",
        default=2,
        type=int,
        help="Number of blocks of GNN convolutions. A block may include multiple different kinds of convolutions (see GNN comments)!",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")

    # Technical
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the qm9 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
