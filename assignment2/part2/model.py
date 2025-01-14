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

import math
import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """

    def __init__(self, lstm_hidden_dim, embedding_size, device='cpu'):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        self.device = device
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # forget gate
        self.W_fx = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_fh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(lstm_hidden_dim))

        # input gate
        self.W_ix = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_ih = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(lstm_hidden_dim))

        # input modulation gate
        self.W_gx = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_gh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(lstm_hidden_dim))

        # output gate
        self.W_ox = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_oh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(lstm_hidden_dim))

        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        bound = 1 / math.sqrt(self.hidden_dim)
        for wt in self.parameters():
            wt.data.uniform_(-bound, bound)
        self.b_f = nn.Parameter(self.b_f + 1)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds, hidden=None):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, embed dimension].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, embed dimension].
        """
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        input_length, batch_size, embed_dim = embeds.shape
        h_t = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        h_arr = torch.zeros(input_length, batch_size, self.hidden_dim).to(self.device)
        first = 0 if hidden is None else hidden.shape[0]
        if hidden is not None and hidden.shape[0]:
            h_arr[:-1] = hidden

        for t in range(first, input_length):
            x_t = embeds[t]
            g_t = torch.tanh(x_t @ self.W_gx + h_t @ self.W_gh + self.b_g).to(self.device)
            i_t = torch.sigmoid(x_t @ self.W_ix + h_t @ self.W_ih + self.b_i).to(self.device)
            f_t = torch.sigmoid(x_t @ self.W_fx + h_t @ self.W_fh + self.b_f).to(self.device)
            o_t = torch.sigmoid(x_t @ self.W_ox + h_t @ self.W_oh + self.b_o).to(self.device)
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t
            h_arr[t] = h_t
        return h_arr, h_t
        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """

    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.device = args.device
        self.vocabulary_size = args.vocabulary_size
        self.embedding = nn.Embedding(args.vocabulary_size, args.embedding_size).to(args.device)
        self.lstm = LSTM(args.lstm_hidden_dim, args.embedding_size, device=args.device)
        self.lstm = self.lstm.to(args.device)
        bound = 1 / math.sqrt(args.lstm_hidden_dim)
        self.linear = nn.Parameter(torch.Tensor(args.lstm_hidden_dim, args.vocabulary_size).to(args.device))
        self.b_l = nn.Parameter(torch.Tensor(args.vocabulary_size))
        self.linear.data.uniform_(-bound, bound)
        self.b_l.data.uniform_(-bound, bound)
        self.softmax = nn.LogSoftmax(dim=-1)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.embedding(x)
        h_arr, h_t = self.lstm(x, hidden)
        out = h_arr @ self.linear + self.b_l
        out = self.softmax(out)
        if hidden is None:
            return out
        else:
            return h_arr, out

        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=5, sample_length=30, temperature=0.):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = torch.zeros((batch_size, sample_length))
        m = nn.Softmax(dim=-1)
        char = 0
        for i in range(batch_size):
            current = np.random.randint(self.vocabulary_size)
            sample = torch.zeros(sample_length).to(torch.int64).to(self.device)
            sample[0] = current
            hidden = np.array([])
            for j in range(1, sample_length):
                hidden, current = self.forward(sample[None, 0:j].T, hidden)
                if temperature:
                    char = np.random.choice(self.vocabulary_size,
                                            p=m(current[j - 1, 0] / temperature).cpu().detach().numpy())
                else:
                    char = current.argmax(dim=-1)[j - 1, 0]
                sample[j] = char
            out[i] = sample
        return out
        #######################
        # END OF YOUR CODE    #
        #######################
