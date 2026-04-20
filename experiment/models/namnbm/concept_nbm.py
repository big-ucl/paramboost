# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict
from itertools import combinations

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU


class ConceptNNBasesNary(nn.Module):
    """Neural Network learning bases."""

    def __init__(
        self, order, num_bases, hidden_dims, dropout=0.0, batchnorm=False, device=torch.device("cuda")
    ) -> None:
        """Initializes ConceptNNBases hyperparameters.
        Args:
            order: Order of N-ary concept interatctions.
            num_bases: Number of bases learned.
            hidden_dims: Number of units in hidden layers.
            dropout: Coefficient for dropout regularization.
            batchnorm (True): Whether to use batchnorm or not.
        """
        super(ConceptNNBasesNary, self).__init__()

        assert order > 0, "Order of N-ary interactions has to be larger than '0'."

        layers = []
        self._model_depth = len(hidden_dims) + 1
        self._batchnorm = batchnorm

        # First input_dim depends on the N-ary order
        input_dim = order
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features=input_dim, out_features=dim, device=device))
            if self._batchnorm is True:
                layers.append(nn.BatchNorm1d(dim, device=device))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(ReLU())
            input_dim = dim

        # Last MLP layer
        layers.append(nn.Linear(in_features=input_dim, out_features=num_bases, device=device))
        # Add batchnorm and relu for bases
        if self._batchnorm is True:
            layers.append(nn.BatchNorm1d(num_bases, device=device))
        layers.append(ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConceptNBMNary(nn.Module):
    """
    Neural network (MLP) learns set of bases functions that are global,
    which are then used on each concept feature tuple individually.

    NBM model where higher order interactions of features are modeled in bases
    as f(xi, xj) for order 2 or f(xi, xj, xk) for arbitrary order d.

    ref:
        Neural Basis Models for Interpretability.
        Filip Radenovic, Abhimanyu Dubey, Dhruv Mahajan.
        https://arxiv.org/pdf/2205.14120.pdf
    """

    def __init__(
        self,
        num_concepts,
        num_classes,
        nary=None,
        num_bases=100,
        hidden_dims=(256, 128, 128),
        num_subnets=1,
        dropout=0.0,
        bases_dropout=0.0,
        batchnorm=True,
        output_penalty=0.0,
        device=torch.device("cuda"),
        verbose=1
    ):
        """Initializing NBM hyperparameters.

        Args:
            num_concepts: Number of concepts used as input to the model.
            num_classes: Number of output classes of the model.
            nary (None):
                None:: unary model with all features is initialized.
                List[int]:: list of n-ary orders to be initialized, eg,
                    [1] or [1, 2, 4] or [2, 3].
                Dict[str, List[Tuple]]:: for each order (key) a list of
                    index tuples (value) is given. Only those preselected indices
                    are used in the model. Eg,
                    {"1": [(0, ), (1, ), (2, )], "2": [(0, 1), (1, 2)]}
            num_bases (100): Number of bases learned.
            hidden_dims ([256, 128, 128]): Number of hidden units for neural
                MLP bases part of model.
            num_subnets (1): Number of neural networks used to learn bases.
            dropout (0.0): Coefficient for dropout within neural MLP bases.
            bases_dropout (0.0): Coefficient for dropping out entire basis.
            batchnorm (True): Whether to use batchnorm or not.
            polynomial (None): Supply SPAM initialization here to train NBM-SPAM.
                Note: if polynomial is not None, nary has to be of order 1.
        """
        super(ConceptNBMNary, self).__init__()

        self._num_concepts = num_concepts
        self._num_classes = num_classes
        self._num_bases = num_bases
        self._num_subnets = num_subnets
        self._batchnorm = batchnorm
        self._output_penalty = output_penalty
        self._verbose = verbose

        if nary is None:
            # if no nary specified, unary model is initialized
            self._nary_indices = {"1": list(combinations(range(self._num_concepts), 1))}
        elif isinstance(nary, list):
            self._nary_indices = {
                str(order): list(combinations(range(self._num_concepts), order))
                for order in nary
            }
        elif isinstance(nary, dict):
            self._nary_indices = nary
        else:
            raise TypeError("'nary': None or list or dict supported")

        self.bases_nary_models = nn.ModuleDict()
        for order in self._nary_indices.keys():
            for subnet in range(self._num_subnets):
                self.bases_nary_models[
                    self.get_key(order, subnet)
                ] = ConceptNNBasesNary(
                    order=int(order),
                    num_bases=self._num_bases,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    batchnorm=batchnorm,
                    device=device,
                )

        self.bases_dropout = nn.Dropout(p=bases_dropout)

        num_out_features = (
            sum(len(self._nary_indices[order]) for order in self._nary_indices.keys())
            * self._num_subnets
        )
        self.featurizer = nn.Conv1d(
            in_channels=num_out_features * self._num_bases,
            out_channels=num_out_features,
            kernel_size=1,
            groups=num_out_features,
            device=device,
        )

        self._use_spam = False
        self.classifier = nn.Linear(
            in_features=num_out_features,
            out_features=self._num_classes,
            bias=True,
            device=device,
        )

    def get_key(self, order, subnet):
        return f"ord{order}_net{subnet}"

    def forward(self, input):
        bases = []
        for order in self._nary_indices.keys():
            for subnet in range(self._num_subnets):
                input_order = input[:, self._nary_indices[order]]

                bases.append(
                    self.bases_dropout(
                        self.bases_nary_models[self.get_key(order, subnet)](
                            input_order.reshape(-1, input_order.shape[-1])
                        ).reshape(input_order.shape[0], input_order.shape[1], -1)
                    )
                )

        bases = torch.cat(bases, dim=-2)

        out_feats = self.featurizer(bases.reshape(input_order.shape[0], -1, 1)).squeeze(
            -1
        )

        if self._use_spam:
            out = []
            for _poly_idx in range(len(self._spam)):
                _start_idx = _poly_idx * self._spam_num_out_features
                _end_idx = (_poly_idx + 1) * self._spam_num_out_features
                out.append(self._spam[_poly_idx](out_feats[:, _start_idx:_end_idx]))
            out = torch.sum(torch.stack(out, dim=-1), dim=-1)
        else:
            out = self.classifier(out_feats)

        if self.training:
            return out
        else:
            return out


    def predict(self, x):
        with torch.no_grad():
            self.eval()
            return self.forward(x)
            

    def fit(self, x, y, x_test, y_test, n_epochs=200, batch_size=512, patience=10, lr=0.1, criterion=nn.CrossEntropyLoss(), weight_decay=1e-5):    
        """
        Fit the model.

        Parameters
        ----------
        x : torch.Tensor
            Training input tensor.
        y : torch.Tensor
            Training output tensor.
        x_test : torch.Tensor
            Test input tensor.
        y_test : torch.Tensor
            Test output tensor.
        n_epochs : int
            Number of epochs to train.
        batch_size : int
            Batch size for training.
        patience : int
            Patience for early stopping.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = criterion
        best_loss = torch.inf
        for epoch in range(n_epochs):
            permutation = torch.randperm(x.shape[0])

            for i in range(0, x.shape[0], batch_size):
                indices = permutation[i:i + batch_size]
                x_batch, y_batch = x[indices], y[indices]

                optimizer.zero_grad()
                output = self.forward(x_batch)
                if self._num_classes == 1:
                    output = output.squeeze(1)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                #print(f"Epoch: {epoch}, iteration: {i/self.batch_size}, Loss: {loss.item()}")

            cross_entropy = 0
            for i in range(0, x_test.shape[0], batch_size):
                x_test_batch, y_test_batch = x_test[i:i + batch_size], y_test[i:i + batch_size]
                preds = self.predict(x_test_batch)
                if self._num_classes == 1:
                    preds = preds.squeeze(1)
                cross_entropy += criterion(preds, y_test_batch).item()
            cross_entropy /= math.ceil(x_test.shape[0] / batch_size)
            
            if self._verbose > 0:
                print(f"Epoch: {epoch}-------------------")
                print(f"Train Loss: {loss.item()}")
                print(f"Test Loss: {cross_entropy}")

            if cross_entropy < best_loss:
                best_loss = cross_entropy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    print("Early stopping")
                    break
