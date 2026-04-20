# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from itertools import combinations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU

class ConceptNNNary(nn.Module):
    """
    Neural Network model for N-ary concept interactions.
    Neural Network takes N-ary input and outputs a single value f(x1, ..., xN).

    v2: Implemented using depthwise 1d convolutions, also knows as group convs.
    """

    def __init__(
        self, order, num_mlps, first_layer, hidden_dims, dropout=0.0, batchnorm=False, device=torch.device("cuda")
    ) -> None:
        """Initializes ConceptNN hyperparameters.
        Args:
            order: Order of N-ary concept interatctions.
            num_mlps: Number of N-ary MLPs to be grouped in convolutions.
            first_layer: First layer can either be 'exu' or 'linear'.
            hidden_dims: Number of units in hidden layers.
            dropout: Coefficient for dropout regularization.
            batchnorm (True): Whether to use batchnorm or not.
        """
        super(ConceptNNNary, self).__init__()

        assert order > 0, "Order of N-ary interactions has to be larger than '0'."

        assert (
            first_layer == "exu" or first_layer == "linear"
        ), "First layer can either be 'exu' or 'linear'."

        layers = []
        self._model_depth = len(hidden_dims) + 1
        self._batchnorm = batchnorm

        if len(hidden_dims) == 1 and hidden_dims[0] == 0:
            # No MLP, simply [order x 1] linear layer as NN
            input_dim = order * num_mlps
        else:
            if first_layer == "exu":
                # First layer is ExU followed by ReLUn
                raise TypeError("'first_layer': 'exu' not implemented yet")
            elif first_layer == "linear":
                # First layer is Linear followed by ReLU
                layers.append(
                    nn.Conv1d(
                        in_channels=order * num_mlps,
                        out_channels=hidden_dims[0] * num_mlps,
                        kernel_size=1,
                        groups=num_mlps,
                        device=device
                    )
                )
                if self._batchnorm is True:
                    layers.append(nn.BatchNorm1d(hidden_dims[0] * num_mlps, device=device))
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                layers.append(ReLU())
            input_dim = hidden_dims[0]

            # Followed by linear layers and ReLU
            for dim in hidden_dims[1:]:
                layers.append(
                    nn.Conv1d(
                        in_channels=input_dim * num_mlps,
                        out_channels=dim * num_mlps,
                        kernel_size=1,
                        groups=num_mlps,
                        device=device
                    )
                )
                if self._batchnorm is True:
                    layers.append(nn.BatchNorm1d(dim * num_mlps, device=device))
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                layers.append(ReLU())
                input_dim = dim

        # Last linear layer
        layers.append(
            nn.Conv1d(
                in_channels=input_dim * num_mlps,
                out_channels=1 * num_mlps,
                kernel_size=1,
                groups=num_mlps,
                device=device
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConceptNAMNary(nn.Module):
    """
    Neural Additive Models with higher order (N-ary) concept interactions.
    v2: Implemented using depthwise 1d convolutions, also knows as group convs.

    ref:
        Neural Additive Models: Interpretable Machine Learning with Neural Nets,
        Agarwal etal, NeurIPS 2021
        https://arxiv.org/pdf/2004.13912.pdf
    """

    def __init__(
        self,
        num_concepts,
        num_classes,
        nary=None,
        first_layer="linear",
        first_hidden_dim=64,
        hidden_dims=(64, 32),
        num_subnets=1,
        dropout=0.0,
        concept_dropout=0.0,
        batchnorm=True,
        output_penalty=0.0,
        polynomial=None,
        device=torch.device("cuda"),
        verbose=1
    ):
        """Initialized NAM hyperparameters.

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
            first_layer ('linear'): First layer can either be 'exu' or 'linear'.
            first_hidden_dim (64): Number of hidden units in the first layer for
                each ConceptNN:
                    if `int` it is repeated for each input concept,
                    if `list` NOT IMPLEMENTED yet.
                    if `dict` NOT IMPLEMENTED yet.
            hidden_dims (None): Number of hidden units in all subsequent linear layers,
                if empty list `[]` or `None` that means there are no more hidden layers,
                just the first hidden layer.
            num_subnets (1): Number of neural networks for each input concept.
            dropout (0.0): Coefficient for dropout within each ConceptNN.
            concept_dropout (0.0): Coefficient for dropping out entire ConceptNN.
            batchnorm (True): Whether to use batchnorm or not.
            polynomial (None): Supply SPAM initialization here to train SPAM-Neural.
        """
        super(ConceptNAMNary, self).__init__()
        self._num_concepts = num_concepts
        self._num_classes = num_classes
        self._num_subnets = num_subnets
        if polynomial:
            self._num_subnets_per_polynomial = num_subnets
            self._num_subnets = (len(polynomial["ranks"]) + 1) * num_subnets
        self._batchnorm = batchnorm
        self._output_penalty = output_penalty
        self._verbose = verbose

        self.concept_dropout = nn.Dropout(p=concept_dropout)

        if isinstance(first_hidden_dim, int):
            self._first_hidden_dim = first_hidden_dim
        else:
            raise TypeError("'first_hidden_dim': only int is supported")

        if hidden_dims is None:
            hidden_dims = []
        elif isinstance(hidden_dims, list):
            hidden_dims = list(hidden_dims)

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

        self.concept_nary_nns = nn.ModuleDict()
        for order in self._nary_indices.keys():
            for subnet in range(self._num_subnets):
                self.concept_nary_nns[self.get_key(order, subnet)] = ConceptNNNary(
                    order=int(order),
                    num_mlps=len(self._nary_indices[order]),
                    first_layer=first_layer,
                    hidden_dims=[self._first_hidden_dim] + hidden_dims,
                    dropout=dropout,
                    batchnorm=self._batchnorm,
                    device=device
                )

        weight_in_feature_size = (
            sum(len(self._nary_indices[order]) for order in self._nary_indices.keys())
            * self._num_subnets
        )
        self._use_spam = False
        self._weight = nn.Parameter(
            torch.empty((self._num_classes, weight_in_feature_size), device=device)
        )
        self._bias = nn.Parameter(torch.empty(self._num_classes, device=device))

        self.reset_parameters()

    def get_key(self, order, subnet):
        return f"ord{order}_net{subnet}"

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self._weight, a=math.sqrt(5))
        if self._bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self._bias, -bound, bound)

    def forward(self, input):
        out_nn = []
        for order in self._nary_indices.keys():
            for subnet in range(self._num_subnets):
                out_nn.append(
                    self.concept_nary_nns[self.get_key(order, subnet)](
                        input[:, self._nary_indices[order]].reshape(
                            input.shape[0], -1, 1
                        )
                    ).squeeze(-1)
                )
        if self._use_spam:
            _concat_output, out = [], []
            for i, _output in enumerate(out_nn):
                _concat_output.append(_output)
                _poly_idx = i // self._num_subnets_per_polynomial
                if (i + 1) % self._num_subnets_per_polynomial == 0:
                    _concat_output = torch.cat(_concat_output, dim=-1)
                    if _poly_idx == 0:
                        out.append(
                            F.linear(
                                self.concept_dropout(_concat_output),
                                self._weight,
                                self._bias,
                            )
                        )
                    else:
                        out.append(
                            self._spam[_poly_idx - 1](
                                self.concept_dropout(_concat_output)
                            )
                        )
                    _concat_output = []
            out = torch.sum(torch.stack(out, dim=-1), dim=-1)
        out_nn = torch.cat(out_nn, dim=-1)
        if not self._use_spam:
            out = F.linear(self.concept_dropout(out_nn), self._weight, self._bias)
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
