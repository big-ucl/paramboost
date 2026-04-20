import sys

sys.path.append("experiment/")

import torch
from .namnbm.concept_nbm import ConceptNBMNary


class NBM:
    def __init__(
        self,
        X,
        y,
        n_bases=100,
        n_neurons=[256, 128, 128],
        dropout=0.0,
        learning_rate=0.1,
        n_epochs=200,
        patience=10,
        batch_size=1024,
        weight_decay=0.0,
        verbose=0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_concepts = X.shape[1]
        if y.unique().shape[0] == 2:
            self.task = "binary"
            n_classes = 1
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif y.unique().shape[0] > 2 and y.unique().shape[0] < 8:
            self.task = "multiclass"
            n_classes = y.unique().shape[0]
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.task = "regression"
            n_classes = 1
            self.criterion = torch.nn.MSELoss()
        self.n_epochs = n_epochs
        self.patience = patience
        self.lr = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConceptNBMNary(
            num_concepts=self.num_concepts,
            num_classes=n_classes,
            num_bases=n_bases,
            hidden_dims=n_neurons,
            dropout=dropout,
            device=device,
            verbose=verbose,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = torch.tensor(X_train.values).to(self.device).float()
        y_train = torch.tensor(y_train.values).to(self.device)
        X_val = torch.tensor(X_val.values).to(self.device).float()
        y_val = torch.tensor(y_val.values).to(self.device)

        if self.task != "multiclass":
            y_train = y_train.float()
            y_val = y_val.float()

        self.model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            weight_decay=self.weight_decay,
            lr=self.lr,
            criterion=self.criterion,
        )

    def predict(self, X):
        X = torch.tensor(X.values).float()
        for i in range(0, X.shape[0], self.batch_size):
            X_batch = X[i : i + self.batch_size].to(self.device)
            if self.task == "multiclass":
                preds = torch.softmax(self.model.predict(X_batch), dim=1)
            elif self.task == "binary":
                preds = torch.sigmoid(self.model.predict(X_batch))
            else:
                preds = self.model.predict(X_batch)
            if i == 0:
                all_preds = preds.cpu()
            else:
                all_preds = torch.cat((all_preds, preds.cpu()), dim=0)
        return all_preds.numpy()
