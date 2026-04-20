import pandas as pd
import numpy as np
from scipy.special import softmax
from pymgcv.gam import GAM
from pymgcv.terms import S, L
from pymgcv.basis_functions import CubicSpline
from pymgcv.families import Binomial, Gaussian, Multinom


class pyMGCV:
    """
    Wrapper class for MGCV model.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        fx: bool = False,
        shrinkage: bool = False,
    ):
        """
        Initialise the MGCV model.

        Parameters:
        -----------
        X_train (pd.DataFrame):
            Training features.
        y_train (pd.Series):
            Training target.
        fx (bool):
            Whether to penalise the model (false means penalised).
        shrinkage (bool):
            Whether to use shrinkage on the CubicSpline basis.
        """
        data = X_train.copy().reset_index(drop=True)
        data.columns = [f"col{i}" for i in range(data.shape[1])]
        terms = {col: (S if data[col].nunique() > 2 else L) for col in data.columns}
        if y_train.unique().shape[0] == 2:
            self.model_type = "binary"
            self.model = GAM(
                {
                    "y": [
                        terms[var](
                            var,
                            fx=fx,
                            bs=CubicSpline(shrinkage=shrinkage),
                            k=min(10, data[var].nunique() - 1),
                        )
                        if isinstance(terms[var], S)
                        else L(var)
                        for var in data.columns
                    ]
                },
                family=Binomial(link="logit"),
            )
        elif y_train.unique().shape[0] > 2 and y_train.unique().shape[0] < 8:
            self.model_type = "multiclass"
            self.model = GAM(
                {
                    f"y{i if i > 0 else ''}": [
                        terms[var](
                            var,
                            fx=fx,
                            bs=CubicSpline(shrinkage=shrinkage),
                            k=min(10, data[var].nunique() - 1),
                        )
                        if isinstance(terms[var], S)
                        else L(var)
                        for var in data.columns
                    ]
                    for i in range(y_train.unique().shape[0]-1)
                },
                family=Multinom(k=y_train.unique().shape[0]-1),
            )
        else:
            self.model_type = "regression"
            self.model = GAM(
                {
                    "y": [
                        terms[var](
                            var,
                            fx=fx,
                            bs=CubicSpline(shrinkage=shrinkage),
                            k=min(10, data[var].nunique() - 1),
                        )
                        if isinstance(terms[var], S)
                        else L(var)
                        for var in data.columns
                    ]
                },
                family=Gaussian(link="identity"),
            )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Fit the MGCV model.

        Parameters:
        -----------
        X_train (pd.DataFrame):
            Training features.
        y_train (pd.Series):
            Training target.
        """
        data = X_train.copy().reset_index(drop=True)
        data.columns = [f"col{i}" for i in range(data.shape[1])]
        data = pd.concat([data, y_train.reset_index(drop=True).rename("y")], axis=1)
        self.model.fit(data)

    def predict(self, X_test):
        """
        Predict using the fitted MGCV model.

        Parameters:
        -----------
        X_test (pd.DataFrame):
            Test data.

        Returns:
        np.array: Predicted values.
        """
        data = X_test.copy().reset_index(drop=True)
        data.columns = [f"col{i}" for i in range(data.shape[1])]
        if self.model_type == "multiclass":
            preds_dict = self.model.predict(
                data, type="link"
            )  # weird bug with predict type="response",
            # returns about 25% more predictions than in dataset
            # so we do the inverse link manually
            preds_array = np.zeros((data.shape[0], len(preds_dict)+1))
            for i, value in enumerate(preds_dict.values()):
                preds_array[:, i+1] = value
            preds_array[:, 0] = 1 - preds_array.sum(axis=1)
            preds = softmax(preds_array, axis=1)
        else:
            preds = self.model.predict(data, type="response")["y"]
        return preds
