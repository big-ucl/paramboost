from aplr import APLRClassifier, APLRRegressor
import pandas as pd
import numpy as np

class APLR:
    """
    Wrapper class for APLR model.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):

        if y_train.unique().shape[0] == 2:
            self.model_type = "binary"
            self.model = APLRClassifier(
                max_interaction_level=0,
                early_stopping_rounds=kwargs.get("early_stopping_rounds", 100),
                m = kwargs.get("m", 25000),
                min_observations_in_split = kwargs.get("min_observations_in_split", 4),
                v = kwargs.get("v", 0.5),
                ridge_penalty = kwargs.get("ridge_penalty", 0.0001),
                bins = kwargs.get("bins", 256),
                random_state=1,
                cv_folds=2,
            )
        elif y_train.unique().shape[0] > 2 and y_train.unique().shape[0] < 8:
            self.model_type = "multiclass"
            self.model = APLRClassifier(
                max_interaction_level=0,
                early_stopping_rounds=kwargs.get("early_stopping_rounds", 100),
                m = kwargs.get("m", 25000),
                min_observations_in_split = kwargs.get("min_observations_in_split", 4),
                v = kwargs.get("v", 0.5),
                ridge_penalty = kwargs.get("ridge_penalty", 0.0001),
                bins = kwargs.get("bins", 256),
                random_state=1,
                cv_folds=2,
            )
        else:
            self.model_type = "regression"
            self.model = APLRRegressor(
                max_interaction_level=0,
                early_stopping_rounds=kwargs.get("early_stopping_rounds", 100),
                m = kwargs.get("m", 25000),
                min_observations_in_split = kwargs.get("min_observations_in_split", 4),
                v = kwargs.get("v", 0.5),
                ridge_penalty = kwargs.get("ridge_penalty", 0.0001),
                bins = kwargs.get("bins", 256),
                random_state=1,
                cv_folds=2,
            )

        self.monotone_constraints = kwargs.get("monotone_constraints", [])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Fits the model to the training data.
        """
        self.model.fit(X_train.values, y_train.values, monotonic_constraints=self.monotone_constraints)

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Predicts the target variable for the test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.

        Returns
        -------
        preds : np.array
            Predicted target variable, as probabilities.
        """
        if self.model_type == "regression":
            preds = self.model.predict(X_test.values)
            return preds
        
        preds = self.model.predict_class_probabilities(X_test.values)
        return preds