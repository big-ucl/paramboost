from paramboost.paramboost import ParamBoost, train
import pandas as pd
import numpy as np


class ParamB:
    """
    Wrapper class for ParamBoost model.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        self.structure = kwargs.get("structure", None)

        if y_train.unique().shape[0] == 2:
            self.num_classes = 2
            self.task = "binary"
            if not self.structure:
                var_names = X_train.columns.tolist()
                self.structure = kwargs.get("structure",[[var_names.index(var) for var in var_names]])
        elif y_train.unique().shape[0] > 2 and y_train.unique().shape[0] < 8:
            self.num_classes = y_train.unique().shape[0]
            self.task = "multiclass"
            if not self.structure:
                var_names = X_train.columns.tolist()
                self.structure = kwargs.get("structure",[[var_names.index(var) for var in var_names] for _ in range(self.num_classes)])
        else:
            self.num_classes = 1
            self.task = "regression"
            if not self.structure:
                var_names = X_train.columns.tolist()
                self.structure = kwargs.get("structure",[[var_names.index(var) for var in var_names]])
        self.order = kwargs.get("order", 0)
        self.continuity = kwargs.get("continuity", -1)
        self.max_bins = kwargs.get("max_bins", 64)
        self.max_bins_poly = kwargs.get("max_bins_poly", 16)

        self.num_iterations = kwargs.get("num_iterations", 3000)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.max_depth = 1
        self.monotone_constraints = kwargs.get("monotone_constraints", None)
        self.curvature_constraints = kwargs.get("curvature_constraints", None)
        self.early_stopping_rounds = kwargs.get("early_stopping_rounds", None)
        self.boosting_level = kwargs.get("boosting_level", 2)
        self.verbosity = kwargs.get("verbosity", 0)
        self.lambda_l1 = kwargs.get("lambda_l1", 0)
        self.lambda_l2 = kwargs.get("lambda_l2", 0)
        self.bagging_fraction = kwargs.get("bagging_fraction", 1)
        self.bagging_freq = kwargs.get("bagging_freq", 0)
        self.min_data_in_leaf = kwargs.get("min_data_in_leaf", 20)
        self.min_data_in_bin = kwargs.get("min_data_in_bin", 3)
        self.min_sum_hessian_in_leaf = kwargs.get("min_sum_hessian_in_leaf", 1e-3)
        self.min_gain_to_split = kwargs.get("min_gain_to_split", 0)
        self.seed = kwargs.get("seed", 1)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ):
        """
        Fits the model to the training data.
        """
        self.model = ParamBoost(
            num_classes=self.num_classes,
            order=self.order,
            continuity=self.continuity,
            max_bins=self.max_bins,
            max_bins_poly=self.max_bins_poly,
            data=X_train,
            data_val=X_valid,
            labels=y_train,
            labels_val=y_valid,
            monotone_constraints=self.monotone_constraints,
            curvature_constraints=self.curvature_constraints,
            boosting_level=self.boosting_level,
            verbosity=self.verbosity,
            task=self.task,
            structure=self.structure,
            num_iterations=self.num_iterations,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_data_in_bin=self.min_data_in_bin,
            min_data_in_leaf=self.min_data_in_leaf,
            lambda_l1=self.lambda_l1,
            lambda_l2=self.lambda_l2,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            min_sum_hessian_in_leaf=self.min_sum_hessian_in_leaf,
            min_gain_to_split=self.min_gain_to_split,
            seed=self.seed,
            early_stopping_rounds=self.early_stopping_rounds,
        )

        self.model = train(self.model)

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
        assert hasattr(self, "model"), (
            "Model not trained yet. Please train the model before predicting."
        )
        preds = self.model.predict(X_test)
        return preds
