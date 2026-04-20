from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
import pandas as pd
import numpy as np

class EBM():
    def __init__(self, _, y, **kwargs):

        self.model_type = 'regression' if y.unique().shape[0] > 7 else 'classification'
        if self.model_type == 'regression':
            self.model = ExplainableBoostingRegressor(interactions = 0., validation_size=0.125, outer_bags=1, random_state=1, **kwargs)
        elif self.model_type == 'classification':
            self.model = ExplainableBoostingClassifier(interactions = 0., validation_size=0.125, outer_bags=1, random_state=1, **kwargs)
        else:
            raise ValueError('task must be either regression or classification')

    def fit(self, X_train, y_train, X_val, y_val):
        # bag_training = np.ones(X_train.shape[0])
        # bag_val = - np.ones(X_val.shape[0])
        # bags = np.concatenate([bag_training, bag_val]).reshape(-1, 1)
        # X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        # y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        # self.model.fit(X_combined.values, y_combined.values, bags=bags)
        self.model.fit(X_train.values, y_train.values)

    def predict(self, X):
        if self.model_type == "regression":
            return self.model.predict(X)
        elif self.model_type == "classification":
            return self.model.predict_proba(X)