from sklearn.linear_model import Ridge, LogisticRegression
import pandas as pd

class LinearModel():
    def __init__(self, _, y, alpha=1.0):
        
        self.model_type = 'regression' if y.unique().shape[0] > 7 else 'classification'
        if self.model_type == 'regression':
            self.model = Ridge(alpha=alpha)
        elif self.model_type == 'classification':
            c = 1.0/alpha
            self.model = LogisticRegression(C=c)
        else:
            raise ValueError('model_type must be either regression or classification')
    
    def fit(self, X_train, y_train, X_val, y_val):
        X = pd.concat([X_train, X_val])
        y = pd.concat([y_train, y_val])
        self.model.fit(X, y)
    
    def predict(self, X):
        if self.model_type == "regression":
            return self.model.predict(X)
        elif self.model_type == "classification":
            return self.model.predict_proba(X)