import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from paramboost.dataset import load_preprocess_LPMC

np.random.seed(1)

dataset_task = {
    "housing": "regression",
    "concrete": "regression",
    "power": "regression",
    "energy": "regression",
    "protein": "regression",
    "msd": "regression",
    "fraud": "binary",
    "diabetes": "binary",
    "cover": "multiclass",
    "har": "multiclass",
    "lpmc": "multiclass",
}

dataset_name_to_loader = {
    "housing": lambda: pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        header=None,
        delim_whitespace=True,
    ),
    "concrete": lambda: pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    ),
    "power": lambda: pd.read_excel("data/uci/power-plant.xlsx"),
    "energy": lambda: pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    ).iloc[:, :-1],
    "protein": lambda: pd.read_csv("data/uci/protein.csv")[
        ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"]
    ],
    "har": lambda: pd.read_csv("data/uci/har/har.csv"),
    "diabetes": lambda: pd.read_csv("data/diabetes/diabetes.csv"),
    "fraud": lambda: pd.read_csv("data/fraud/creditcard.csv"),
    "lpmc": lambda: load_preprocess_LPMC("data/"),
    "cover": lambda: pd.read_csv(
        "data/uci/cover/covtype.data.gz",
        header=None,
        compression="gzip",
    ),    
    "msd": lambda: pd.read_csv("data/uci/YearPredictionMSD.txt").iloc[:, ::-1],
}

try:
    score_df = pd.read_csv("experiment/results/all_results.csv", index_col=[0])
except FileNotFoundError:
    score_df = pd.DataFrame(
        {"train_score": [], "val_score": [], "test_score": [], "time": []}
    )


for d_name in dataset_name_to_loader.keys():
    # load dataset -- use last column as label
    if d_name == "lpmc":
        data_train, data_test, folds = load_preprocess_LPMC(path="data/")
        n_rep = 1
        X = pd.concat([data_train, data_test], axis=0)
        X_trainall, y_trainall = (
            data_train.drop(columns=["choice", "household_id"]),
            data_train["choice"].astype(int),
        )
        X_test, y_test = (
            data_test.drop(columns=["choice", "household_id"]),
            data_test["choice"].astype(int),
        )
    else:
        data = dataset_name_to_loader[d_name]()

        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        if X.shape[1] > 100 or X.shape[0] > 100000:
            n_rep = 1
        else:
            n_rep = 10

        if d_name == "cover":
            y = y - 1
        elif d_name == "har":
            X.columns = [str(i) for i in range(X.shape[1])]

        X.columns = [str(i) for i in X.columns]

        if dataset_task[d_name] in ["binary", "multiclass"]:
            y = y.astype(int)

        print("== Dataset=%s X.shape=%s" % (d_name, str(X.shape)))

        if d_name == "msd":
            folds = [(np.arange(463715), np.arange(463715, len(X)))]
        else:
            # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
            n = X.shape[0]
            np.random.seed(1)
            folds = []
            for i in range(n_rep):
                permutation = np.random.choice(range(n), n, replace=False)
                end_train = round(n * 8.0 / 10)
                end_test = n

                train_index = permutation[0:end_train]
                test_index = permutation[end_train:n]
                folds.append((train_index, test_index))

    for itr, (train_index, test_index) in enumerate(folds):
        if d_name == "lpmc":
            if itr > 0:
                break
            X_train, y_train = (
                X_trainall.iloc[train_index],
                y_trainall.iloc[train_index],
            )
            X_val, y_val = X_trainall.iloc[test_index], y_trainall.iloc[test_index]
        else:
            X_trainall, X_test = X.iloc[train_index], X.iloc[test_index]
            y_trainall, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train, X_val, y_train, y_val = train_test_split(
                X_trainall, y_trainall, test_size=0.125, random_state=1
            )
        save_data_path = f"experiment/data/{d_name}/iteration_{itr}/"
        os.makedirs(save_data_path, exist_ok=True)
        X_train.to_csv(f"{save_data_path}/X_train.csv", index=False)
        X_val.to_csv(f"{save_data_path}/X_val.csv", index=False)
        X_test.to_csv(f"{save_data_path}/X_test.csv", index=False)
        y_train.to_csv(f"{save_data_path}/y_train.csv", index=False)
        y_val.to_csv(f"{save_data_path}/y_val.csv", index=False)
        y_test.to_csv(f"{save_data_path}/y_test.csv", index=False)