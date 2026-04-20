import pandas as pd
import numpy as np
import pickle
from collections import defaultdict, Counter
import random
import sys
sys.path.append('../')

def load_preprocess_LPMC(
    path="/Data/",
):
    """
    Load and preprocess the LPMC dataset.

    Returns
    -------
    dataset_train : pandas Dataframe
        The training dataset ready to use.
    dataset_test : pandas Dataframe
        The training dataset ready to use.
    folds : zip(list, list)
        5 folds of indices grouped by household for CV.
    """
    # source: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
    data_train = pd.read_csv(path + "LPMC_train.csv")
    data_test = pd.read_csv(path + "LPMC_test.csv")

    # data_train_2 = pd.read_csv('Data/LTDS_train.csv')
    # data_test_2 = pd.read_csv('Data/LTDS_test.csv')

    # distance in km
    data_train["distance"] = data_train["distance"] / 1000
    data_test["distance"] = data_test["distance"] / 1000

    # rename label
    label_name = {"travel_mode": "choice"}
    dataset_train = data_train.rename(columns=label_name)
    dataset_test = data_test.rename(columns=label_name)

    # get all features
    target = "choice"
    features = [f for f in dataset_test.columns if f != target]

    # get household ids
    hh_id = np.array(data_train["household_id"].values)

    # k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(
            open(
                path + "strat_group_k_fold_london.pickle",
                "rb",
            )
        )
    except FileNotFoundError:
        for train_i, test_i in stratified_group_k_fold(
            data_train[features], data_train[target], hh_id, k=5
        ):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump(
            [train_idx, test_idx],
            open(
                path + "strat_group_k_fold_london.pickle",
                "wb",
            ),
        )

    folds = zip(train_idx, test_idx)

    return dataset_train, dataset_test, folds

# Sample a dataset grouped by `groups` and stratified by `y`
# Source: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
def stratified_group_k_fold(X, y, groups, k, seed=None):
    """
    Stratified Group K-Fold cross-validator
    Provides train/test indices to split data in train/test sets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    groups : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into train/test set.
    k : int
        Number of folds. Must be at least 2.
    seed : int, optional
        Random seed for shuffling the data.

    Yields
    ------
    train : ndarray
        The training set indices for that split.
    test : ndarray
        The testing set indices for that split.
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
            )
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices
