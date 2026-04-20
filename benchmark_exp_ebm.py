from argparse import ArgumentParser
import sys

sys.path.append("../")
import gc
import os
import time
import json
import copy

import numpy as np
import pandas as pd
# import torch
# import jax
from sklearn.metrics import mean_squared_error, log_loss

from experiment.models.ebm import EBM
# from experiment.models.nam import NAM
# from experiment.models.nbm import NBM
# from experiment.models.linear_model import LinearModel
# from experiment.models.pymgcv import pyMGCV
# from experiment.models.paramb import ParamB
# from experiment.models.aplr import APLR


# jax.config.update("jax_default_matmul_precision", "float32")

# np.random.seed(1)
# torch.manual_seed(1)


def generate_hyperparameters(model_name, X_train, big_dataset=False):
    if model_name == "nam":
        hyperparameters = {
            "dropout": float(np.random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])),
            "learning_rate": np.random.uniform(0.0001, 0.1),
            "weight_decay": np.random.uniform(1e-6, 1e-3),
        }
    elif model_name == "nbm":
        hyperparameters = {
            "dropout": float(np.random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])),
            "learning_rate": np.random.uniform(0.0001, 0.1),
            "weight_decay": np.random.uniform(1e-6, 1e-3),
        }
    elif model_name == "ebm":
        hyperparameters = {
            "max_bins": int(np.random.choice([64, 128, 256, 512])),
            "min_samples_leaf": int(np.random.choice([1, 2, 4, 8, 10, 15, 20])),
            "min_hessian": float(np.random.choice([0.1, 0.01, 0.001, 0.0001])),
            "learning_rate": np.random.uniform(0.001, 0.1),
        }
    elif model_name == "linear_model":
        hyperparameters = {
            "alpha": np.random.uniform(1e-5, 1),
        }
    elif model_name == "xgboost":
        hyperparameters = {
            "num_iterations": 25000,
        }
    elif model_name == "pymgcv":
        hyperparameters = {
            "fx": bool(np.random.choice([True, False])),
            "shrinkage": bool(np.random.choice([True, False])),
        }
    elif model_name == "aplr":
        hyperparameters = {
            "early_stopping_rounds": 100,
            "m": 25000,
            "min_observations_in_split": int(np.random.choice([1, 2, 4, 8, 10, 15, 20])),
            "v": np.random.uniform(0.001, 0.1),
            "ridge_penalty": float(np.random.choice([0.1, 0.01, 0.001, 0.0001])),
            "bins": 300,
        }
    elif model_name == "paramb":
        hyperparameters = {
            "num_iterations": 25000,
            "learning_rate": 0.1,
            "max_depth": 1,
            "monotone_constraints": None,
            "boosting_level": 0,
            "verbosity": 0,
            "lambda_l1": 1e-3,
            "lambda_l2": 1e-2,
            "bagging_fraction": 1,
            "bagging_freq": 0,
            "min_data_in_leaf": 10,
            "min_data_in_bin": 1,
            "min_sum_hessian_in_leaf": 0,
            "min_gain_to_split": 0,
            "seed": 1,
            "max_bins": 256,
            "max_bins_poly": 20,
            "early_stopping_rounds": 100,
        }
    return hyperparameters


num_searches = 1

model_list = {
    "ebm": EBM,
    # "nam": NAM,
    # "nbm": NBM,
    # "linear_model": LinearModel,
    # "pymgcv": pyMGCV,
    # "paramb": ParamB,
    # "aplr": APLR,
}

metrics_list = {
    "regression": mean_squared_error,
    "binary": log_loss,
    "multiclass": log_loss,
}

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
dataset_iterations = {
    "housing": 10,
    "concrete": 10,
    "power": 10,
    "energy": 10,
    "protein": 1,
    "msd": 1,
    "fraud": 1,
    "diabetes": 1,
    "cover": 1,
    "har": 1,
    "lpmc": 1,
}
dataset_hyper_num_searches = {
    "housing": 50,
    "concrete": 50,
    "power": 50,
    "energy": 50,
    "protein": 15,
    "msd": 15,
    "fraud": 15,
    "diabetes": 15,
    "cover": 15,
    "har": 15,
    "lpmc": 15,
}

try:
    score_df = pd.read_csv("experiment/results/all_results.csv", index_col=[0])
except FileNotFoundError:
    score_df = pd.DataFrame(
        {"train_score": [], "val_score": [], "test_score": [], "time": []}
    )

for model_name, model in model_list.items():
    for d_name in dataset_task.keys():
        for itr in range(dataset_iterations[d_name]):
            save_data_path = f"experiment/data/{d_name}/iteration_{itr}/"

            if d_name == "msd":
                name = [str(i) for i in range(90)]
                names_y = ["target"]
            else:
                name = None
                names_y = None
            X_train = pd.read_csv(f"{save_data_path}/X_train.csv", names=name)
            X_val = pd.read_csv(f"{save_data_path}/X_val.csv", names=name)
            X_test = pd.read_csv(f"{save_data_path}/X_test.csv", names=name)
            y_train = pd.read_csv(f"{save_data_path}/y_train.csv", names=names_y).iloc[:, 0]
            y_val = pd.read_csv(f"{save_data_path}/y_val.csv", names=names_y).iloc[:, 0]
            y_test = pd.read_csv(f"{save_data_path}/y_test.csv", names=names_y).iloc[:, 0]

            for order in [0, 1, 2, 3]:
                if model_name not in ["paramb"] and order > 0:
                    continue
                continuities = [i for i in range(-1, order)]
                for continuity in continuities:
                    paramb_str = (
                        f"_order{order}_cont{continuity}"
                        if model_name
                        in [
                            "paramb",
                        ]
                        else ""
                    )

                    path = f"experiment/results/{d_name}/{itr}/{model_name}"
                    os.makedirs(path, exist_ok=True)

                    # if os.path.exists(os.path.join(path, f"results{paramb_str}.csv")):
                    #     print(
                    #         f"Skipping {model_name} on {d_name} dataset iteration {itr}, already trained"
                    #     )
                    #     continue

                    if model_name in [
                        "paramb",
                    ]:
                        num_searches = 1
                    else:
                        num_searches = dataset_hyper_num_searches[d_name]

                    best_train_score = 0
                    best_val_score = np.inf
                    best_test_score = 0
                    best_time = 0
                    best_hyperparameters = None
                    best_iter = 0
                    for i in range(num_searches):
                        print(
                            f"Training {model_name} on {d_name} dataset, iteration {i + 1}/{num_searches}"
                        )

                        hyperparameters = generate_hyperparameters(
                            model_name,
                            X_train,
                            big_dataset=dataset_hyper_num_searches[d_name] == 15,
                        )

                        if model_name in [
                            "paramb",
                        ]:
                            hyperparameters["order"] = order
                            hyperparameters["continuity"] = continuity

                        current_model = model(X_train, y_train, **hyperparameters)

                        try:
                            start_time = time.time()
                            current_model.fit(X_train, y_train, X_val, y_val)
                            end_time = time.time() - start_time
                        except Exception as e:
                            print(
                                f"Error training {model_name} on {d_name} dataset, iteration {i + 1}/{num_searches}: {e}"
                            )
                            continue

                        y_train_preds = current_model.predict(X_train)
                        y_val_preds = current_model.predict(X_val)
                        y_test_preds = current_model.predict(X_test)

                        train_score = metrics_list[dataset_task[d_name]](
                            y_train, y_train_preds
                        )
                        val_score = metrics_list[dataset_task[d_name]](
                            y_val, y_val_preds
                        )
                        test_score = metrics_list[dataset_task[d_name]](
                            y_test, y_test_preds
                        )

                        if val_score < best_val_score:
                            best_train_score = train_score
                            best_val_score = val_score
                            best_test_score = test_score
                            best_hyperparameters = hyperparameters
                            best_time = end_time
                            if model_name in [
                                "paramb",
                            ]:
                                best_iter = current_model.model.best_iteration

                        del current_model
                        gc.collect()
                        # torch.cuda.empty_cache()

                    # Save hyperparameters and performance results
                    with open(
                        os.path.join(path, f"hyperparameters{paramb_str}.json"), "w"
                    ) as f:
                        json.dump(best_hyperparameters, f)

                    index = pd.MultiIndex.from_tuples(
                        [(d_name, model_name, itr)],
                        names=["dataset", "model", "iteration"],
                    )
                    current_score_df = pd.DataFrame(
                        {
                            "train_score": best_train_score,
                            "val_score": best_val_score,
                            "test_score": best_test_score,
                            "time": best_time,
                            "best_iter": best_iter,
                        },
                        index=index,
                    )
                    current_score_df.to_csv(
                        os.path.join(path, f"results{paramb_str}.csv"),
                        sep=",",
                        index=True,
                    )

                    score_df = pd.concat([score_df, current_score_df], axis=0)


score_df.to_csv("experiment/results/all_results.csv", sep=",", index=True)
