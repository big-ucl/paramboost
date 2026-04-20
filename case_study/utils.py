import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from paramboost.dataset import load_preprocess_LPMC

def load_preprocess_return_constraints_LPMC():

    data_train, data_test, folds = load_preprocess_LPMC(path="data/")
    X_trainall, y_trainall = (
        data_train.drop(columns=["choice", "household_id"]),
        data_train["choice"].astype(int),
    )
    X_test, y_test = (
        data_test.drop(columns=["choice", "household_id"]),
        data_test["choice"].astype(int),
    )
    for itr, (train_index, test_index) in enumerate(folds):
        if itr > 0:
            break
        X_train, y_train = (
            X_trainall.iloc[train_index],
            y_trainall.iloc[train_index],
        )
        X_val, y_val = X_trainall.iloc[test_index], y_trainall.iloc[test_index]

    structure = {
        0: [
            "age",
            "female",
            "day_of_week",
            "start_time_linear",
            "car_ownership",
            "driving_license",
            "purpose_B",
            "purpose_HBE",
            "purpose_HBO",
            "purpose_HBW",
            "purpose_NHBO",
            "fueltype_Average",
            "fueltype_Diesel",
            "fueltype_Hybrid",
            "fueltype_Petrol",
            "distance",
            "dur_walking",
        ],
        1: [
            "age",
            "female",
            "day_of_week",
            "start_time_linear",
            "car_ownership",
            "driving_license",
            "purpose_B",
            "purpose_HBE",
            "purpose_HBO",
            "purpose_HBW",
            "purpose_NHBO",
            "fueltype_Average",
            "fueltype_Diesel",
            "fueltype_Hybrid",
            "fueltype_Petrol",
            "distance",
            "dur_cycling",
        ],
        2: [
            "age",
            "female",
            "day_of_week",
            "start_time_linear",
            "car_ownership",
            "driving_license",
            "purpose_B",
            "purpose_HBE",
            "purpose_HBO",
            "purpose_HBW",
            "purpose_NHBO",
            "fueltype_Average",
            "fueltype_Diesel",
            "fueltype_Hybrid",
            "fueltype_Petrol",
            "distance",
            "dur_pt_access",
            "dur_pt_bus",
            "dur_pt_rail",
            "dur_pt_int_waiting",
            "dur_pt_int_walking",
            "pt_n_interchanges",
            "cost_transit",
        ],
        3: [
            "age",
            "female",
            "day_of_week",
            "start_time_linear",
            "car_ownership",
            "driving_license",
            "purpose_B",
            "purpose_HBE",
            "purpose_HBO",
            "purpose_HBW",
            "purpose_NHBO",
            "fueltype_Average",
            "fueltype_Diesel",
            "fueltype_Hybrid",
            "fueltype_Petrol",
            "distance",
            "dur_driving",
            "cost_driving_fuel",
            "congestion_charge",
            "driving_traffic_percent",
        ],
    }
    col_names = X_train.columns.tolist()
    new_structure = []
    for _, struct in structure.items():
        new_struct = []
        for var in struct:
            if var in col_names:
                new_struct.append(col_names.index(var))
        new_structure.append(new_struct)

    mono_cons = {
        "age": 0,
        "female": 0,
        "day_of_week": 0,
        "start_time_linear": 0,
        "car_ownership": 0,
        "driving_license": 0,
        "purpose_B": 0,
        "purpose_HBE": 0,
        "purpose_HBO": 0,
        "purpose_HBW": 0,
        "purpose_NHBO": 0,
        "fueltype_Average": 0,
        "fueltype_Diesel": 0,
        "fueltype_Hybrid": 0,
        "fueltype_Petrol": 0,
        "distance": 0,
        "dur_walking": -1,
        "dur_cycling": -1,
        "dur_pt_access": -1,
        "dur_pt_bus": -1,
        "dur_pt_rail": -1,
        "dur_pt_int_waiting": -1,
        "dur_pt_int_walking": -1,
        "pt_n_interchanges": -1,
        "dur_driving": -1,
        "cost_transit": -1,
        "cost_driving_fuel": -1,
        "congestion_charge": -1,
        "driving_traffic_percent": -1,
    }

    curv_cons = {
        "age": 0,
        "female": 0,
        "day_of_week": 0,
        "start_time_linear": 0,
        "car_ownership": 0,
        "driving_license": 0,
        "purpose_B": 0,
        "purpose_HBE": 0,
        "purpose_HBO": 0,
        "purpose_HBW": 0,
        "purpose_NHBO": 0,
        "fueltype_Average": 0,
        "fueltype_Diesel": 0,
        "fueltype_Hybrid": 0,
        "fueltype_Petrol": 0,
        "distance": 0,
        "dur_walking": 1,
        "dur_cycling": 1,
        "dur_pt_access": 1,
        "dur_pt_bus": -1,
        "dur_pt_rail": -1,
        "dur_pt_int_waiting": 1,
        "dur_pt_int_walking": 1,
        "pt_n_interchanges": 1,
        "dur_driving": 1,
        "cost_transit": 1,
        "cost_driving_fuel": 1,
        "congestion_charge": 1,
        "driving_traffic_percent": 1,
    }

    monotone_constraints = []
    curvature_constraints = []
    for var in col_names:
        monotone_constraints.append(mono_cons[var])
        curvature_constraints.append(curv_cons[var])

    hyperparameters = {
        "num_iterations": 25000,
        "order": 3,
        "continuity": 2,
        "learning_rate": 0.1,
        "max_depth": 1,
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
    return X_train, y_train, X_val, y_val, X_test, y_test, new_structure, monotone_constraints, curvature_constraints, hyperparameters


def plot_features_with_ci(model, dataset, feature_names, num_points=10000, save_name = ""):
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.serif": "Computer Modern Roman",
        # Use 14pt font in plots, to match 10pt font in document
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.labelpad": 1,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "legend.fancybox": False,
        "legend.edgecolor": "inherit",
        "legend.borderaxespad": 0.4,
        "legend.borderpad": 0.4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.pad": 0.5,
        "ytick.major.pad": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8,
    }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    # sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            # "text.usetex": True,
            "font.family": "serif",
            # "font.sans-serif": "Computer Modern Roman",
        }
    )

    x_plot = np.linspace(0, 1, num_points)
    for i, f in feature_names:
        plot_dataset = pd.DataFrame(
            {
                f_i: np.zeros_like(x_plot)
                for f_i in dataset.columns
            },
            columns=dataset.columns
        )
        plot_dataset[f] = x_plot

        y_pred, y_lower, y_upper = model.model.predict_with_ci(
            plot_dataset.values
        )

        intercept = y_pred[0]
        y_pred -= intercept
        y_lower -= intercept
        y_upper -= intercept

        plt.figure(figsize=(6, 4))

        plt.plot(
            x_plot * dataset[f].max(),
            y_pred[:, i],
            label="Shape function with 95% CI",
            color="k",
            linestyle="-",
            linewidth=2,
        )
        plt.fill_between(
            x_plot * dataset[f].max(),
            y_lower[:, i],
            y_upper[:, i],
            color="k",
            alpha=0.2,
        )
        plt.plot(
            x_plot * dataset[f].max(),
            y_lower[:, i],
            color="k",
            linestyle="-",
            linewidth=1,
            alpha = 0.5,
        )
        plt.plot(
            x_plot * dataset[f].max(),
            y_upper[:, i],
            color="k",
            linestyle="-",
            linewidth=1,
            alpha=0.5,
        )

        plt.ylabel(
            "Shape function",
            fontsize=12,
        )
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=11)
        plt.tight_layout()

        if save_name:
            plt.savefig(f"case_study/{save_name}{f}.png")
        plt.show()


def plot_features(models: list, model_names: list, dataset, feature_names, num_points=10000, save_name = ""):
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.serif": "Computer Modern Roman",
        # Use 14pt font in plots, to match 10pt font in document
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.labelpad": 1,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "legend.fancybox": False,
        "legend.edgecolor": "inherit",
        "legend.borderaxespad": 0.4,
        "legend.borderpad": 0.4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.pad": 0.5,
        "ytick.major.pad": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8,
    }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    # sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            # "text.usetex": True,
            "font.family": "serif",
            # "font.sans-serif": "Computer Modern Roman",
        }
    )
    grays_hex = ["#2F2F2F", "#555555", "#808080", "#A9A9A9", "#D3D3D3"]


    x_plot = np.linspace(0, 1, num_points)
    for i, f in feature_names:
        plot_dataset = pd.DataFrame(
            {
                f_i: np.zeros_like(x_plot)
                for f_i in dataset.columns
            },
            columns=dataset.columns
        )
        plot_dataset[f] = x_plot

        plt.figure(figsize=(6, 4))

        for model, model_name in zip(models, model_names):

            y_pred = model.model.predict(
                plot_dataset.values, utilities=True
            )
            y_pred -= y_pred[0]

            plt.plot(
                x_plot * dataset[f].max(),
                y_pred[:, i],
                label=model_name,
                color=grays_hex[model_names.index(model_name)],
                linestyle="-",
                linewidth=2,
            )

        plt.ylabel(
            "Shape function",
            fontsize=12,
        )
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        if i == 3:
            plt.legend(fontsize=11)
        plt.tight_layout()

        if save_name:
            plt.savefig(f"case_study/{save_name}{f}.png")
        plt.show()