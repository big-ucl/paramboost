import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from jax import jit, vmap
from functools import partial
from paramboost.histogram_builder import BinMapper
from paramboost.utils import (
    draw_new_indices,
    _one_hot_encode_labels,
    print_cel,
    _compute_standard_error,
    _create_non_constant_idx,
)
from paramboost.predictions import (
    _init_predictions_classification,
    _init_predictions_regression,
    predict_vectorised,
    _get_bin_indices,
    predict_with_se,
)
from paramboost.objective import f_obj_cel, f_obj_binary, f_obj_regression
from paramboost.loss import cel_loss, binary_loss, regression_loss
from paramboost.boosting import _update_one_iter, presum_binned_array
from paramboost.preprocessing import (
    process_structure,
    process_continuity,
    process_binary_features,
    process_unfeasible_splits,
    create_bounds,
    create_bounds_c,
    load_data,
    load_labels,
    preprocess_data_for_boostable_grads,
)
import matplotlib.pyplot as plt


class ParamBoost:
    def __init__(
        self,
        data: jnp.ndarray | pd.DataFrame | np.ndarray,
        num_classes: int,
        order: int = 3,
        continuity: int = -1,
        max_bins: int = 16,
        max_bins_poly: int = 16,
        structure: list[list[int]] = None,
        monotone_constraints: list[int] = None,
        curvature_constraints: list[int] = None,
        max_depth: int = 1,
        num_leaves: int = 3,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        lambda_l1: float = 0,
        lambda_l2: float = 0,
        bagging_fraction: float = 1,
        bagging_freq: int = 0,
        min_data_in_leaf: int = 20,
        min_data_in_bin: int = 3,
        min_sum_hessian_in_leaf: float = 1e-3,
        min_gain_to_split: int = 0,
        seed: int = 0,
        labels: jnp.array = jnp.array([]),
        labels_val: jnp.array = jnp.array([]),
        early_stopping_rounds: int = 10,
        data_val: jnp.array = None,
        task: str = "multiclass",
        boosting_level: int = 0,
        verbosity: int = 1,
    ):
        self.min_data, self.max_data, self.data = load_data(data)
        self.labels = load_labels(labels)

        self.num_classes = num_classes
        self.num_features = self.data.shape[1]
        self.num_observations = self.data.shape[0]
        self.order = jnp.arange(order + 1)
        self.continuity = continuity
        self.max_bins = max_bins
        self.monotone_constraints = (
            jnp.array(monotone_constraints)
            if monotone_constraints is not None
            else jnp.zeros(self.num_features, dtype=jnp.int32)
        )
        self.curvature_constraints = (
            jnp.array(curvature_constraints)
            if curvature_constraints is not None
            else jnp.zeros(self.num_features, dtype=jnp.int32)
        )  # 1 for convex, -1 for concave, 0 for no constraint
        self.structure = (
            structure
            if structure is not None
            else [list(range(self.num_features)) for _ in range(self.num_classes)]
        )
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_data_in_leaf = min_data_in_leaf
        self.min_data_in_bin = min_data_in_bin
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.min_gain_to_split = min_gain_to_split
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds

        self.feature_importance_dict = {
            "gain": jnp.zeros(
                (self.num_classes, self.num_features, self.order.shape[0]),
                dtype=jnp.float32,
            ),
            "num_splits": jnp.zeros(
                (self.num_classes, self.num_features, self.order.shape[0]),
                dtype=jnp.int32,
            ),
        }

        self.bin_mapper = BinMapper(
            max_bin=self.max_bins, min_data_in_bin=self.min_data_in_bin
        )

        self.key = jax.random.key(seed)
        self.draw_indices = draw_new_indices(
            self.key, self.num_observations, self.bagging_fraction
        )

        if self.bagging_freq > 0:
            indices_to_ignore, new_key = self.draw_indices(self.key)
            self.indices_to_ignore = indices_to_ignore
            self.key = new_key

        else:
            self.indices_to_ignore = jnp.array([], jnp.int32)

        (
            self.bin_edges,
            self.histograms,
            self.data_bin_indices,
            self.bin_edges_indices,
        ) = self.bin_mapper.build_histograms(self.data)

        self.non_constant_idx = _create_non_constant_idx(
            self.bin_edges,
            max_bins_poly,
        )

        if task == "multiclass":
            self.init_preds = _init_predictions_classification(
                jnp.arange(self.num_classes, dtype=jnp.float32),
                self.labels,
            )
            self.labels = _one_hot_encode_labels(self.labels, self.num_classes)
            self.pred_fct = partial(jax.nn.softmax, axis=1)
            self.f_obj = f_obj_cel
            self.loss_fct = cel_loss
        elif task == "binary":
            self.init_preds = _init_predictions_classification(
                jnp.array([1], dtype=jnp.float32), self.labels
            )
            self.pred_fct = jax.nn.sigmoid
            self.f_obj = f_obj_binary
            self.loss_fct = binary_loss
            self.num_classes = 1
        elif task == "regression":
            self.init_preds = _init_predictions_regression(self.labels)
            self.pred_fct = jax.nn.identity
            self.f_obj = f_obj_regression
            self.loss_fct = regression_loss
            self.num_classes = 1
        else:
            raise ValueError(f"Task {task} not supported.")

        self.early_stopping_rounds = early_stopping_rounds
        self.task = task

        if data_val is not None:
            _, _, self.data_val = load_data(data_val, self.min_data, self.max_data)
            self.labels_val = load_labels(labels_val)
            self.data_bin_indices_val = _get_bin_indices(self.data_val, self.bin_edges)
            if task == "multiclass":
                self.labels_val = _one_hot_encode_labels(
                    self.labels_val, self.num_classes
                )
        else:
            self.data_bin_indices_val = self.data_bin_indices
            self.data_val = self.data
            self.labels_val = self.labels

        self.inner_preds = self.init_preds * jnp.ones(
            (self.data.shape[0], self.num_classes)
        )
        self.inner_preds_val = self.init_preds * jnp.ones(
            (self.data_val.shape[0], self.num_classes)
        )

        self.preds = self.pred_fct(self.inner_preds)

        self.ensembles = jnp.zeros(
            (
                self.num_classes,
                self.num_features,
                self.order.shape[0],
                self.max_bins,
            ),
            dtype=jnp.float32,
        )

        indices_grid = jnp.indices(self.ensembles.shape[: boosting_level + 1])
        self.leading_indices = tuple(indices_grid[i] for i in range(boosting_level + 1))
        (
            self.lower_bound_left,
            self.upper_bound_left,
            self.lower_bound_right,
            self.upper_bound_right,
        ) = create_bounds(
            jnp.arange(self.num_classes),
            jnp.arange(self.num_features),
            self.order,
            jnp.arange(self.max_bins - 1),
            self.monotone_constraints,
        )
        (
            self.lower_bound_left,
            self.upper_bound_left,
            self.lower_bound_right,
            self.upper_bound_right,
        ) = process_structure(
            self.structure,
            self.lower_bound_left,
            self.upper_bound_left,
            self.lower_bound_right,
            self.upper_bound_right,
        )
        (
            self.lower_bound_left,
            self.upper_bound_left,
            self.lower_bound_right,
            self.upper_bound_right,
        ) = process_continuity(
            self.continuity,
            self.lower_bound_left,
            self.upper_bound_left,
            self.lower_bound_right,
            self.upper_bound_right,
        )

        binary_features = [
            i
            for i in range(self.num_features)
            if jnp.unique(self.data[:, i]).shape[0] == 2
        ]
        self.binary_features = jnp.array(
            [1 if i in binary_features else 0 for i in range(self.num_features)],
            dtype=jnp.int32,
        )
        if binary_features:
            (
                self.lower_bound_left,
                self.upper_bound_left,
                self.lower_bound_right,
                self.upper_bound_right,
            ) = process_binary_features(
                binary_features,
                self.lower_bound_left,
                self.upper_bound_left,
                self.lower_bound_right,
                self.upper_bound_right,
            )
        (
            self.lower_bound_left,
            self.upper_bound_left,
            self.lower_bound_right,
            self.upper_bound_right,
        ) = process_unfeasible_splits(
            self.histograms,
            self.min_data_in_bin,
            self.lower_bound_left,
            self.upper_bound_left,
            self.lower_bound_right,
            self.upper_bound_right,
            self.bin_edges,
            self.non_constant_idx,
        )

        (
            self.lower_bound_left_c,
            self.upper_bound_left_c,
            self.lower_bound_right_c,
            self.upper_bound_right_c,
        ) = create_bounds_c(
            self.lower_bound_left,
            self.upper_bound_left,
            self.lower_bound_right,
            self.upper_bound_right,
            self.curvature_constraints,
        )

        feasible_splits = jnp.where(
            jnp.min(self.lower_bound_right[:, :, 1:, :], axis=0) < jnp.inf, 1, 0
        )
        # last indices is for the flat array
        self.boostable_indices = (
            *jnp.nonzero(feasible_splits),
            jnp.flatnonzero(feasible_splits),
        )

        # adjust bounds for order sign. This is because
        # the leafs are multiplied by (x-s)^order, which
        # changes sign for even/odd order on the left of the
        # split point.
        order_mask = 2 * (self.order % 2) - 1  # 1 if odd order, -1 if even order
        self.lower_bound_left = self.lower_bound_left * order_mask[None, None, :, None]
        self.upper_bound_left = self.upper_bound_left * order_mask[None, None, :, None]
        order_mask_c = -order_mask  # 1 if even order, -1 if odd order
        # sign is reversed for curvature constraints since it is second derivative
        # as opposed to first derivative for monotonicity constraints
        self.lower_bound_left_c = (
            self.lower_bound_left_c * order_mask_c[None, None, :, None]
        )
        self.upper_bound_left_c = (
            self.upper_bound_left_c * order_mask_c[None, None, :, None]
        )

        arange = jnp.arange(self.max_bins)
        edgerange = jnp.arange(self.max_bins - 1) + 1
        self.mask = arange[None, :] < edgerange[:, None]
        self.verbosity = verbosity
        self.best_iteration = 0
        if self.verbosity > 0:
            init_cel = self.loss_fct(self.preds, self.labels)
            val_cel = self.loss_fct(
                self.pred_fct(self.inner_preds_val), self.labels_val
            )
            print_cel(init_cel, val_cel, 0, self.num_iterations, self.verbosity)

    def _inner_predict(self, utilities: bool = False):
        if utilities:
            return self.inner_preds
        return self.preds

    def predict(
        self, data: np.ndarray | pd.DataFrame | jnp.ndarray, utilities: bool = False
    ):
        """
        Make predictions using the trained model.

        Parameters
        ----------
        data : np.ndarray | pd.DataFrame | jnp.ndarray
            Feature values for which to make predictions.
        utilities : bool, optional
            If True, return raw predictions without applying the link function.
            Default is False.

        Returns
        -------
        jnp.ndarray
            Predictions for the input data.
        """
        if not hasattr(self, "best_ensembles"):
            raise ValueError("Model has not been trained yet.")
        _, _, data = load_data(data, self.min_data, self.max_data)
        raw_preds = (
            predict_vectorised(
                data,
                self.best_ensembles,
                self.bin_edges,
                self.order,
            )
            + self.init_preds
        )

        if utilities:
            return raw_preds

        return self.pred_fct(raw_preds)

    def standard_error(
        self,
    ) -> np.ndarray:
        """
        Compute the approximated standard error of the coefficients.

        Returns
        -------
        np.ndarray
            Standard error of the linear coefficients.
        """
        preds = self.predict(self.data)
        data = self.data
        indices = self.data_bin_indices
        _, hess = self.f_obj(preds, self.labels, self.num_classes)

        se = _compute_standard_error(
            hess.T,
            indices,
            self.order,
            data,
            self.bin_edges,
            self.continuity,
            self.lambda_l2,
        )
        return se

    def predict_with_ci(
        self,
        data: np.ndarray | pd.DataFrame | jnp.ndarray,
        alpha: float = 0.05,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Predict with confidence intervals.

        Parameters
        ----------
        data : np.ndarray
            Feature values for which to make predictions.
        alpha : float, optional
            Significance level for the confidence interval.
            Default is 0.05 for a 95% confidence interval.

        Returns
        -------
        tuple of jax.array
            Tuple containing:
            - raw_preds: Raw predictions without applying the link function.
            - lower_raw_preds: Lower bound of the confidence interval.
            - upper_raw_preds: Upper bound of the confidence interval.
        """
        if not hasattr(self, "best_ensembles"):
            raise ValueError("Model has not been trained yet.")

        _, _, data = load_data(data, self.min_data, self.max_data)

        se = self.standard_error()

        # combines standard errors over features and orders
        raw_preds, lower_raw_preds, upper_raw_preds = predict_with_se(
            data,
            se,
            self.best_ensembles,
            self.bin_edges,
            self.order,
            alpha,
        )

        return raw_preds, lower_raw_preds, upper_raw_preds

    def plot_shape_functions(self):
        """
        Plot the shape functions for each class and feature.
        """
        if not hasattr(self, "best_ensembles"):
            raise ValueError("Model has not been trained yet.")
        plot_data = jnp.zeros_like(self.data)
        x_plot = jnp.linspace(-0.1, 1.1, self.data.shape[0])

        for class_idx in range(self.num_classes):
            for feature_idx in range(self.num_features):
                plot_data = jnp.zeros_like(self.data)
                plot_data = plot_data.at[:, feature_idx].set(x_plot)
                y_plot = predict_vectorised(
                    plot_data,
                    self.best_ensembles,
                    self.bin_edges,
                    self.order,
                )
                +self.init_preds

                plt.figure(figsize=(6, 4))
                plt.plot(x_plot, y_plot[:, class_idx], color="k")
                plt.legend()
                plt.title(
                    f"Shape Function for Class {class_idx}, Feature {feature_idx}"
                )

                plt.tight_layout()
                plt.show()


def train(model: ParamBoost) -> ParamBoost:
    """
    Train the ParamBoost model.

    Parameters
    ----------
    model : ParamBoost
        An instance of the ParamBoost class.

    Returns
    -------
    model : ParamBoost
        The trained ParamBoost model.
    """
    (
        _,
        model.inner_preds,
        model.inner_preds_val,
        model.preds,
        model.feature_importance_dict,
        _,
        model.indices_to_ignore,
        model.key,
        model.lower_bound_left,
        model.upper_bound_left,
        model.lower_bound_right,
        model.upper_bound_right,
        model.lower_bound_left_c,
        model.upper_bound_left_c,
        model.lower_bound_right_c,
        model.upper_bound_right_c,
        model.best_ensembles,
        model.best_score_train,
        model.best_score_val,
        model.best_feature_importance_dict,
        model.best_iteration,
    ) = _train(
        model.data_bin_indices,
        model.data,
        model.bin_edges,
        model.bin_edges_indices,
        model.ensembles,
        model.indices_to_ignore,
        model.num_leaves,
        model.max_depth,
        model.lambda_l1,
        model.lambda_l2,
        model.monotone_constraints,
        model.curvature_constraints,
        model.learning_rate,
        model.min_sum_hessian_in_leaf,
        model.min_data_in_leaf,
        model.min_gain_to_split,
        model.feature_importance_dict,
        model.inner_preds,
        model.inner_preds_val,
        model.preds,
        model.labels,
        model.num_classes,
        model.num_observations,
        model.bagging_fraction,
        model.bagging_freq,
        model.early_stopping_rounds,
        model.num_iterations,
        model.key,
        model.f_obj,
        model.pred_fct,
        model.loss_fct,
        model.labels_val,
        model.data_bin_indices_val,
        model.data_val,
        model.leading_indices,
        model.order,
        model.continuity,
        model.lower_bound_left,
        model.upper_bound_left,
        model.lower_bound_right,
        model.upper_bound_right,
        model.lower_bound_left_c,
        model.upper_bound_left_c,
        model.lower_bound_right_c,
        model.upper_bound_right_c,
        model.boostable_indices,
        model.binary_features,
        model.mask,
        model.verbosity,
    )

    return model


@partial(
    jit,
    static_argnames=[
        "num_leaves",
        "max_depth",
        "lambda_l1",
        "lambda_l2",
        "learning_rate",
        "min_sum_hessian_in_leaf",
        "min_data_in_leaf",
        "min_gain_to_split",
        "bagging_freq",
        "early_stopping_rounds",
        # "key",
        "num_classes",
        "num_iterations",
        "num_observation",
        "bagging_fraction",
        "f_obj",
        "loss_fct",
        "pred_fct",
        "continuity",
        "verbosity",
    ],
)
def _train(
    data_bin_indices,
    data,
    bin_edges,
    bin_indices,
    ensembles,
    indices_to_ignore,
    num_leaves,
    max_depth,
    lambda_l1,
    lambda_l2,
    monotone_constraints,
    curvature_constraints,
    learning_rate,
    min_sum_hessian_in_leaf,
    min_data_in_leaf,
    min_gain_to_split,
    feature_importance_dict,
    inner_preds,
    inner_preds_val,
    preds,
    labels,
    num_classes,
    num_observation,
    bagging_fraction,
    bagging_freq,
    early_stopping_rounds,
    num_iterations,
    key,
    f_obj,
    pred_fct,
    loss_fct,
    labels_val,
    data_bin_indices_val,
    data_val,
    leading_indices,
    order,
    continuity,
    lower_bound_left,
    upper_bound_left,
    lower_bound_right,
    upper_bound_right,
    lower_bound_left_c,
    upper_bound_left_c,
    lower_bound_right_c,
    upper_bound_right_c,
    boostable_indices,
    binary_features,
    mask,
    verbosity,
):
    data_for_grads, data_for_grads_left = preprocess_data_for_boostable_grads(
        data.astype(jnp.float16),
        bin_edges.astype(jnp.float16),
        boostable_indices,
    )
    data_for_hess = jnp.copy(data_for_grads) ** 2
    data_for_hess_left = jnp.copy(data_for_grads_left) ** 2
    data_val_precomputed, _ = preprocess_data_for_boostable_grads(
        data_val.astype(jnp.float16),
        bin_edges.astype(jnp.float16),
        boostable_indices,
    )

    hist_presummed = presum_binned_array(  # n_potential_splits x n_features
        data_bin_indices, jnp.ones(data.shape[0]), bin_edges.shape[0]
    )
    sum_hist = data_for_grads.shape[0]

    # initialising best values
    best_ensembles = ensembles
    best_loss_val = jnp.inf
    best_loss_train = jnp.inf
    best_feature_importance_dict = feature_importance_dict
    best_iter = 0

    init_value = (
        ensembles,
        inner_preds,
        inner_preds_val,
        preds,
        feature_importance_dict,
        0,
        indices_to_ignore,
        key,
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
        lower_bound_left_c,
        upper_bound_left_c,
        lower_bound_right_c,
        upper_bound_right_c,
        best_ensembles,
        best_loss_train,
        best_loss_val,
        best_feature_importance_dict,
        best_iter,
    )

    def _early_stopping_cond(carry: tuple) -> bool:
        (
            _,
            _,
            _,
            _,
            _,
            n_boosting_round,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            best_iter,
        ) = carry

        return jnp.where(
            n_boosting_round < num_iterations,
            jnp.where(
                n_boosting_round - best_iter >= early_stopping_rounds, False, True
            ),
            False,
        )

    def _boost_one_iter(carry: tuple) -> tuple[tuple, None]:
        (
            ensembles,
            inner_preds,
            inner_preds_val,
            preds,
            feature_importance_dict,
            n_boosting_round,
            indices_to_ignore,
            key,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            lower_bound_left_c,
            upper_bound_left_c,
            lower_bound_right_c,
            upper_bound_right_c,
            best_ensembles,
            best_loss_train,
            best_loss_val,
            best_feature_importance_dict,
            best_iter,
        ) = carry

        grad, hess = f_obj(preds, labels, num_classes)

        (
            ensembles,
            feature_importance_dict,
            inner_preds,
            inner_preds_val,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            lower_bound_left_c,
            upper_bound_left_c,
            lower_bound_right_c,
            upper_bound_right_c,
        ) = _update_one_iter(
            grad,
            hess,
            data_bin_indices,
            data_bin_indices_val,
            data_for_grads,
            data_for_hess,
            data_for_grads_left,
            data_for_hess_left,
            data_val_precomputed,
            data,
            data_val,
            hist_presummed,
            sum_hist,
            boostable_indices,
            bin_edges,
            bin_indices,
            ensembles,
            feature_importance_dict,
            inner_preds,
            inner_preds_val,
            lambda_l1,
            lambda_l2,
            monotone_constraints,
            curvature_constraints,
            learning_rate,
            min_sum_hessian_in_leaf,
            min_data_in_leaf,
            min_gain_to_split,
            leading_indices,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            lower_bound_left_c,
            upper_bound_left_c,
            lower_bound_right_c,
            upper_bound_right_c,
            order,
            continuity,
            binary_features,
            mask,
        )

        n_boosting_round += 1

        preds = pred_fct(inner_preds)

        loss_train = loss_fct(preds, labels)

        preds_val = pred_fct(inner_preds_val)

        loss_val = loss_fct(preds_val, labels_val)

        (
            best_loss_val,
            best_loss_train,
            best_ensembles,
            best_feature_importance_dict,
            best_iter,
        ) = jax.lax.cond(
            loss_val < best_loss_val,
            lambda: (
                loss_val,
                loss_train,
                ensembles,
                feature_importance_dict,
                n_boosting_round,
            ),
            lambda: (
                best_loss_val,
                best_loss_train,
                best_ensembles,
                best_feature_importance_dict,
                best_iter,
            ),
        )

        indices_to_ignore, new_key = jax.lax.cond(
            bagging_freq > 0 and n_boosting_round % bagging_freq == 0,
            lambda x: draw_new_indices(x, num_observation, bagging_fraction),
            lambda x: (indices_to_ignore, x),
            key,
        )

        _ = jax.lax.cond(
            verbosity > 0,
            lambda: print_cel(
                loss_train, loss_val, n_boosting_round, num_iterations, verbosity
            ),
            lambda: jnp.int32(0),
        )

        return (
            ensembles,
            inner_preds,
            inner_preds_val,
            preds,
            feature_importance_dict,
            n_boosting_round,
            indices_to_ignore,
            new_key,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            lower_bound_left_c,
            upper_bound_left_c,
            lower_bound_right_c,
            upper_bound_right_c,
            best_ensembles,
            best_loss_train,
            best_loss_val,
            best_feature_importance_dict,
            best_iter,
        )

    return jax.lax.while_loop(
        _early_stopping_cond,
        _boost_one_iter,
        init_val=init_value,
    )
