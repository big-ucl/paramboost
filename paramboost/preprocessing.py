import numpy as np
import pandas as pd
import jax.numpy as jnp
from functools import partial
from jax import vmap


def process_structure(
    structure: list[list[int]],
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process the structure parameter to ensure it is in the correct format.

    Parameters
    ----------
    structure : list of list of int
        List of lists containing indices of features in the respective class.
    lower_bound_left : jnp.ndarray
        Lower bound left array. (shape: [n_classes, n_features, n_order, n_bins])
    upper_bound_left : jnp.ndarray
        Upper bound left array. (shape: [n_classes, n_features, n_order, n_bins])
    lower_bound_right : jnp.ndarray
        Lower bound right array. (shape: [n_classes, n_features, n_order, n_bins])
    upper_bound_right : jnp.ndarray
        Upper bound right array. (shape: [n_classes, n_features, n_order, n_bins])

    Returns
    -------
    tuple
        Processed lower and upper bounds for left and right.
    """
    full_features_indices = np.arange(upper_bound_right.shape[1])
    for i, struct in enumerate(structure):
        indices_missing = np.setdiff1d(full_features_indices, struct)
        for j in indices_missing:
            lower_bound_left = lower_bound_left.at[i, j].set(jnp.inf)
            upper_bound_left = upper_bound_left.at[i, j].set(-jnp.inf)
            lower_bound_right = lower_bound_right.at[i, j].set(jnp.inf)
            upper_bound_right = upper_bound_right.at[i, j].set(-jnp.inf)

    return lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right


def process_continuity(
    continuity: int,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process the continuity parameter to set bounds accordingly.

    Parameters
    ----------
    continuity : int
        Continuity level (-1, 0, 1, or 2).
    lower_bound_left : jnp.ndarray
        Lower bound left array. (shape: [n_classes, n_features, n_order, n_bins])
    upper_bound_left : jnp.ndarray
        Upper bound left array. (shape: [n_classes, n_features, n_order, n_bins])
    lower_bound_right : jnp.ndarray
        Lower bound right array. (shape: [n_classes, n_features, n_order, n_bins])
    upper_bound_right : jnp.ndarray
        Upper bound right array. (shape: [n_classes, n_features, n_order, n_bins])

    Returns
    -------
    tuple
        Processed lower and upper bounds for left and right.
    """
    if continuity < 0:
        return lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right

    lower_bound_left = lower_bound_left.at[:, :, : continuity + 1, 1:].set(jnp.inf)
    upper_bound_left = upper_bound_left.at[:, :, : continuity + 1, 1:].set(-jnp.inf)
    lower_bound_right = lower_bound_right.at[:, :, : continuity + 1, 1:].set(jnp.inf)
    upper_bound_right = upper_bound_right.at[:, :, : continuity + 1, 1:].set(-jnp.inf)

    return lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right


def process_binary_features(
    binary_features: list[int],
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process the categorical features list into a boolean mask.

    Parameters
    ----------
    categorical_features : list of int
        List of indices of categorical features.
    lower_bound_left : jnp.ndarray
        Lower bound left array. (shape: [n_classes, n_features, n_order, n_bins])
    upper_bound_left : jnp.ndarray
        Upper bound left array. (shape: [n_classes, n_features, n_order, n_bins])
    lower_bound_right : jnp.ndarray
        Lower bound right array. (shape: [n_classes, n_features, n_order, n_bins])
    upper_bound_right : jnp.ndarray
        Upper bound right array. (shape: [n_classes, n_features, n_order, n_bins])

    Returns
    -------
    jnp.ndarray
        Boolean mask indicating which features are binary.
    """
    for feature in binary_features:
        lower_bound_left = lower_bound_left.at[:, feature, 1:, :].set(jnp.inf)
        upper_bound_left = upper_bound_left.at[:, feature, 1:, :].set(-jnp.inf)
        lower_bound_right = lower_bound_right.at[:, feature, 1:, :].set(jnp.inf)
        upper_bound_right = upper_bound_right.at[:, feature, 1:, :].set(-jnp.inf)
    return lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right


def process_unfeasible_splits(
    histograms: jnp.ndarray,
    min_data_in_bin: int,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    non_constant_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process features to ensure minimum data in bins.

    Parameters
    ----------
    histograms : jnp.ndarray
        Histograms of the features. (shape: [n_bins, n_features])
    min_data_in_bin : int
        Minimum number of data points required in each bin.
    lower_bound_left : jnp.ndarray
        Lower bound left array. (shape: [n_classes, n_features, n_order, n_bins])
    upper_bound_left : jnp.ndarray
        Upper bound left array. (shape: [n_classes, n_features, n_order, n_bins])
    lower_bound_right : jnp.ndarray
        Lower bound right array. (shape: [n_classes, n_features, n_order, n_bins])
    upper_bound_right : jnp.ndarray
        Upper bound right array. (shape: [n_classes, n_features, n_order, n_bins])
    bin_edges : jnp.ndarray
        Edges of the bins for each feature. (shape: [n_bins, n_features])
    non_constant_idx : jnp.ndarray
        Indices of non-constant features for each class. (shape: [n_bins_poly, n_features])

    Returns
    -------
    tuple
        Processed lower and upper bounds for left and right.
    """
    # no potential split points
    mask_insufficient_data = (histograms < min_data_in_bin).sum(axis=0) - (
        jnp.isinf(bin_edges).sum(axis=0)
    ) > 0
    lower_bound_left = lower_bound_left.at[:, mask_insufficient_data, :, :].set(jnp.inf)
    upper_bound_left = upper_bound_left.at[:, mask_insufficient_data, :, :].set(
        -jnp.inf
    )
    lower_bound_right = lower_bound_right.at[:, mask_insufficient_data, :, :].set(
        jnp.inf
    )
    upper_bound_right = upper_bound_right.at[:, mask_insufficient_data, :, :].set(
        -jnp.inf
    )

    # not a split point
    mask_no_split_point = (bin_edges >= 1).swapaxes(0, 1)
    lower_bound_left = (
        lower_bound_left.swapaxes(1, 2)
        .at[:, :, mask_no_split_point]
        .set(jnp.inf)
        .swapaxes(2, 1)
    )
    upper_bound_left = (
        upper_bound_left.swapaxes(1, 2)
        .at[:, :, mask_no_split_point]
        .set(-jnp.inf)
        .swapaxes(2, 1)
    )
    lower_bound_right = (
        lower_bound_right.swapaxes(1, 2)
        .at[:, :, mask_no_split_point]
        .set(jnp.inf)
        .swapaxes(2, 1)
    )
    upper_bound_right = (
        upper_bound_right.swapaxes(1, 2)
        .at[:, :, mask_no_split_point]
        .set(-jnp.inf)
        .swapaxes(2, 1)
    )

    indices_to_mask = jnp.ones((lower_bound_left.shape[1], lower_bound_left.shape[-1]))
    for feature_idx in range(lower_bound_left.shape[1]):
        for nc_idx in non_constant_idx[:, feature_idx]:
            indices_to_mask = indices_to_mask.at[feature_idx, nc_idx].set(0)

    mask = jnp.where(indices_to_mask == 1, True, False)
    lower_bound_left = (
        lower_bound_left.swapaxes(1, 2).at[:, 1:, mask].set(jnp.inf).swapaxes(2, 1)
    )
    upper_bound_left = (
        upper_bound_left.swapaxes(1, 2).at[:, 1:, mask].set(-jnp.inf).swapaxes(2, 1)
    )
    lower_bound_right = (
        lower_bound_right.swapaxes(1, 2).at[:, 1:, mask].set(jnp.inf).swapaxes(2, 1)
    )
    upper_bound_right = (
        upper_bound_right.swapaxes(1, 2).at[:, 1:, mask].set(-jnp.inf).swapaxes(2, 1)
    )

    return (
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
    )


def load_data(
    data: jnp.ndarray | pd.DataFrame | np.ndarray,
    min_data: jnp.ndarray = None,
    max_data: jnp.ndarray = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Load data from various formats and encode min/max values.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input samples.
    min_data : jnp.ndarray, optional
        Minimum values for each feature. If None, computed from data.
    max_data : jnp.ndarray, optional
        Maximum values for each feature. If None, computed from data.
    """

    if isinstance(data, pd.DataFrame):
        data = jnp.array(data.values)
    elif isinstance(data, np.ndarray):
        data = jnp.array(data)
    elif not isinstance(data, jnp.ndarray):
        raise ValueError(
            "Data must be a pandas DataFrame, numpy ndarray, or jax jnp.ndarray."
        )
    min_data, max_data, data = encode_min_max(data, min_data, max_data)
    return min_data, max_data, data


def load_labels(
    labels: jnp.ndarray | pd.Series | pd.DataFrame | np.ndarray,
) -> jnp.ndarray:
    """
    Load labels from various formats.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        The target values.

    Returns
    -------
    jnp.ndarray
        Labels as a jax jnp.ndarray.
    """
    if isinstance(labels, pd.Series) or isinstance(labels, pd.DataFrame):
        labels = jnp.array(labels.values).flatten()
    elif isinstance(labels, np.ndarray):
        labels = jnp.array(labels).flatten()
    elif not isinstance(labels, jnp.ndarray):
        raise ValueError(
            "Labels must be a pandas Series/DataFrame, numpy ndarray, or jax jnp.ndarray."
        )
    return labels


def encode_min_max(
    data: jnp.ndarray, min_values: jnp.ndarray = None, max_values: jnp.ndarray = None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Encode the min and max values for each feature.

    Parameters
    ----------
    data : jnp.ndarray of shape (n_samples, n_features)
        2D array where each column represents a feature.
    min_values : jnp.ndarray of shape (n_features,), optional
        Minimum values for each feature. If None, computed from data.
    max_values : jnp.ndarray of shape (n_features,), optional
        Maximum values for each feature. If None, computed from data.

    Returns
    -------
    tuple
        min_values : jnp.ndarray
            Minimum values for each feature.
        max_values : jnp.ndarray
            Maximum values for each feature.
        data_scaled : jnp.ndarray
            data scaled to [0, 1] range for each feature.
    """
    if min_values is None:
        min_values = jnp.min(data, axis=0)
    if max_values is None:
        max_values = jnp.max(data, axis=0)
    data_scaled = (data - min_values) / (max_values - min_values)
    return min_values, max_values, data_scaled


def decode_min_max(
    data_scaled: jnp.ndarray, min_values: jnp.ndarray, max_values: jnp.ndarray
) -> jnp.ndarray:
    """
    Decode the scaled data back to original values.

    Parameters
    ----------
    data_scaled : jnp.ndarray of shape (n_samples, n_features)
        Scaled data in [0, 1] range.
    min_values : jnp.ndarray of shape (n_features,)
        Minimum values for each feature.
    max_values : jnp.ndarray of shape (n_features,)
        Maximum values for each feature.

    Returns
    -------
    jnp.ndarray
        Decoded data in original scale.
    """
    data = data_scaled * (max_values - min_values) + min_values
    return data


@partial(vmap, in_axes=(0, None, None, None, None), out_axes=0)
@partial(vmap, in_axes=(None, 0, None, None, 0), out_axes=0)
@partial(vmap, in_axes=(None, None, 0, None, None), out_axes=0)
@partial(vmap, in_axes=(None, None, None, 0, None), out_axes=0)
def create_bounds(
    classes: jnp.ndarray,
    features: jnp.ndarray,
    order: jnp.ndarray,
    bins: int,
    monotonicity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create initial bounds for monotonicity constraints.

    Parameters
    ----------
    classes : jnp.ndarray of shape (n_classes,)
        Class vector.
    features : jnp.ndarray of shape (n_features,)
        Feature matrix.
    order : jnp.ndarray of shape (n_order,)
        Order of features.
    bins : int
        Number of bins.
    monotonicity : jnp.ndarray of shape (n_features,)
        Monotonicity constraints for each feature.

    Returns
    -------
    bounds : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right
    """
    lower_bound_left = jnp.where(
        order == 0,
        -jnp.inf,
        jnp.where(
            monotonicity == 1,
            0,
            -jnp.inf,
        ),
    )
    upper_bound_left = jnp.where(
        order == 0,
        jnp.inf,
        jnp.where(
            monotonicity == -1,
            0,
            jnp.inf,
        ),
    )
    lower_bound_right = jnp.where(
        order == 0,
        -jnp.inf,
        jnp.where(
            monotonicity == 1,
            0,
            -jnp.inf,
        ),
    )
    upper_bound_right = jnp.where(
        order == 0,
        jnp.inf,
        jnp.where(
            monotonicity == -1,
            0,
            jnp.inf,
        ),
    )
    return lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right


@partial(vmap, in_axes=(0, 0, 0, 0, None), out_axes=0)
@partial(vmap, in_axes=(0, 0, 0, 0, 0), out_axes=0)
@partial(vmap, in_axes=(0, 0, 0, 0, None), out_axes=0)
@partial(vmap, in_axes=(0, 0, 0, 0, None), out_axes=0)
def create_bounds_c(
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    curvature: jnp.ndarray,
):
    """
    Create initial bounds for curvature constraints.

    Parameters
    ----------
    lower_bound_left : jnp.ndarray
        Lower bound left array.
    upper_bound_left : jnp.ndarray
        Upper bound left array.
    lower_bound_right : jnp.ndarray
        Lower bound right array.
    upper_bound_right : jnp.ndarray
        Upper bound right array.
    curvature : jnp.ndarray
        Curvature constraints for each feature.
        Order of features.

    Returns
    -------
    bounds for curvature : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
        lower_bound_left_c, upper_bound_left_c, lower_bound_right_c, upper_bound_right_c
    """
    lower_bound_left_c = jnp.where(
        lower_bound_left == jnp.inf,
        jnp.inf,
        jnp.where(
            curvature == 1,
            0,
            -jnp.inf,
        ),
    )
    upper_bound_left_c = jnp.where(
        upper_bound_left == -jnp.inf,
        -jnp.inf,
        jnp.where(
            curvature == -1,
            0,
            jnp.inf,
        ),
    )
    lower_bound_right_c = jnp.where(
        lower_bound_right == jnp.inf,
        jnp.inf,
        jnp.where(
            curvature == 1,
            0,
            -jnp.inf,
        ),
    )
    upper_bound_right_c = jnp.where(
        upper_bound_right == -jnp.inf,
        -jnp.inf,
        jnp.where(
            curvature == -1,
            0,
            jnp.inf,
        ),
    )
    return (
        lower_bound_left_c,
        upper_bound_left_c,
        lower_bound_right_c,
        upper_bound_right_c,
    )


@partial(vmap, in_axes=(None, None, 0), out_axes=1)  # boostable indices axis
def preprocess_data_for_boostable_grads(
    data: jnp.ndarray,
    bin_edges: jnp.ndarray,
    boostable_indices: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    """
    Preprocess data for boostable gradient and hessian computations.

    Parameters
    ----------
    data : jnp.ndarray
        Input data (shape: [n_samples, n_features]).
    bin_edges : jnp.ndarray
        Edges of the bins (shape: [n_bins, n_features]).
    boostable_indices : tuple[jnp.ndarray, ...]
        Boostable indices containing feature, order, and split point.

    Returns
    -------
    jnp.ndarray
        Processed data for boostable gradients (shape: [n_samples, n_boostables]).
    """
    feature, order, split_point = boostable_indices[:-1]
    data_processed = (data[:, feature] - bin_edges[split_point, feature]) ** (order + 1)
    data_processed_left = jnp.where(
        data[:, feature] <= bin_edges[split_point, feature], data_processed, 0
    )
    return data_processed, data_processed_left
