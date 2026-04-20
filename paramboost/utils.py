from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap


@partial(jit, static_argnames=["num_observation", "bagging_fraction"])
def draw_new_indices(
    key: jax.random.PRNGKey, num_observation: int, bagging_fraction: float
):
    """
    Draw new indices to ignore for bagging.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility.
    num_observation : int
        Total number of observations in the dataset.
    bagging_fraction : float
        Fraction of data to use for bagging (between 0 and 1).

    Returns
    -------
    indices_to_ignore : jnp.ndarray
        Indices of the observations to ignore.
    """

    def _draw_indices(key):
        new_key, subkey = jax.random.split(key)
        indices_to_ignore = jax.random.choice(
            subkey,
            jnp.arange(num_observation, dtype=jnp.int32),
            shape=[int((1 - bagging_fraction) * num_observation)],
            replace=False,
        )
        return indices_to_ignore, new_key

    return _draw_indices(key)


@partial(jit, static_argnames=["num_classes"])
def _one_hot_encode_labels(labels: jnp.array, num_classes: int) -> jnp.array:
    """
    One-hot encode the labels.

    Parameters
    ----------
    labels : jnp.array
        True labels (shape: [n_samples]).
    num_classes : int
        Number of classes.

    Returns
    -------
    jnp.array
        One-hot encoded labels (shape: [n_samples, n_classes]).
    """
    return jax.nn.one_hot(labels, num_classes=num_classes, dtype=jnp.int32)


def print_cel(cel_train, cel_valid, n_boosting_round, num_iterations, verbosity):
    """
    Print the loss at specified intervals.

    Parameters
    ----------
    cel_train : float
        Training categorical cross-entropy loss.
    cel_valid : float
        Validation categorical cross-entropy loss.
    n_boosting_round : int
        Current boosting round.
    num_iterations : int
        Total number of boosting iterations.
    verbosity : int
        Interval for printing the loss.
    """

    fake_value = jax.lax.cond(
        n_boosting_round % verbosity == 0,
        lambda: actually_print(cel_train, cel_valid, n_boosting_round, num_iterations),
        lambda: jnp.int32(0),
    )
    return fake_value


def actually_print(cel_train, cel_valid, n_boosting_round, num_iterations):
    """
    Actual printing function for the loss.

    Parameters
    ----------
    cel_train : float
        Training categorical cross-entropy loss.
    cel_valid : float
        Validation categorical cross-entropy loss.
    n_boosting_round : int
        Current boosting round.
    num_iterations : int
        Total number of boosting iterations.
    """

    def print():
        jax.debug.print("--- {}/{} --- ", n_boosting_round, num_iterations)
        jax.debug.print("Train loss: {:.5f}", cel_train)
        jax.debug.print("Valid loss: {:.5f}", cel_valid)
        return jnp.int32(0)

    return print()


@partial(jit, static_argnames=("continuity", "lambda_l2"))
@partial(
    vmap, in_axes=(1, None, None, None, None, None, None), out_axes=0
)  # class axis
@partial(vmap, in_axes=(None, 1, None, 1, 1, None, None), out_axes=0)  # feature axis
@partial(
    vmap, in_axes=(None, None, 0, None, None, None, None), out_axes=0
)  # order axis
def _compute_standard_error(
    hess: jnp.ndarray,
    data_bin_indices: jnp.ndarray,
    order: int,
    dataset: jnp.ndarray,
    bin_edges: jnp.ndarray,
    continuity: int,
    lambda_l2: float,
) -> jnp.ndarray:
    """
    Compute the standard error for predictions.

    Parameters
    ----------
    hess : jnp.ndarray
        Hessian values (shape: [n_samples]).
    data_bin_indices : jnp.ndarray
        Indices of the bins for each data point (shape: [n_samples]).
    order : int
        Order of the model.
    dataset : jnp.ndarray
        Input dataset (shape: [n_samples]).
    bin_edges : jnp.ndarray
        Edges of the bins (shape: [n_bins]).
    continuity : int
        Continuity parameter for standard error computation.
    lambda_l2 : float
        L2 regularization parameter.

    Returns
    -------
    std_error : jnp.ndarray
        Standard error for predictions (shape: [n_bins]).
    """
    hess_x2_sum = jax.lax.cond(
        continuity < order,
        lambda: sum_hess_in_bins(hess, data_bin_indices, order, dataset, bin_edges),
        lambda: sum_hess_one_bin(hess, order, dataset, bin_edges),
    )
    std_error = jnp.sqrt(hess_x2_sum / (hess_x2_sum + lambda_l2) ** 2)
    return std_error


@jit
def sum_hess_in_bins(
    hess: jnp.ndarray,
    data_bin_indices: jnp.ndarray,
    order: int,
    dataset: jnp.ndarray,
    bin_edges: jnp.ndarray,
) -> jnp.ndarray:
    """
    Sum the Hessian multiplied by the squared dataset values in each bin.

    Parameters
    ----------
    hess : jnp.ndarray
        Hessian values (shape: [n_samples]).
    data_bin_indices : jnp.ndarray
        Indices of the bins for each data point (shape: [n_samples]).
    order : int
        Order of the model.
    dataset : jnp.ndarray
        Input dataset (shape: [n_samples]).
    bin_edges : jnp.ndarray
        Edges of the bins (shape: [n_bins]).

    Returns
    -------
    jnp.ndarray
        Summed Hessian values for each bin (shape: [n_bins]).
    """
    hess_x2 = hess * dataset ** (2 * order)
    hess_x2_sum = jax.ops.segment_sum(
        hess_x2, data_bin_indices, num_segments=bin_edges.shape[0] + 1
    )
    return hess_x2_sum


@jit
def sum_hess_one_bin(
    hess: jnp.ndarray, order: int, dataset: jnp.ndarray, bin_edges: jnp.ndarray
) -> jnp.ndarray:
    """
    Sum the Hessian multiplied by the squared dataset values across all data points.

    Parameters
    ----------
    hess : jnp.ndarray
        Hessian values (shape: [n_samples]).
    order : int
        Order of the model.
    dataset : jnp.ndarray
        Input dataset (shape: [n_samples]).
    bin_edges : jnp.ndarray
        Edges of the bins (shape: [n_bins]).

    Returns
    -------
    jnp.ndarray
        Summed Hessian values for each bin (shape: [n_bins]).
    """
    hess_x2 = hess * dataset ** (2 * order)
    hess_x2_sum = jnp.ones(bin_edges.shape[0] + 1) * hess_x2.sum()
    return hess_x2_sum


@jit
@partial(vmap, in_axes=(1, 1), out_axes=1)
def _digitise(
    feature_values: jnp.ndarray,
    bin_edges: jnp.ndarray,
) -> jnp.ndarray:
    """
    Digitise feature values into bins based on precomputed bin edges.

    Parameters
    ----------
    feature_values : jnp.ndarray
        Feature values to be digitised.
    bin_edges : jnp.ndarray
        Precomputed edges of the bins.

    Returns
    -------
    jnp.ndarray
        Indices of the bins for each feature value.
    """
    bin_indices = jnp.digitize(feature_values, bins=bin_edges, right=True)
    return bin_indices


@partial(
    vmap,
    in_axes=(
        1,
        None,
    ),
    out_axes=1,
)
def _create_non_constant_idx(
    bin_edges_indices: jnp.ndarray,
    num_indices: int = 16,
) -> jnp.ndarray:
    """
    Create an array indicating which features are non-constant based on their bin edges.

    Parameters
    ----------
    bin_edges : jnp.ndarray
        Precomputed edges of the bins for each feature.
    num_indices : int, optional
        Number of indices to include for non constant terms. Default is 16.

    Returns
    -------
    jnp.ndarray
        Boolean array indicating non-constant features.
    """
    num_bins = bin_edges_indices.shape[0] - jnp.isposinf(bin_edges_indices).sum(axis=0)
    non_constant_idx = jnp.unique(
        jnp.round(jnp.linspace(0, num_bins, num_indices)),
        size=num_indices,
        fill_value=num_bins,
    ).astype(jnp.int32)
    return non_constant_idx
