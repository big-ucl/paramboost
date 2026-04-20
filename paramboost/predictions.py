from functools import partial
import jax.numpy as jnp
from jax import vmap, jit
import jax


@jit
@partial(vmap, in_axes=(0, None), out_axes=1)
def _init_predictions_classification(
    class_index: int, labels: jnp.ndarray
) -> jnp.ndarray:
    """
    Initialise predictions for classification tasks.

    Parameters
    ----------
    class_index : int
        The index of the class for which to initialise predictions.
    labels : jnp.ndarray
        True labels (shape: [n_samples]).

    Returns
    -------
    inner_predictions : jnp.ndarray
        Initial predictions for the specified class (shape: [n_samples]).
    """
    inner_predictions = jnp.ones((1), dtype=jnp.float32) * jnp.log(
        jnp.mean(labels == class_index)
    )
    return inner_predictions


@jit
def _init_predictions_regression(labels: jnp.ndarray) -> jnp.ndarray:
    """
    Initialise predictions for regression tasks.

    Parameters
    ----------
    labels : jnp.ndarray
        True labels (shape: [n_samples]).

    Returns
    -------
    inner_predictions : jnp.ndarray
        Initial predictions for regression (shape: [n_samples]).
    """
    mean_label = jnp.mean(labels)
    inner_predictions = jnp.ones((1), dtype=jnp.float32) * mean_label
    return inner_predictions.reshape(1, -1)


@jit
@partial(vmap, in_axes=(None, 0, None, None), out_axes=1)
def predict_vectorised(
    data: jnp.ndarray, ensemble: jnp.ndarray, bin_edges: jnp.ndarray, order: jnp.ndarray
) -> jnp.ndarray:
    """
    Make predictions using the ensemble of models.

    Parameters
    ----------
    data : jnp.ndarray
        Input data (shape: [n_samples]).
    ensemble : jnp.ndarray
        Ensemble of model parameters (shape: [n_bins, n_models, n_orders]).
    bin_edges : jnp.ndarray
        Edges of the bins (shape: [n_bins]).
    order : jnp.ndarray
        Orders of the models (shape: [n_models]).

    Returns
    -------
    predictions : jnp.ndarray
        Predictions for the input data (shape: [n_samples]).
    """
    data_bin_indices = _get_bin_indices(data, bin_edges)

    predictions = _get_predictions(data, data_bin_indices, ensemble, order)

    return predictions.sum(axis=2).sum(axis=1)


@jit
@partial(vmap, in_axes=(1, 1), out_axes=1)
def _get_bin_indices(data: jnp.ndarray, bin_edges: jnp.ndarray) -> jnp.ndarray:
    """
    Get bin indices for the input data based on the bin edges.

    Parameters
    ----------
    data : jnp.ndarray
        Input data (shape: [n_samples]).
    bin_edges : jnp.ndarray
        Edges of the bins (shape: [n_bins]).

    Returns
    -------
    data_bin_indices : jnp.ndarray
        Indices of the bins for each data point (shape: [n_samples]).
    """
    data_bin_indices = jnp.digitize(data, bins=bin_edges, right=True)
    return data_bin_indices


@partial(vmap, in_axes=(1, 1, 0, None), out_axes=1)
@partial(vmap, in_axes=(None, None, 0, 0), out_axes=1)
def _get_predictions(
    data: jnp.ndarray,
    data_bin_indices: jnp.ndarray,
    ensemble: jnp.ndarray,
    order: jnp.ndarray,
) -> jnp.ndarray:
    """
    Get predictions for the input data using the ensemble and bin indices.

    Parameters
    ----------
    data : jnp.ndarray
        Input data (shape: [n_samples]).
    data_bin_indices : jnp.ndarray
        Indices of the bins for each data point (shape: [n_samples]).
    ensemble : jnp.ndarray
        Ensemble of model parameters (shape: [n_classes, n_features, n_order, n_bins]).
    order : jnp.ndarray
        Orders of the models (shape: [n_order]).
    Returns
    -------
    predictions : jnp.ndarray
        Predictions for the input data (shape: [n_samples]).
    """
    predictions = ensemble[data_bin_indices] * data**order
    return predictions


@partial(jit, static_argnames=("alpha",))
@partial(vmap, in_axes=(None, 0, 0, None, None, None), out_axes=1)
def predict_with_se(
    data: jnp.ndarray,
    se: jnp.ndarray,
    ensemble: jnp.ndarray,
    bin_edges: jnp.ndarray,
    order: jnp.ndarray,
    alpha: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Make predictions with standard error and confidence intervals.

    Parameters
    ----------
    data : jnp.ndarray
        Input data (shape: [n_samples, n_features]).
    se : jnp.ndarray
        Standard errors for the ensemble (shape: [n_classes, n_features, n_order, n_bins]).
    ensemble : jnp.ndarray
        Ensemble of model parameters (shape: [n_classes, n_features, n_order, n_bins]).
    bin_edges : jnp.ndarray
        Edges of the bins (shape: [n_bins, n_features]).
    order : jnp.ndarray
        Orders of the models (shape: [n_order]).
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    raw_preds : jnp.ndarray
        Raw predictions (shape: [n_samples]).
    lower_ci : jnp.ndarray
        Lower bound of the confidence interval (shape: [n_samples]).
    upper_ci : jnp.ndarray
        Upper bound of the confidence interval (shape: [n_samples]).
    """
    data_bin_indices = _get_bin_indices(data, bin_edges)

    predictions = _get_predictions(data, data_bin_indices, ensemble, order)

    raw_preds = predictions.sum(axis=2).sum(axis=1)

    se2_x = square_and_mult_se(se, data, data_bin_indices, order, bin_edges)
    combined_se = jnp.sqrt(se2_x.sum(axis=2).sum(axis=1))

    z_score = jax.scipy.stats.norm.ppf(1 - alpha / 2)

    return (
        raw_preds,
        raw_preds - z_score * combined_se,
        raw_preds + z_score * combined_se,
    )


@partial(vmap, in_axes=(0, 1, 1, None, 1), out_axes=1)
@partial(vmap, in_axes=(0, None, None, 0, None), out_axes=1)
def square_and_mult_se(
    se: jnp.ndarray,
    data: jnp.ndarray,
    data_bin_indices: jnp.ndarray,
    order: jnp.ndarray,
    bin_edges: jnp.ndarray,
) -> jnp.ndarray:
    """
    Square and multiply standard errors by the appropriate power of data.

    Parameters
    ----------
    se : jnp.ndarray
        Standard errors for the ensemble (shape: [n_classes, n_features, n_order, n_bins]).
    data : jnp.ndarray
        Input data (shape: [n_samples, n_features]).
    data_bin_indices : jnp.ndarray
        Indices of the bins for each data point (shape: [n_samples, n_features]).
    order : jnp.ndarray
        Orders of the models (shape: [n_order]).
    bin_edges : jnp.ndarray
        Edges of the bins (shape: [n_bins, n_features]).

    Returns
    -------
    se2_x : jnp.ndarray
        Squared and multiplied standard errors (shape: [n_samples]).
    """
    bin_edges = jnp.where(bin_edges < 1, bin_edges, data.max())
    order_2 = order * 2
    se2_x = se[data_bin_indices] ** 2 * (
        (data - bin_edges[data_bin_indices]) ** order_2
    )  # need to shift the data by the next knot value - not exactly sure why
    return se2_x
