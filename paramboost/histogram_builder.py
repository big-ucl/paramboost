import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


class BinMapper:
    """Maps feature values to discrete bins."""

    def __init__(self, max_bin: int, min_data_in_bin: int):
        """Initialise the BinMapper.

        Parameters
        ----------
        max_bin : int
            Maximum number of bins to create.
        min_data_in_bin : int
            Minimum number of data points required in each bin.
        """
        self.max_bin = max_bin
        self.min_data_in_bin = min_data_in_bin

    def build_histograms(self, data: jnp.array):
        """
        Build histograms for all features.

        Parameters
        ----------
        data : jnp.array
            2D array where each column represents a feature.
        """
        return _build_feature_histogram(data, self.max_bin, self.min_data_in_bin)


def _build_feature_histogram(
    feature_values: jnp.ndarray, max_bin: int = 255, min_data_in_bin: int = 3
):
    """
    Build histograms for all features.

    Parameters
    ----------
    feature_values : jnp.ndarray
        2D array where each column represents a feature.
    max_bin : int, optional
        Maximum number of bins to create. Default is 255.
    min_data_in_bin : int, optional
        Minimum number of data points required in each bin. Default is 3.

    Returns
    -------
    bin_edges : jnp.ndarray
        Edges of the bins for each feature.
    histogram : jnp.ndarray
        Histogram of the feature values for each feature.
    bin_indices : jnp.ndarray
        Indices of the bins for each feature value.
    bin_edges_indices : jnp.ndarray
        Indices of the bin edges for each feature.
    """
    percentiles = jnp.linspace(100 / max_bin, 100 - 100 / max_bin, max_bin - 1)
    # percentiles = jnp.linspace(1 / max_bin, 1 - 1 / max_bin, max_bin - 1)

    bin_edges, histogram, bin_indices, bin_edges_indices = (
        _build_single_feature_histogram(
            feature_values, max_bin, min_data_in_bin, percentiles
        )
    )

    if jnp.any(
        (histogram < min_data_in_bin).sum(axis=0) - jnp.isinf(bin_edges).sum(axis=0) > 0
    ):
        jax.debug.print(
            "Warning: Some bins have less than the minimum required data points."
        )
    return (
        bin_edges,
        histogram,
        bin_indices,
        bin_edges_indices,
    )


@partial(jit, static_argnames=["max_bin", "min_data_in_bin"])
@partial(vmap, in_axes=(1, None, None, None), out_axes=1)
def _build_single_feature_histogram(
    feature_values: jnp.ndarray,
    max_bin: int = 255,
    min_data_in_bin: int = 3,
    percentiles: jnp.ndarray = jnp.linspace(0, 100, 256),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build histogram for feature values similar to LightGBM's histogram-based binning.

    Parameters
    ----------
    feature_values : jnp.ndarray
        Feature values to be binned.
    max_bin : int, optional
        Maximum number of bins to create. Default is 255.
    min_data_in_bin : int, optional
        Minimum number of data points required in each bin. Default is 3.
    percentiles : jnp.ndarray, optional
        Precomputed percentiles for max_bin. Default is jnp.linspace(0, 100, 256).

    Returns
    -------
    bin_edges : jnp.ndarray
        Edges of the bins.
    histogram : jnp.ndarray
        Histogram of the feature values.
    bin_indices : jnp.ndarray
        Indices of the bins for each feature value.
    bin_edges_indices : jnp.ndarray
        Indices of the bin edges.
    """
    bin_edges, histogram, bin_indices, bin_edges_indices = jax.lax.cond(
        jnp.unique(feature_values, size=feature_values.shape[0], fill_value=0).sum()
        == 1,
        lambda: _find_bins_binary(feature_values, max_bin),
        lambda: _find_bins_continuous(
            feature_values,
            max_bin,
            min_data_in_bin,
            percentiles,
        ),
    )
    return bin_edges, histogram, bin_indices, bin_edges_indices


def _find_bins_continuous(
    feature_values: jnp.ndarray,
    max_bin: int = 255,
    min_data_in_bin: int = 3,
    percentiles: jnp.ndarray = jnp.linspace(0, 100, 256),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Find bins for continuous feature values.

    Parameters
    ----------
    feature_values : jnp.ndarray
        Feature values to be binned.
    max_bin : int, optional
        Maximum number of bins to create. Default is 255.
    min_data_in_bin : int, optional
        Minimum number of data points required in each bin. Default is 3.
    percentiles : jnp.ndarray, optional
        Precomputed percentiles for max_bin. Default is jnp.linspace(0, 100, 256).

    Returns
    -------
    bin_edges : jnp.ndarray
        Edges of the bins.
    histogram : jnp.ndarray
        Histogram of the feature values.
    bin_indices : jnp.ndarray
        Indices of the bins for each feature value.
    bin_edges_indices : jnp.ndarray
        Indices of the bin edges.
    """

    def _while_cond(values):
        return values[0] > 1

    def _while_body(values):
        temp_bin = values[0]

        distinct_values = jnp.unique(
            feature_values,
            size=feature_values.shape[0],
            fill_value=jnp.inf,
        )
        n_distinct = jnp.isfinite(distinct_values).sum()

        used_percentiles = jnp.where(
            jnp.arange(max_bin - 1) < temp_bin - 1, percentiles, 0
        )
        used_percentiles = (
            used_percentiles / jnp.max(used_percentiles) * (100 - 100 / temp_bin)
        )
        # rescale to consider only distinct values and not infinite fillers
        used_percentiles = used_percentiles * n_distinct / feature_values.shape[0]
        used_percentiles = jnp.where(used_percentiles == 0.0, 100, used_percentiles)

        bin_edges = jnp.unique(
            jnp.percentile(
                distinct_values,
                used_percentiles,
                method="linear",
            ),
            size=max_bin - 1,
            fill_value=jnp.inf,
        )
        bin_edges = jnp.where(bin_edges == 0.0, 1e-6, bin_edges)
        bin_edges = jnp.where(
            bin_edges < feature_values.max(), bin_edges, jnp.inf
        ).sort()
        bin_indices = jnp.digitize(feature_values, bins=bin_edges, right=True)
        histogram = jnp.bincount(bin_indices, length=max_bin)
        temp_bin = jnp.where(
            (histogram < min_data_in_bin).sum() - (jnp.isinf(bin_edges).sum()) > 0,
            temp_bin - 1,
            1,
        )
        return temp_bin, bin_edges, histogram, bin_indices

    init_values = (
        max_bin,
        jnp.zeros((max_bin - 1,), dtype=jnp.float32),
        jnp.zeros((max_bin,), dtype=jnp.int32),
        jnp.zeros_like(feature_values, dtype=jnp.int32),
    )

    _, bin_edges, histogram, bin_indices = jax.lax.while_loop(
        _while_cond,
        _while_body,
        init_values,
    )

    bin_edges_indices = jnp.where(
        jnp.isposinf(bin_edges), jnp.inf, jnp.arange(max_bin - 1, dtype=jnp.int32)
    )

    return bin_edges, histogram, bin_indices, bin_edges_indices


def _find_bins_binary(
    feature_values: jnp.ndarray, max_bin: int = 255
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Find bins for binary feature values.

    Parameters
    ----------
    feature_values : jnp.ndarray
        Feature values to be binned.
    max_bin : int, optional
        Maximum number of bins to create. Default is 255.

    Returns
    -------
    bin_edges : jnp.ndarray
        Edges of the bins.
    histogram : jnp.ndarray
        Histogram of the feature values.
    bin_indices : jnp.ndarray
        Indices of the bins for each feature value.
    bin_edges_indices : jnp.ndarray
        Indices of the bin edges.
    """
    bin_edges = jnp.full((max_bin - 1,), jnp.inf)
    bin_edges = bin_edges.at[:1].set(jnp.array([0.5]))
    bin_indices = jnp.digitize(feature_values, bins=bin_edges, right=True)
    histogram = jnp.bincount(bin_indices, length=len(bin_edges) + 1)
    bin_edges_indices = jnp.where(
        jnp.isposinf(bin_edges), jnp.inf, jnp.arange(max_bin - 1, dtype=jnp.int32)
    )
    return bin_edges, histogram, bin_indices, bin_edges_indices
