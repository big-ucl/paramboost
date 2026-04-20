import jax.numpy as jnp


def cel_loss(preds: jnp.ndarray, labels: jnp.ndarray):
    """
    Compute the categorical cross-entropy loss.

    Parameters
    ----------
    preds : jnp.ndarray
        Predicted probabilities for each class (shape: [n_samples, n_classes]).
    labels : jnp.ndarray
        One-hot encoded true labels (shape: [n_samples, n_classes]).

    Returns
    -------
    loss : float
        The computed categorical cross-entropy loss.
    """
    preds = jnp.clip(preds, 1e-15, 1 - 1e-15)
    loss = -jnp.mean(jnp.sum(labels * jnp.log(preds), axis=1))
    return loss


def binary_loss(preds: jnp.ndarray, labels: jnp.ndarray):
    """
    Compute the binary cross-entropy loss.

    Parameters
    ----------
    preds : jnp.ndarray
        Predicted probabilities (shape: [n_samples]).
    labels : jnp.ndarray
        True binary labels (shape: [n_samples]).

    Returns
    -------
    loss : float
        The computed binary cross-entropy loss.
    """
    preds = jnp.clip(preds, 1e-15, 1 - 1e-15).reshape(-1)
    loss = -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))
    return loss


def regression_loss(preds: jnp.ndarray, labels: jnp.ndarray):
    """
    Compute the mean squared error loss.

    Parameters
    ----------
    preds : jnp.ndarray
        Predicted values (shape: [n_samples]).
    labels : jnp.ndarray
        True values (shape: [n_samples]).

    Returns
    -------
    loss : float
        The computed mean squared error loss.
    """
    loss = jnp.mean((preds.reshape(-1) - labels) ** 2)
    return loss
