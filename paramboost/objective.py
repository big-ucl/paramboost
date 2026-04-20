import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


@partial(vmap, in_axes=(1, 1, None), out_axes=0)
def f_obj_cel(
    preds: jnp.ndarray, labels: jnp.ndarray, num_classes: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the gradient and hessian for categorical cross-entropy loss.

    Parameters
    ----------
    preds : jnp.ndarray
        Predicted probabilities for each class (shape: [n_classes]).
    labels : jnp.ndarray
        One-hot encoded true labels (shape: [n_classes]).
    num_classes : int
        Number of classes.

    Returns
    -------
    grad : jnp.ndarray
        Gradient of the loss (shape: [n_classes]).
    hess : jnp.ndarray
        Hessian of the loss (shape: [n_classes]).
    """
    factor = num_classes / (num_classes - 1)

    grad = preds - labels

    hess = preds * (1 - preds) * factor

    return grad, hess


def f_obj_binary(
    preds: jnp.ndarray, labels: jnp.ndarray, _
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the gradient and hessian for binary cross-entropy loss.

    Parameters
    ----------
    preds : jnp.ndarray
        Predicted probabilities (shape: [n_samples]).
    labels : jnp.ndarray
        True binary labels (shape: [n_samples]).

    Returns
    -------
    grad : jnp.ndarray
        Gradient of the loss (shape: [1, n_samples]).
    hess : jnp.ndarray
        Hessian of the loss (shape: [1, n_samples]).
    """

    preds = preds.reshape(-1)
    grad = preds - labels

    hess = jnp.maximum(preds * (1 - preds), 1e-6)

    return grad.reshape(1, -1), hess.reshape(1, -1)


def f_obj_regression(
    preds: jnp.ndarray, labels: jnp.ndarray, _
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the gradient and hessian for mean squared error loss.

    Parameters
    ----------
    preds : jnp.ndarray
        Predicted values (shape: [n_samples]).
    labels : jnp.ndarray
        True values (shape: [n_samples]).

    Returns
    -------
    grad : jnp.ndarray
        Gradient of the loss (shape: [1, n_samples]).
    hess : jnp.ndarray
        Hessian of the loss (shape: [1, n_samples]).
    """
    grad = 2 * (preds.reshape(-1) - labels)

    hess = 2 * jnp.ones_like(grad)

    return grad.reshape(1, -1), hess.reshape(1, -1)
