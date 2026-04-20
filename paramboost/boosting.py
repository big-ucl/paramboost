import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import jax


def _update_one_iter(
    grad: jnp.ndarray,
    hess: jnp.ndarray,
    data_bin_indices: jnp.ndarray,
    data_bin_indices_val: jnp.ndarray,
    data_for_grads: jnp.ndarray,
    data_for_hess: jnp.ndarray,
    data_for_grads_left: jnp.ndarray,
    data_for_hess_left: jnp.ndarray,
    data_val_precomputed: jnp.ndarray,
    data: jnp.ndarray,
    data_val: jnp.ndarray,
    hist_presummed: jnp.ndarray,
    sum_hist: jnp.ndarray,
    boostable_indices: jnp.ndarray,
    bin_edges: jnp.ndarray,
    bin_indices: jnp.ndarray,
    ensembles: jnp.ndarray,
    feature_importance_dict: dict,
    inner_preds: jnp.ndarray,
    inner_preds_val: jnp.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    monotone_constraints: jnp.ndarray,
    curvature_constraints: jnp.ndarray,
    learning_rate: float,
    min_sum_hessian_in_leaf: float,
    min_data_in_leaf: int,
    min_gain_to_split: float,
    leading_indices: jnp.ndarray,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    lower_bound_left_c: jnp.ndarray,
    upper_bound_left_c: jnp.ndarray,
    lower_bound_right_c: jnp.ndarray,
    upper_bound_right_c: jnp.ndarray,
    order: jnp.ndarray,
    continuity: jnp.ndarray,
    binary_features: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    dict,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    grad_presummed_constant = presum_binned_grads(
        grad, data_bin_indices, bin_edges.shape[0]
    )  # degree 0
    grad_presummed_boostable_poly = grad @ data_for_grads_left  # degree > 0
    grad_presummed_poly = (
        jnp.zeros_like(lower_bound_right[:, :, 1:, :])
        .reshape(lower_bound_right.shape[0], -1)
        .at[:, boostable_indices[-1]]
        .set(grad_presummed_boostable_poly)
        .reshape(lower_bound_right[:, :, 1:, :].shape)
        .astype(jnp.float32)
    )
    # concatenate degree 0 and degree > 0
    grad_presummed = jnp.zeros_like(lower_bound_right)
    grad_presummed = grad_presummed.at[:, :, 0, :].set(grad_presummed_constant)
    grad_presummed = grad_presummed.at[:, :, 1:, :].set(grad_presummed_poly)

    sum_grad_constant = grad.sum(axis=1)  # degree 0
    sum_grad_boostable_poly = grad @ data_for_grads  # degree > 0
    sum_grad_poly = (
        jnp.zeros_like(lower_bound_right[:, :, 1:, :])
        .reshape(lower_bound_right.shape[0], -1)
        .at[:, boostable_indices[-1]]
        .set(sum_grad_boostable_poly)
        .reshape(lower_bound_right[:, :, 1:, :].shape)
        .astype(jnp.float32)
    )
    # concatenate degree 0 and degree > 0
    sum_grad = jnp.zeros_like(lower_bound_right)
    sum_grad = sum_grad.at[:, :, 0, :].set(sum_grad_constant[:, None, None])
    sum_grad = sum_grad.at[:, :, 1:, :].set(sum_grad_poly)

    hess_presummed_constant = presum_binned_grads(  # degree 0
        hess, data_bin_indices, bin_edges.shape[0]
    )
    hess_presummed_boostable_poly = hess @ data_for_hess_left  # degree > 0
    hess_presummed_poly = (
        jnp.zeros_like(lower_bound_left[:, :, 1:, :])
        .reshape(lower_bound_left.shape[0], -1)
        .at[:, boostable_indices[-1]]
        .set(hess_presummed_boostable_poly)
        .reshape(lower_bound_left[:, :, 1:, :].shape)
        .astype(jnp.float32)
    )
    # concatenate degree 0 and degree > 0
    hess_presummed = jnp.zeros_like(lower_bound_left)
    hess_presummed = hess_presummed.at[:, :, 0, :].set(hess_presummed_constant)
    hess_presummed = hess_presummed.at[:, :, 1:, :].set(hess_presummed_poly)

    sum_hess_constant = hess.sum(axis=1)  # degree 0
    sum_hess_boostable_poly = hess @ data_for_hess  # degree > 0
    sum_hess_poly = (
        jnp.zeros_like(lower_bound_left[:, :, 1:, :])
        .reshape(lower_bound_left.shape[0], -1)
        .at[:, boostable_indices[-1]]
        .set(sum_hess_boostable_poly)
        .reshape(lower_bound_left[:, :, 1:, :].shape)
        .astype(jnp.float32)
    )
    # concatenate degree 0 and degree > 0
    sum_hess = jnp.zeros_like(lower_bound_left)
    sum_hess = sum_hess.at[:, :, 0, :].set(sum_hess_constant[:, None, None])
    sum_hess = sum_hess.at[:, :, 1:, :].set(sum_hess_poly)

    gain, left_leaf, right_leaf, split_point = _boost(
        hist_presummed,
        grad_presummed,
        hess_presummed,
        sum_hist,
        sum_grad,
        sum_hess,
        lambda_l1,
        lambda_l2,
        learning_rate,
        min_sum_hessian_in_leaf,
        min_data_in_leaf,
        min_gain_to_split,
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
        monotone_constraints,
        curvature_constraints,
        binary_features,
    )

    best_gain, best_left_leaf, best_right_leaf, best_indices = (  # unraveled indices
        get_best_values(gain, left_leaf, right_leaf, leading_indices, split_point)
    )

    ensembles = (
        _update_ensembles(  # need a function according to the numbe rof leading axis
            ensembles,
            best_left_leaf.ravel(),
            best_right_leaf.ravel(),
            best_gain.ravel(),
            jax.tree_util.tree_map(lambda x: x.ravel(), best_indices),
            bin_edges,
        )
    )

    lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right = (
        _update_bounds_monotonicity(
            ensembles,
            jax.tree_util.tree_map(lambda x: x.ravel(), best_indices),
            best_gain.ravel(),
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            bin_edges,
            jnp.concatenate([mask, mask, mask], axis=1),
            monotone_constraints,
        )
    )

    lower_bound_left_c, upper_bound_left_c, lower_bound_right_c, upper_bound_right_c = (
        _update_bounds_curvature(
            ensembles,
            jax.tree_util.tree_map(lambda x: x.ravel(), best_indices),
            best_gain.ravel(),
            lower_bound_left_c,
            upper_bound_left_c,
            lower_bound_right_c,
            upper_bound_right_c,
            bin_edges,
            jnp.concatenate([mask, mask], axis=1),
            curvature_constraints,
        )
    )

    inner_preds, inner_preds_val = _update_preds(
        inner_preds,
        inner_preds_val,
        best_left_leaf.ravel(),
        best_right_leaf.ravel(),
        jax.tree_util.tree_map(lambda x: x.ravel(), best_indices),
        data_bin_indices,
        data_bin_indices_val,
        data,
        data_val,
        best_gain.ravel(),
        bin_edges,
    )

    feature_importance_dict = _update_feature_importance(
        feature_importance_dict,
        best_gain,
        best_indices,
    )

    return (
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
    )


@partial(
    vmap,  # class axis
    in_axes=(
        None,
        0,
        0,
        None,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
    ),
    out_axes=(0, 0, 0, 0),
)
@partial(
    vmap,  # feature axis
    in_axes=(
        1,
        0,
        0,
        None,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        0,
        0,
        0,
    ),
    out_axes=(0, 0, 0, 0),
)
@partial(
    vmap,  # order axis
    in_axes=(
        None,
        0,
        0,
        None,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        None,
        None,
    ),
    out_axes=(0, 0, 0, 0),
)
def _boost(
    hist_presummed: jnp.ndarray,
    grad_presummed: jnp.ndarray,
    hess_presummed: jnp.ndarray,
    sum_hist: jnp.ndarray,
    sum_grad: jnp.ndarray,
    sum_hess: jnp.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    learning_rate: float,
    min_sum_hessian_in_leaf: float,
    min_data_in_leaf: int,
    min_gain_to_split: float,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    lower_bound_left_c: jnp.ndarray,
    upper_bound_left_c: jnp.ndarray,
    lower_bound_right_c: jnp.ndarray,
    upper_bound_right_c: jnp.ndarray,
    order: jnp.ndarray,
    continuity: jnp.ndarray,
    monotonicity: jnp.ndarray,
    curvature: jnp.ndarray,
    binary_features: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gain, left_leaf, right_leaf, split_point = jax.lax.cond(
        # use of lower right bound is arbitrary,
        # any right bounds could be used (not left
        # since bounds are alternating because of (x-s)^order sign)
        jnp.min(lower_bound_right) == jnp.inf,  # no splitting allowed,
        lambda: (
            jnp.array(-jnp.inf, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
        ),
        lambda: _check_binary_feature_split(
            hist_presummed,
            grad_presummed,
            hess_presummed,
            sum_hist,
            sum_grad,
            sum_hess,
            lambda_l1,
            lambda_l2,
            learning_rate,
            min_sum_hessian_in_leaf,
            min_data_in_leaf,
            min_gain_to_split,
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
            monotonicity,
            curvature,
            binary_features,
        ),
    )

    return gain, left_leaf, right_leaf, split_point


def _check_binary_feature_split(
    hist_presummed: jnp.ndarray,
    grad_presummed: jnp.ndarray,
    hess_presummed: jnp.ndarray,
    sum_hist: jnp.ndarray,
    sum_grad: jnp.ndarray,
    sum_hess: jnp.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    learning_rate: float,
    min_sum_hessian_in_leaf: float,
    min_data_in_leaf: int,
    min_gain_to_split: float,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    lower_bound_left_c: jnp.ndarray,
    upper_bound_left_c: jnp.ndarray,
    lower_bound_right_c: jnp.ndarray,
    upper_bound_right_c: jnp.ndarray,
    order: jnp.ndarray,
    continuity: jnp.ndarray,
    monotonicity: jnp.ndarray,
    curvature: jnp.ndarray,
    binary_features: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gain, left_leaf, right_leaf, split_point = jax.lax.cond(
        binary_features > 0,  # if any binary features
        lambda: _compute_actual_split(  # compute split for binary features and skip the continuity constraint as there are only one split point possible
            hist_presummed,
            grad_presummed,
            hess_presummed,
            sum_hist,
            sum_grad,
            sum_hess,
            lambda_l1,
            lambda_l2,
            learning_rate,
            min_sum_hessian_in_leaf,
            min_data_in_leaf,
            min_gain_to_split,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            lower_bound_left_c,
            upper_bound_left_c,
            lower_bound_right_c,
            upper_bound_right_c,
            order,
            monotonicity,
            curvature,
        ),
        lambda: _compute_split(  # compute split normally
            hist_presummed,
            grad_presummed,
            hess_presummed,
            sum_hist,
            sum_grad,
            sum_hess,
            lambda_l1,
            lambda_l2,
            learning_rate,
            min_sum_hessian_in_leaf,
            min_data_in_leaf,
            min_gain_to_split,
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
            monotonicity,
            curvature,
        ),
    )

    return gain, left_leaf, right_leaf, split_point


def _compute_split(
    hist_presummed: jnp.ndarray,
    grad_presummed: jnp.ndarray,
    hess_presummed: jnp.ndarray,
    sum_hist: jnp.ndarray,
    sum_grad: jnp.ndarray,
    sum_hess: jnp.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    learning_rate: float,
    min_sum_hessian_in_leaf: float,
    min_data_in_leaf: int,
    min_gain_to_split: float,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    lower_bound_left_c: jnp.ndarray,
    upper_bound_left_c: jnp.ndarray,
    lower_bound_right_c: jnp.ndarray,
    upper_bound_right_c: jnp.ndarray,
    order: jnp.ndarray,
    continuity: jnp.ndarray,
    monotonicity: jnp.ndarray,
    curvature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gain, left_leaf, right_leaf, split_point = jax.lax.cond(
        order > continuity,
        lambda: _compute_actual_split(  # actually splitting
            hist_presummed,
            grad_presummed,
            hess_presummed,
            sum_hist,
            sum_grad,
            sum_hess,
            lambda_l1,
            lambda_l2,
            learning_rate,
            min_sum_hessian_in_leaf,
            min_data_in_leaf,
            min_gain_to_split,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            lower_bound_left_c,
            upper_bound_left_c,
            lower_bound_right_c,
            upper_bound_right_c,
            order,
            monotonicity,
            curvature,
        ),
        lambda: _compute_simple_split(  # no splitting -> gradient descent step
            sum_grad[0],
            sum_hess[0],
            lambda_l1,
            lambda_l2,
            learning_rate,
            min_sum_hessian_in_leaf,
            min_gain_to_split,
            lower_bound_left[0],
            upper_bound_left[0],
            lower_bound_right[0],
            upper_bound_right[0],
            lower_bound_left_c[0],
            upper_bound_left_c[0],
            lower_bound_right_c[0],
            upper_bound_right_c[0],
            order,
        ),
    )

    return gain, left_leaf, right_leaf, split_point


def _compute_actual_split(
    hist_presummed: jnp.ndarray,
    grad_presummed: jnp.ndarray,
    hess_presummed: jnp.ndarray,
    sum_hist: jnp.ndarray,
    sum_grad: jnp.ndarray,
    sum_hess: jnp.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    learning_rate: float,
    min_sum_hessian_in_leaf: float,
    min_data_in_leaf: int,
    min_gain_to_split: float,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    lower_bound_left_c: jnp.ndarray,
    upper_bound_left_c: jnp.ndarray,
    lower_bound_right_c: jnp.ndarray,
    upper_bound_right_c: jnp.ndarray,
    order: jnp.ndarray,
    monotonicity: jnp.ndarray,
    curvature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gain, left_leaf, right_leaf = _check_split_point_validity(
        hist_presummed,
        grad_presummed,
        hess_presummed,
        sum_hist,
        sum_grad,
        sum_hess,
        lambda_l1,
        lambda_l2,
        learning_rate,
        min_sum_hessian_in_leaf,
        min_data_in_leaf,
        min_gain_to_split,
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
        lower_bound_left_c,
        upper_bound_left_c,
        lower_bound_right_c,
        upper_bound_right_c,
        order,
        monotonicity,
        curvature,
    )
    best_split = jnp.argmax(gain)  # choose best split point
    return (
        gain[best_split],
        left_leaf[best_split],
        right_leaf[best_split],
        best_split,
    )


@partial(
    vmap,  # split point axis
    in_axes=(
        0,
        0,
        0,
        None,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        None,
    ),
    out_axes=(0, 0, 0),
)
def _check_split_point_validity(
    hist_presummed: jnp.ndarray,
    grad_presummed: jnp.ndarray,
    hess_presummed: jnp.ndarray,
    sum_hist: jnp.ndarray,
    sum_grad: jnp.ndarray,
    sum_hess: jnp.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    learning_rate: float,
    min_sum_hessian_in_leaf: float,
    min_data_in_leaf: int,
    min_gain_to_split: float,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    lower_bound_left_c: jnp.ndarray,
    upper_bound_left_c: jnp.ndarray,
    lower_bound_right_c: jnp.ndarray,
    upper_bound_right_c: jnp.ndarray,
    order: jnp.ndarray,
    monotonicity: jnp.ndarray,
    curvature: jnp.ndarray,
):
    gain, left_leaf, right_leaf = jax.lax.cond(
        hist_presummed == sum_hist,  # invalid split point
        lambda: (
            jnp.array(-jnp.inf, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        ),
        lambda: _compute_one_split(
            hist_presummed,
            grad_presummed,
            hess_presummed,
            sum_hist,
            sum_grad,
            sum_hess,
            lambda_l1,
            lambda_l2,
            learning_rate,
            min_sum_hessian_in_leaf,
            min_data_in_leaf,
            min_gain_to_split,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            lower_bound_left_c,
            upper_bound_left_c,
            lower_bound_right_c,
            upper_bound_right_c,
            order,
            monotonicity,
            curvature,
        ),
    )

    return (
        gain,
        left_leaf,
        right_leaf,
    )


def _compute_one_split(
    hist_presummed: jnp.ndarray,
    grad_presummed: jnp.ndarray,
    hess_presummed: jnp.ndarray,
    sum_hist: jnp.ndarray,
    sum_grad: jnp.ndarray,
    sum_hess: jnp.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    learning_rate: float,
    min_sum_hessian_in_leaf: float,
    min_data_in_leaf: int,
    min_gain_to_split: float,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    lower_bound_left_c: jnp.ndarray,
    upper_bound_left_c: jnp.ndarray,
    lower_bound_right_c: jnp.ndarray,
    upper_bound_right_c: jnp.ndarray,
    order: jnp.ndarray,
    monotonicity: jnp.ndarray,
    curvature: jnp.ndarray,
):
    # agregate bins left and right of split points
    sum_hist_left = hist_presummed
    sum_hist_right = sum_hist - sum_hist_left
    sum_grad_left = grad_presummed
    sum_grad_right = sum_grad - sum_grad_left
    sum_hess_left = hess_presummed
    sum_hess_right = sum_hess - sum_hess_left

    # compute gains and leaves
    current_gain = compute_gain(sum_grad, sum_hess, lambda_l1, lambda_l2)

    # left
    left_leaf = (
        compute_leaf(sum_grad_left, sum_hess_left, lambda_l1, lambda_l2) * learning_rate
    )
    left_gain = compute_gain(sum_grad_left, sum_hess_left, lambda_l1, lambda_l2)
    # right
    right_leaf = (
        compute_leaf(sum_grad_right, sum_hess_right, lambda_l1, lambda_l2)
        * learning_rate
    )
    right_gain = compute_gain(sum_grad_right, sum_hess_right, lambda_l1, lambda_l2)

    if min_data_in_leaf > 0:
        mdil_mask_left = create_mask(sum_hist_left, min_data_in_leaf)
        mdil_mask_right = create_mask(sum_hist_right, min_data_in_leaf)
        left_gain = ignore_splits(left_gain, mdil_mask_left, value=-jnp.inf)
        left_leaf = ignore_splits(left_leaf, mdil_mask_left, value=0)
        right_gain = ignore_splits(right_gain, mdil_mask_right, value=-jnp.inf)
        right_leaf = ignore_splits(right_leaf, mdil_mask_right, value=0)

    # apply monotonic constraints
    left_gain, right_gain, left_leaf, right_leaf = jax.lax.cond(
        order > 0,
        lambda: disregard_bound_violation(
            left_gain,
            right_gain,
            left_leaf,
            right_leaf,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            order,
        ),
        lambda: disregard_nonmonotonic_split(
            left_gain, right_gain, left_leaf, right_leaf, monotonicity
        ),
    )

    # apply curvature constraints
    left_gain, right_gain, left_leaf, right_leaf = jax.lax.switch(
        order,
        (
            lambda: (left_gain, right_gain, left_leaf, right_leaf),
            lambda: disregard_wrong_curvature_split(
                left_gain, right_gain, left_leaf, right_leaf, curvature
            ),
            lambda: disregard_bound_c_violation(
                left_gain,
                right_gain,
                left_leaf,
                right_leaf,
                lower_bound_left_c,
                upper_bound_left_c,
                lower_bound_right_c,
                upper_bound_right_c,
                order,
            ),
            lambda: disregard_bound_c_violation(
                left_gain,
                right_gain,
                left_leaf,
                right_leaf,
                lower_bound_left_c,
                upper_bound_left_c,
                lower_bound_right_c,
                upper_bound_right_c,
                order,
            ),
        ),
    )

    # gain of split
    gain = sum_gains(left_gain, right_gain, current_gain)

    if min_sum_hessian_in_leaf > 0:
        mshil_mask_left = create_mask(sum_hess_left, min_sum_hessian_in_leaf)
        mshil_mask_right = create_mask(sum_hess_right, min_sum_hessian_in_leaf)
        gain = ignore_splits(gain, mshil_mask_left, value=0.0)
        gain = ignore_splits(gain, mshil_mask_right, value=0.0)
        left_leaf = ignore_splits(left_leaf, mshil_mask_left, value=0.0)
        right_leaf = ignore_splits(right_leaf, mshil_mask_right, value=0.0)

    if min_gain_to_split > 0:
        mgts_mask = create_mask(gain, min_gain_to_split)
        gain = ignore_splits(gain, mgts_mask, value=-jnp.inf)
        left_leaf = ignore_splits(left_leaf, mgts_mask, value=0)
        right_leaf = ignore_splits(right_leaf, mgts_mask, value=0)

    return (
        gain,
        left_leaf,
        right_leaf,
    )


def _compute_simple_split(
    sum_grad: jnp.ndarray,
    sum_hess: jnp.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    learning_rate: float,
    min_sum_hessian_in_leaf: float,
    min_gain_to_split: float,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    lower_bound_left_c: jnp.ndarray,
    upper_bound_left_c: jnp.ndarray,
    lower_bound_right_c: jnp.ndarray,
    upper_bound_right_c: jnp.ndarray,
    order: jnp.ndarray,
):
    # compute gains and leaves
    gain = compute_gain(sum_grad, sum_hess, lambda_l1, lambda_l2)

    # left
    left_leaf = compute_leaf(sum_grad, sum_hess, lambda_l1, lambda_l2) * learning_rate

    # apply monotonic constraints and regularisation
    gain, _, left_leaf_l, left_leaf_r = jax.lax.cond(
        order > 0,
        lambda: disregard_bound_violation(
            gain,
            gain,  # right gain is same as left gain since no split
            left_leaf,
            left_leaf,  # right leaf is same as left leaf since no split
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            order,
        ),
        lambda: (gain, gain, left_leaf, left_leaf),  # no split, so right same as left
    )
    left_leaf = jnp.where(left_leaf_l * left_leaf_r == 0, 0.0, left_leaf_l)
    gain = jnp.where(left_leaf_l * left_leaf_r == 0, 0.0, gain)
    # apply curvature constraints
    gain, _, left_leaf_l, left_leaf_r = jax.lax.cond(
        order > 1,
        lambda: disregard_bound_c_violation(
            gain,
            gain,  # right gain is same as left gain since no split
            left_leaf,
            left_leaf,  # right leaf is same as left leaf since no split
            lower_bound_left_c,
            upper_bound_left_c,
            lower_bound_right_c,
            upper_bound_right_c,
            order,
        ),
        lambda: (gain, gain, left_leaf, left_leaf),  # no split, so right same as left
    )
    left_leaf = jnp.where(left_leaf_l * left_leaf_r == 0, 0.0, left_leaf_l)
    gain = jnp.where(left_leaf_l * left_leaf_r == 0, 0.0, gain)

    if min_sum_hessian_in_leaf > 0:
        # min_sum_hessian_in_leaf = jnp.minimum(
        #     min_sum_hessian_in_leaf, min_sum_hessian_in_leaf ** (2 * order)
        # )
        mshil_mask = create_mask(sum_hess, min_sum_hessian_in_leaf)
        gain = ignore_splits(gain, mshil_mask, value=-jnp.inf)
        left_leaf = ignore_splits(left_leaf, mshil_mask, value=0)

    if min_gain_to_split > 0:
        mgts_mask = create_mask(gain, min_gain_to_split)
        gain = ignore_splits(gain, mgts_mask, value=-jnp.inf)
        left_leaf = ignore_splits(left_leaf, mgts_mask, value=0)

    return (
        gain,
        left_leaf,
        left_leaf,  # right leaf is same as left leaf since no split
        jnp.array(0, dtype=jnp.int32),  # fake split point since no split is done
    )


def compute_gain(
    sum_grad: jnp.ndarray, sum_hess: jnp.ndarray, l1: float, l2: float
) -> jnp.ndarray:
    """
    Compute the gain given the sum of gradients and hessians.

    Parameters
    ----------
    sum_grad : jnp.ndarray
        The sum of gradients.
    sum_hess : jnp.ndarray
        The sum of hessians.
    l1 : float
        The L1 regularisation parameter.
    l2 : float
        The L2 regularisation parameter.

    Returns
    -------
    jnp.ndarray
        The computed gain.
    """
    temp_gain = jnp.maximum(jnp.abs(sum_grad) - l1, 0) ** 2 / (sum_hess + l2)
    gain = jnp.nan_to_num(temp_gain, nan=-jnp.inf)
    return gain


def compute_leaf(
    sum_grad: jnp.ndarray, sum_hess: jnp.ndarray, l1: float, l2: float
) -> jnp.ndarray:
    """
    Compute the leaf value given the sum of gradients and hessians.

    Parameters
    ----------
    sum_grad : jnp.ndarray
        The sum of gradients.
    sum_hess : jnp.ndarray
        The sum of hessians.
    l1 : float
        The L1 regularisation parameter.
    l2 : float
        The L2 regularisation parameter.

    Returns
    -------
    jnp.ndarray
        The computed leaf value.
    """
    temp_leaf = (
        -jnp.sign(sum_grad) * jnp.maximum(jnp.abs(sum_grad) - l1, 0) / (sum_hess + l2)
    )
    leaf = jnp.nan_to_num(temp_leaf, nan=0)
    return leaf


def create_mask(left_side: jnp.ndarray, right_side: jnp.ndarray) -> jnp.ndarray:
    """
    Create a mask where elements of left_side are less than elements of right_side.

    Parameters
    ----------
    left_side : jnp.ndarray
        The left side array.
    right_side : jnp.ndarray
        The right side array.

    Returns
    -------
    jnp.ndarray
        A boolean mask array.
    """
    mask = left_side < right_side
    return mask


def ignore_splits(vec: jnp.ndarray, mask: jnp.ndarray, value=0) -> jnp.ndarray:
    return jnp.where(mask, value, vec)


def sum_gains(
    left_gain: jnp.ndarray, right_gain: jnp.ndarray, no_split_gain: jnp.ndarray
) -> jnp.ndarray:
    temp_gain = 0.5 * (left_gain + right_gain - no_split_gain)
    gain = jnp.nan_to_num(temp_gain, nan=-jnp.inf)
    return gain


def disregard_nonmonotonic_split(
    left_gain: jnp.ndarray,
    right_gain: jnp.ndarray,
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    monotonic_constraint: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    left_gain = jnp.where(
        monotonic_constraint * (right_leaf - left_leaf) < 0, -jnp.inf, left_gain
    )
    right_gain = jnp.where(
        monotonic_constraint * (right_leaf - left_leaf) < 0, -jnp.inf, right_gain
    )
    left_leaf = jnp.where(
        monotonic_constraint * (right_leaf - left_leaf) < 0, 0.0, left_leaf
    )
    right_leaf = jnp.where(
        monotonic_constraint * (right_leaf - left_leaf) < 0, 0.0, right_leaf
    )
    return left_gain, right_gain, left_leaf, right_leaf


def disregard_wrong_curvature_split(
    left_gain: jnp.ndarray,
    right_gain: jnp.ndarray,
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    curvature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    left_gain = jnp.where(curvature * (right_leaf - left_leaf) < 0, -jnp.inf, left_gain)
    right_gain = jnp.where(
        curvature * (right_leaf - left_leaf) < 0, -jnp.inf, right_gain
    )
    left_leaf = jnp.where(curvature * (right_leaf - left_leaf) < 0, 0.0, left_leaf)
    right_leaf = jnp.where(curvature * (right_leaf - left_leaf) < 0, 0.0, right_leaf)
    return left_gain, right_gain, left_leaf, right_leaf


def disregard_bound_violation(
    left_gain: jnp.ndarray,
    right_gain: jnp.ndarray,
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    order: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # if square coefficients, need to account division by negatice (x - s) for the inequality
    left_sign = 2 * (order % 2) - 1  # 1 if odd order, -1 if even order

    mc_mask = create_mask(
        left_sign * left_leaf, left_sign * lower_bound_left
    )  # if left_leaf < lower_bound_left (-left_leaf < -lower_bound_left for even order)
    left_gain = ignore_splits(left_gain, mc_mask, value=0)
    left_leaf = ignore_splits(left_leaf, mc_mask, value=0)
    mc_mask_right = create_mask(
        right_leaf, lower_bound_right
    )  # if right_leaf < lower_bound_right
    right_gain = ignore_splits(right_gain, mc_mask_right, value=0)
    right_leaf = ignore_splits(right_leaf, mc_mask_right, value=0)

    mc_mask_upper = create_mask(
        left_sign * upper_bound_left, left_sign * left_leaf
    )  # if left_leaf > upper_bound_left ( -upper_bound_left > -left_leaf for even order)
    left_gain = ignore_splits(left_gain, mc_mask_upper, value=0)
    left_leaf = ignore_splits(left_leaf, mc_mask_upper, value=0)
    mc_mask_right_upper = create_mask(upper_bound_right, right_leaf)
    right_gain = ignore_splits(right_gain, mc_mask_right_upper, value=0)
    right_leaf = ignore_splits(right_leaf, mc_mask_right_upper, value=0)
    return left_gain, right_gain, left_leaf, right_leaf


def disregard_bound_c_violation(
    left_gain: jnp.ndarray,
    right_gain: jnp.ndarray,
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    lower_bound_left_c: jnp.ndarray,
    upper_bound_left_c: jnp.ndarray,
    lower_bound_right_c: jnp.ndarray,
    upper_bound_right_c: jnp.ndarray,
    order: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # if square coefficients, need to account division by negatice (x - s) for the inequality
    left_sign = 2 * ((order + 1) % 2) - 1  # 1 if even order, -1 if odd order

    mc_mask = create_mask(
        left_sign * left_leaf, left_sign * lower_bound_left_c
    )  # if left_leaf < lower_bound_left (-left_leaf < -lower_bound_left for odd order)
    left_gain = ignore_splits(left_gain, mc_mask, value=0)
    left_leaf = ignore_splits(left_leaf, mc_mask, value=0)
    mc_mask_right = create_mask(
        right_leaf, lower_bound_right_c
    )  # if right_leaf < lower_bound_right
    right_gain = ignore_splits(right_gain, mc_mask_right, value=0)
    right_leaf = ignore_splits(right_leaf, mc_mask_right, value=0)

    mc_mask_upper = create_mask(
        left_sign * upper_bound_left_c, left_sign * left_leaf
    )  # if left_leaf > upper_bound_left ( -upper_bound_left > -left_leaf for odd order)
    left_gain = ignore_splits(left_gain, mc_mask_upper, value=0)
    left_leaf = ignore_splits(left_leaf, mc_mask_upper, value=0)
    mc_mask_right_upper = create_mask(upper_bound_right_c, right_leaf)
    right_gain = ignore_splits(right_gain, mc_mask_right_upper, value=0)
    right_leaf = ignore_splits(right_leaf, mc_mask_right_upper, value=0)
    return left_gain, right_gain, left_leaf, right_leaf


def get_best_values(
    gain: jnp.ndarray,
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    leading_indices,
    split_points: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple]:
    gain_reshaped = gain.reshape(*gain.shape[: len(leading_indices)], -1)
    left_leaf_reshaped = left_leaf.reshape(*left_leaf.shape[: len(leading_indices)], -1)
    right_leaf_reshaped = right_leaf.reshape(
        *right_leaf.shape[: len(leading_indices)], -1
    )
    split_points_reshaped = split_points.reshape(
        *split_points.shape[: len(leading_indices)], -1
    )
    best_indices = jnp.argmax(gain_reshaped, axis=-1)
    best_split_points = split_points_reshaped[*leading_indices, best_indices]
    return (
        gain_reshaped[
            *leading_indices,
            best_indices,
        ],
        left_leaf_reshaped[
            *leading_indices,
            best_indices,
        ],
        right_leaf_reshaped[
            *leading_indices,
            best_indices,
        ],
        leading_indices
        + jnp.unravel_index(best_indices, gain.shape[len(leading_indices) :])
        + tuple([best_split_points]),
    )


@partial(vmap, in_axes=(0, None, None), out_axes=0)
@partial(vmap, in_axes=(None, 1, None), out_axes=0)
def presum_binned_grads(
    grads: jnp.ndarray, data_bin_indices: jnp.ndarray, n_bins: int
) -> jnp.ndarray:
    return jax.ops.segment_sum(
        data=grads, segment_ids=data_bin_indices, num_segments=n_bins
    ).cumsum(axis=0)

@partial(vmap, in_axes=(1, None, None), out_axes=1)
def presum_binned_array(arr, weights, n_bins):
    return jax.ops.segment_sum(
        data=weights, segment_ids=arr, num_segments=n_bins
    ).cumsum(axis=0)


def _update_ensembles(
    ensembles: jnp.ndarray,
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    gain: jnp.ndarray,
    indices: tuple,
    bin_edges: jnp.ndarray,
) -> jnp.ndarray:
    carry = (
        ensembles,
        bin_edges,
    )
    xs = (
        indices,
        left_leaf,
        right_leaf,
        gain,
    )

    def update_step(carry, x):
        ensembles, bin_edges = carry
        indices, left_leaf, right_leaf, gain = x

        ensembles, bin_edges = jax.lax.cond(
            gain > 0,
            lambda: _update_ensembles_positive_gain(
                ensembles, left_leaf, right_leaf, indices, bin_edges
            ),
            lambda: (ensembles, bin_edges),
        )

        return (ensembles, bin_edges), None

    carry, _ = jax.lax.scan(update_step, carry, xs)
    ensembles, _ = carry
    return ensembles


def _update_ensembles_positive_gain(
    ensembles: jnp.ndarray,
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    indices: tuple,
    bin_edges: jnp.ndarray,
) -> jnp.ndarray:
    split_point = indices[-1]
    leading_indices = indices[:-2]
    bins = jnp.arange(bin_edges.shape[0] + 1)
    n_order = ensembles.shape[2]

    # update ensembles with new leaves
    new_leaves = jnp.where(bins <= split_point, left_leaf, right_leaf)

    # if order higher than 0, need to account fow lower incidental terms coming from
    # l * (x - split_point)^order transformed into a pure x basis (for predictions)
    order = indices[-2]

    incidental_leaves = jax.lax.switch(
        order,
        [
            lambda: jnp.zeros((4, bins.shape[0])),  # order 0, no incidental terms
            lambda: _linear_incidental_terms(
                left_leaf,
                right_leaf,
                indices,
                bin_edges,
                bins,
            ),
            lambda: _square_incidental_terms(
                left_leaf,
                right_leaf,
                indices,
                bin_edges,
                bins,
            ),
            lambda: _cubic_incidental_terms(
                left_leaf,
                right_leaf,
                indices,
                bin_edges,
                bins,
            ),
        ],
    )

    new_leaves_full = incidental_leaves.at[order, :].set(new_leaves)

    ensembles = ensembles.at[*leading_indices, :, :].add(new_leaves_full[:n_order, :])

    return ensembles, bin_edges


def _linear_incidental_terms(
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    indices: tuple,
    split_points: jnp.ndarray,
    bins: jnp.ndarray,
) -> jnp.ndarray:
    split_point = indices[-1]
    feature = indices[-3]
    s = split_points[split_point, feature]

    # constant value adjustment due to (x - split_point) transformation
    new_leaves = jnp.where(bins <= split_point, left_leaf * (-s), right_leaf * (-s))

    return jnp.concatenate(
        [new_leaves.reshape(1, -1), jnp.zeros((3, bins.shape[0]))], axis=0
    )


def _square_incidental_terms(
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    indices: tuple,
    split_points: jnp.ndarray,
    bins: jnp.ndarray,
) -> jnp.ndarray:
    split_point = indices[-1]
    feature = indices[-3]
    s = split_points[split_point, feature]

    # linear value adjustment due to (x - split_point)^2 transformation
    new_leaves_l = jnp.where(
        bins <= split_point, left_leaf * (-2 * s), right_leaf * (-2 * s)
    )

    # constant value adjustment due to (x - split_point)^2 transformation
    new_leaves_c = jnp.where(
        bins <= split_point, left_leaf * (s**2), right_leaf * (s**2)
    )

    return jnp.concatenate(
        [
            new_leaves_c.reshape(1, -1),
            new_leaves_l.reshape(1, -1),
            jnp.zeros((2, bins.shape[0])),
        ],
        axis=0,
    )


def _cubic_incidental_terms(
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    indices: tuple,
    split_points: jnp.ndarray,
    bins: jnp.ndarray,
) -> jnp.ndarray:
    split_point = indices[-1]
    feature = indices[-3]
    s = split_points[split_point, feature]

    # square value adjustment due to (x - split_point)^3 transformation
    new_leaves_s = jnp.where(
        bins <= split_point, left_leaf * (-3 * s), right_leaf * (-3 * s)
    )

    # linear value adjustment due to (x - split_point)^3 transformation
    new_leaves_l = jnp.where(
        bins <= split_point, left_leaf * (3 * s**2), right_leaf * (3 * s**2)
    )

    # constant value adjustment due to (x - split_point)^3 transformation
    new_leaves_c = jnp.where(
        bins <= split_point, left_leaf * (-(s**3)), right_leaf * (-(s**3))
    )
    return jnp.concatenate(
        [
            new_leaves_c.reshape(1, -1),
            new_leaves_l.reshape(1, -1),
            new_leaves_s.reshape(1, -1),
            jnp.zeros((1, bins.shape[0])),
        ],
        axis=0,
    )


def _update_bounds_monotonicity(
    ensembles: jnp.ndarray,
    indices: tuple,
    gain: jnp.ndarray,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    monotone_constraints: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    carry = (
        ensembles,
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
        bin_edges,
        mask_concat,
        monotone_constraints,
    )
    xs = (indices, gain)

    def update_step(carry, x):
        (
            ensembles,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            bin_edges,
            mask_concat,
            monotone_constraints,
        ) = carry
        indices, gain = x

        lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right = (
            jax.lax.cond(
                gain > 0,
                lambda: _update_bounds_positive_gain(
                    ensembles,
                    indices,
                    lower_bound_left,
                    upper_bound_left,
                    lower_bound_right,
                    upper_bound_right,
                    bin_edges,
                    mask_concat,
                    monotone_constraints,
                ),
                lambda: (
                    lower_bound_left,
                    upper_bound_left,
                    lower_bound_right,
                    upper_bound_right,
                ),
            )
        )

        return (
            ensembles,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            bin_edges,
            mask_concat,
            monotone_constraints,
        ), None

    carry, _ = jax.lax.scan(update_step, carry, xs)
    (
        _,
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
        _,
        _,
        _,
    ) = carry
    return (
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
    )


def _update_bounds_positive_gain(
    ensembles: jnp.ndarray,
    indices: tuple,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    monotone_constraints: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    feature = indices[1]
    mono_cons = monotone_constraints[feature]

    lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right = (
        jax.lax.cond(
            mono_cons == 0,
            lambda: (
                lower_bound_left,
                upper_bound_left,
                lower_bound_right,
                upper_bound_right,
            ),
            lambda: _check_max_order(
                ensembles,
                indices,
                lower_bound_left,
                upper_bound_left,
                lower_bound_right,
                upper_bound_right,
                bin_edges,
                mask_concat,
                mono_cons,
            ),
        )
    )
    return (
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
    )


def _check_max_order(
    ensembles: jnp.ndarray,
    indices: tuple,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    mono_cons: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    max_order = ensembles.shape[2]

    return jax.lax.cond(
        max_order == 1,  # no need to update if max order is constant leaf
        lambda: (
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
        ),
        lambda: _check_split_order(
            ensembles,
            indices,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            bin_edges,
            mask_concat,
            mono_cons,
        ),
    )


def _check_split_order(
    ensembles: jnp.ndarray,
    indices: tuple,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    mono_cons: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    order = indices[-2]

    return jax.lax.cond(
        order == 0,  # no need to update if split on constant leaf
        lambda: (
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
        ),
        lambda: _compute_update(
            ensembles,
            indices,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            bin_edges,
            mask_concat,
            mono_cons,
        ),
    )


def _compute_update(
    ensembles: jnp.ndarray,
    indices: tuple,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    mono_cons: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # gather leaves for current class and feature
    leaves = ensembles[*indices[:-2], :, :]  # n_orders-1 x n_bins
    n_order = leaves.shape[0]

    # gather feature
    feature = indices[1]

    # repeat leaves because 3 possible maxima/minima per bin
    leaves_concat = jnp.concatenate([leaves] * 3, axis=1).reshape(n_order, -1)

    # filter bin edges to not account for unfeasible bins (bin edges infitinity)
    bin_edges_filtered = jnp.where(
        bin_edges[:, feature] < 1, bin_edges[:, feature], 1.0
    )

    # get all three possible maxima/minima per bin
    first_bins = jnp.concatenate([jnp.zeros((1,)), bin_edges_filtered])
    second_bins = jnp.concatenate([bin_edges_filtered, jnp.ones((1,))])
    x_minima = jnp.where(
        n_order < 3,
        first_bins,
        jnp.where(
            leaves[2, :] == 0,
            first_bins,
            jnp.where(
                leaves[3, :] == 0,
                first_bins,
                jnp.where(
                    -leaves[2, :] / (3 * leaves[3, :]) > second_bins,
                    first_bins,
                    jnp.maximum(-leaves[2, :] / (3 * leaves[3, :]), first_bins),
                ),
            ),
        ),
    )
    x_concat = jnp.concatenate([first_bins, x_minima, second_bins], axis=0).reshape(
        1, -1
    )
    order_vec = jnp.arange(n_order).reshape(-1, 1)
    coeffs_vec = order_vec

    # evaluate polynomial at all possible maxima/minima
    y_concat = coeffs_vec * leaves_concat * (x_concat ** (order_vec - 1))
    y_concat = jnp.nan_to_num(y_concat, nan=0.0).sum(axis=0)
    y_concat = jnp.where(
        y_concat * mono_cons < 0, 0.0, y_concat
    )  # to get rid of numerical issues

    # compute denominators and numerators for bounds update
    bin_edges_concat = bin_edges_filtered.reshape(-1, 1)
    denominators = (x_concat - bin_edges_concat)[None, :, :] ** (order_vec - 1).reshape(
        -1, 1, 1
    )
    denominators = jnp.nan_to_num(denominators, nan=0.0)

    num_coeffs_vec = jnp.array([0, 0, 2, 1]).reshape(-1, 1)
    mult_coeffs_vec = jnp.array([0, 0, 1, 3]).reshape(-1, 1)
    exp_coeffs_vec = jnp.array([0, 0, 1, 0]).reshape(-1, 1)
    num_coeffs_vec2 = jnp.array([0, 0, 2, 0]).reshape(-1, 1)
    numerators = num_coeffs_vec[:n_order] * mult_coeffs_vec[n_order - 1] * (
        leaves_concat[n_order - 1, :]
        * x_concat ** (exp_coeffs_vec[:n_order] * (n_order - 3))
    ) + num_coeffs_vec2[:n_order] * leaves_concat[n_order - 2, :] * (n_order - 3)
    numerators = jnp.nan_to_num(numerators, nan=0.0)

    left_sign = 2 * (jnp.arange(n_order) % 2) - 1  # 1 if even order, -1 if odd order

    (
        new_lower_bound_left,
        new_upper_bound_left,
        new_lower_bound_right,
        new_upper_bound_right,
    ) = _update_bounds_with_mask(
        y_concat[None, :],
        mask_concat,
        denominators,
        numerators,
        mono_cons,
        left_sign,
    )

    new_lower_bound_left = jnp.where(
        lower_bound_left[*indices[:-2]] * left_sign[:, None] == jnp.inf,
        left_sign[:, None] * jnp.inf,
        new_lower_bound_left,
    )
    new_upper_bound_left = jnp.where(
        upper_bound_left[*indices[:-2]] * left_sign[:, None] == -jnp.inf,
        -left_sign[:, None] * jnp.inf,
        new_upper_bound_left,
    )
    new_lower_bound_right = jnp.where(
        lower_bound_right[*indices[:-2]] == jnp.inf, jnp.inf, new_lower_bound_right
    )
    new_upper_bound_right = jnp.where(
        upper_bound_right[*indices[:-2]] == -jnp.inf, -jnp.inf, new_upper_bound_right
    )
    lower_bound_left = lower_bound_left.at[*indices[:-2], 1:, :].set(
        new_lower_bound_left[1:, :]
    )
    upper_bound_left = upper_bound_left.at[*indices[:-2], 1:, :].set(
        new_upper_bound_left[1:, :]
    )
    lower_bound_right = lower_bound_right.at[*indices[:-2], 1:, :].set(
        new_lower_bound_right[1:, :]
    )
    upper_bound_right = upper_bound_right.at[*indices[:-2], 1:, :].set(
        new_upper_bound_right[1:, :]
    )

    return (
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
    )


@partial(
    vmap, in_axes=(None, None, 0, 0, None, 0), out_axes=(0, 0, 0, 0)
)  # vectorising over order axis
def _update_bounds_with_mask(
    y_concat: jnp.ndarray,
    mask_concat: jnp.ndarray,
    denominators: jnp.ndarray,
    numerators: jnp.ndarray,
    mono_cons: jnp.ndarray,
    left_sign: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    lower_bound_left = -left_sign * jnp.min(
        jnp.where(
            denominators == 0,
            jnp.where(
                y_concat * mask_concat == 0,
                left_sign * numerators,
                jnp.inf,
            ),
            left_sign * y_concat * mask_concat / denominators,
        ),
        axis=1,
        initial=jnp.inf,
        where=mask_concat & (mono_cons == 1),
    )

    upper_bound_left = -left_sign * jnp.max(
        jnp.where(
            denominators == 0,
            jnp.where(
                y_concat * mask_concat == 0,
                left_sign * numerators,
                -jnp.inf,
            ),
            left_sign * y_concat * mask_concat / denominators,
        ),
        axis=1,
        initial=-jnp.inf,
        where=mask_concat & (mono_cons == -1),
    )
    lower_bound_right = -jnp.min(
        jnp.where(
            denominators == 0,
            jnp.where(
                y_concat * ~mask_concat == 0,
                numerators,
                jnp.inf,
            ),
            y_concat * ~mask_concat / denominators,
        ),
        axis=1,
        initial=jnp.inf,
        where=~mask_concat & (mono_cons == 1),
    )
    upper_bound_right = -jnp.max(
        jnp.where(
            denominators == 0,
            jnp.where(
                y_concat * ~mask_concat == 0,
                numerators,
                -jnp.inf,
            ),
            y_concat * ~mask_concat / denominators,
        ),
        axis=1,
        initial=-jnp.inf,
        where=~mask_concat & (mono_cons == -1),
    )

    return (
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
    )


def _update_bounds_curvature(
    ensembles: jnp.ndarray,
    indices: tuple,
    gain: float,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    curvature_constraints: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    carry = (
        ensembles,
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
        bin_edges,
        mask_concat,
        curvature_constraints,
    )
    xs = (indices, gain)

    def update_step(carry, x):
        (
            ensembles,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            bin_edges,
            mask_concat,
            curvature_constraints,
        ) = carry
        indices, gain = x

        lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right = (
            jax.lax.cond(
                gain > 0,
                lambda: _update_bounds_positive_gain_c(
                    ensembles,
                    indices,
                    lower_bound_left,
                    upper_bound_left,
                    lower_bound_right,
                    upper_bound_right,
                    bin_edges,
                    mask_concat,
                    curvature_constraints,
                ),
                lambda: (
                    lower_bound_left,
                    upper_bound_left,
                    lower_bound_right,
                    upper_bound_right,
                ),
            )
        )

        return (
            ensembles,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            bin_edges,
            mask_concat,
            curvature_constraints,
        ), None

    carry, _ = jax.lax.scan(update_step, carry, xs)
    (
        _,
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
        _,
        _,
        _,
    ) = carry
    return (
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
    )


def _update_bounds_positive_gain_c(
    ensembles: jnp.ndarray,
    indices: tuple,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    curvature_constraints: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    feature = indices[1]
    curv_cons = curvature_constraints[feature]

    lower_bound_left, upper_bound_left, lower_bound_right, upper_bound_right = (
        jax.lax.cond(
            curv_cons == 0,
            lambda: (
                lower_bound_left,
                upper_bound_left,
                lower_bound_right,
                upper_bound_right,
            ),
            lambda: _check_max_order_c(
                ensembles,
                indices,
                lower_bound_left,
                upper_bound_left,
                lower_bound_right,
                upper_bound_right,
                bin_edges,
                mask_concat,
                curv_cons,
            ),
        )
    )
    return (
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
    )


def _check_max_order_c(
    ensembles: jnp.ndarray,
    indices: tuple,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    curv_cons: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    max_order = ensembles.shape[2]

    return jax.lax.cond(
        max_order < 3,  # no need to update if max order is constant or linear leaf
        lambda: (
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
        ),
        lambda: _check_split_order_c(
            ensembles,
            indices,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            bin_edges,
            mask_concat,
            curv_cons,
        ),
    )


def _check_split_order_c(
    ensembles: jnp.ndarray,
    indices: tuple,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    curv_cons: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    order = indices[-2]

    return jax.lax.cond(
        order < 2,  # no need to update if split on constant or linear leaf
        lambda: (
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
        ),
        lambda: _compute_update_c(
            ensembles,
            indices,
            lower_bound_left,
            upper_bound_left,
            lower_bound_right,
            upper_bound_right,
            bin_edges,
            mask_concat,
            curv_cons,
        ),
    )


def _compute_update_c(
    ensembles: jnp.ndarray,
    indices: tuple,
    lower_bound_left: jnp.ndarray,
    upper_bound_left: jnp.ndarray,
    lower_bound_right: jnp.ndarray,
    upper_bound_right: jnp.ndarray,
    bin_edges: jnp.ndarray,
    mask_concat: jnp.ndarray,
    curv_cons: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # gather leaves for current class and feature
    leaves = ensembles[*indices[:-2], :, :]  # n_orders x n_bins
    n_order = leaves.shape[0]
    # gather feature
    feature = indices[1]

    # repeat leaves because 2 possible maxima/minima per bin
    leaves_concat = jnp.concatenate([leaves] * 2, axis=1).reshape(n_order, -1)

    # filter bin edges to not account for unfeasible bins (bin edges infitinity)
    bin_edges_filtered = jnp.where(
        bin_edges[:, feature] < 1, bin_edges[:, feature], 1.0
    )

    # get all three possible maxima/minima per bin
    first_bins = jnp.concatenate([jnp.zeros((1,)), bin_edges_filtered])
    second_bins = jnp.concatenate([bin_edges_filtered, jnp.ones((1,))])
    x_concat = jnp.concatenate([first_bins, second_bins], axis=0).reshape(1, -1)
    order_vec = jnp.arange(n_order).reshape(-1, 1)
    coeffs_vec = jnp.array([0, 0, 2, 6]).reshape(-1, 1)[:n_order, :]

    # evaluate polynomial at all possible maxima/minima
    y_concat = coeffs_vec * leaves_concat * x_concat ** (order_vec - 2)
    y_concat = jnp.nan_to_num(y_concat, nan=0.0).sum(axis=0)
    y_concat = jnp.where(
        y_concat * curv_cons < 0, 0.0, y_concat
    )  # to get rid of numerical issues

    # compute denominators and numerators for bounds update
    bin_edges_concat = bin_edges_filtered.reshape(-1, 1)
    denominators = (x_concat - bin_edges_concat)[None, :, :] ** (order_vec - 2).reshape(
        -1, 1, 1
    )
    denominators = jnp.nan_to_num(denominators, nan=0.0)

    num_coeffs_vec = jnp.array([0, 0, 0, 6]).reshape(-1, 1)
    numerators = num_coeffs_vec[:n_order] * leaves_concat[n_order - 1, :]
    numerators = jnp.nan_to_num(numerators, nan=0.0)

    left_sign = (
        2 * ((jnp.arange(n_order) + 1) % 2) - 1
    )  # -1 if odd order, 1 if even order

    (
        new_lower_bound_left,
        new_upper_bound_left,
        new_lower_bound_right,
        new_upper_bound_right,
    ) = _update_bounds_with_mask_c(
        y_concat[None, :], mask_concat, denominators, numerators, curv_cons, left_sign
    )

    new_lower_bound_left = jnp.where(
        lower_bound_left[*indices[:-2]] * left_sign[:, None] == jnp.inf,
        left_sign[:, None] * jnp.inf,
        new_lower_bound_left,
    )
    new_upper_bound_left = jnp.where(
        upper_bound_left[*indices[:-2]] * left_sign[:, None] == -jnp.inf,
        -left_sign[:, None] * jnp.inf,
        new_upper_bound_left,
    )
    new_lower_bound_right = jnp.where(
        lower_bound_right[*indices[:-2]] == jnp.inf, jnp.inf, new_lower_bound_right
    )
    new_upper_bound_right = jnp.where(
        upper_bound_right[*indices[:-2]] == -jnp.inf, -jnp.inf, new_upper_bound_right
    )

    lower_bound_left = lower_bound_left.at[*indices[:-2], 2:, :].set(
        new_lower_bound_left[2:, :]
    )
    upper_bound_left = upper_bound_left.at[*indices[:-2], 2:, :].set(
        new_upper_bound_left[2:, :]
    )
    lower_bound_right = lower_bound_right.at[*indices[:-2], 2:, :].set(
        new_lower_bound_right[2:, :]
    )
    upper_bound_right = upper_bound_right.at[*indices[:-2], 2:, :].set(
        new_upper_bound_right[2:, :]
    )

    return (
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
    )


@partial(
    vmap, in_axes=(None, None, 0, 0, None, 0), out_axes=(0, 0, 0, 0)
)  # vectorising over order axis
def _update_bounds_with_mask_c(
    y_concat: jnp.ndarray,
    mask_concat: jnp.ndarray,
    denominators: jnp.ndarray,
    numerators: jnp.ndarray,
    curv_cons: jnp.ndarray,
    left_sign: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    lower_bound_left = -left_sign * jnp.min(
        jnp.where(
            denominators == 0,
            jnp.where(
                y_concat * mask_concat == 0,
                left_sign * numerators,
                jnp.inf,
            ),
            (left_sign * y_concat * mask_concat) / denominators,
        ),
        axis=1,
        initial=jnp.inf,
        where=mask_concat & (curv_cons == 1),
    )

    upper_bound_left = -left_sign * jnp.max(
        jnp.where(
            denominators == 0,
            jnp.where(
                y_concat * mask_concat == 0,
                left_sign * numerators,
                -jnp.inf,
            ),
            (left_sign * y_concat * mask_concat) / denominators,
        ),
        axis=1,
        initial=-jnp.inf,
        where=mask_concat & (curv_cons == -1),
    )
    lower_bound_right = -jnp.min(
        jnp.where(
            denominators == 0.0,
            jnp.where(
                y_concat * ~mask_concat == 0,
                numerators,
                jnp.inf,
            ),
            (y_concat * ~mask_concat) / denominators,
        ),
        axis=1,
        initial=jnp.inf,
        where=~mask_concat & (curv_cons == 1),
    )
    upper_bound_right = -jnp.max(
        jnp.where(
            denominators == 0,
            jnp.where(
                y_concat * ~mask_concat == 0,
                numerators,
                -jnp.inf,
            ),
            (y_concat * ~mask_concat) / denominators,
        ),
        axis=1,
        initial=-jnp.inf,
        where=~mask_concat & (curv_cons == -1),
    )

    return (
        lower_bound_left,
        upper_bound_left,
        lower_bound_right,
        upper_bound_right,
    )


def _update_preds(
    inner_preds: jnp.ndarray,
    inner_preds_val: jnp.ndarray,
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    indices: tuple,
    data_bin_indices: jnp.ndarray,
    data_bin_indices_val: jnp.ndarray,
    data: jnp.ndarray,
    data_val: jnp.ndarray,
    gain: jnp.ndarray,
    bin_edges: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    carry = (
        inner_preds,
        inner_preds_val,
        data_bin_indices,
        data_bin_indices_val,
        data,
        data_val,
        bin_edges,
    )
    xs = (indices, left_leaf, right_leaf, gain)

    def update_step(carry, x):
        (
            inner_preds,
            inner_preds_val,
            data_bin_indices,
            data_bin_indices_val,
            data,
            data_val,
            bin_edges,
        ) = carry
        indices, left_leaf, right_leaf, gain = x

        inner_preds, inner_preds_val = jax.lax.cond(
            gain > 0,
            lambda: _update_preds_positive_gain(
                inner_preds,
                inner_preds_val,
                left_leaf,
                right_leaf,
                indices,
                data_bin_indices,
                data_bin_indices_val,
                data,
                data_val,
                bin_edges,
            ),
            lambda: (inner_preds, inner_preds_val),
        )

        return (
            inner_preds,
            inner_preds_val,
            data_bin_indices,
            data_bin_indices_val,
            data,
            data_val,
            bin_edges,
        ), None

    carry, _ = jax.lax.scan(update_step, carry, xs)
    inner_preds, inner_preds_val, _, _, _, _, _ = carry
    return inner_preds, inner_preds_val


def _update_preds_positive_gain(
    inner_preds: jnp.ndarray,
    inner_preds_val: jnp.ndarray,
    left_leaf: jnp.ndarray,
    right_leaf: jnp.ndarray,
    indices: tuple,
    data_bin_indices: jnp.ndarray,
    data_bin_indices_val: jnp.ndarray,
    data: jnp.ndarray,
    data_val: jnp.ndarray,
    bin_edges: jnp.ndarray,
)-> tuple[jnp.ndarray, jnp.ndarray]:
    split_point = indices[-1]
    order = indices[-2]
    feature = indices[-3]
    class_ = indices[-4]

    # gather data indices for the correct feature
    data_indices = data_bin_indices[:, feature]
    data_indices_val = data_bin_indices_val[:, feature]

    inner_preds = inner_preds.at[:, class_].add(
        jnp.where(
            data_indices <= split_point,
            left_leaf * (data[:, feature] - bin_edges[split_point, feature]) ** order,
            right_leaf * (data[:, feature] - bin_edges[split_point, feature]) ** order,
        )
    )
    # can be optimised by not repeting when training and eval sets are the same
    inner_preds_val = inner_preds_val.at[:, class_].add(
        jnp.where(
            data_indices_val <= split_point,
            left_leaf
            * (data_val[:, feature] - bin_edges[split_point, feature]) ** order,
            right_leaf
            * (data_val[:, feature] - bin_edges[split_point, feature]) ** order,
        )
    )
    return inner_preds, inner_preds_val


def _update_feature_importance(
    feature_importance_dict: dict,
    best_gain: jnp.ndarray,
    indices: tuple,
)-> dict:
    carry = feature_importance_dict
    xs = (best_gain, indices)

    def update_step(carry, x):
        feature_importance_dict = carry
        best_gain, indices = x
        class_ = indices[0]
        feature = indices[1]
        order = indices[2]
        feature_importance_dict["gain"] = (
            feature_importance_dict["gain"]
            .at[class_, feature, order]
            .add(jnp.maximum(best_gain, 0))
        )
        feature_importance_dict["num_splits"] = (
            feature_importance_dict["num_splits"]
            .at[class_, feature, order]
            .add(jnp.where(best_gain > 0, 1, 0))
        )
        return feature_importance_dict, None

    carry, _ = jax.lax.scan(update_step, carry, xs)
    feature_importance_dict = carry
    return feature_importance_dict
