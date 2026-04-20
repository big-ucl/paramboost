import os
import sys
import time
import copy

sys.path.append(os.getcwd())
# sys.path.append(os.path.join(os.getcwd(), "paramboost"))

from experiment.models.paramb import ParamB
from sklearn.metrics import log_loss as cross_entropy
from case_study.utils import (
    load_preprocess_return_constraints_LPMC,
    plot_features_with_ci,
    plot_features
)

import pandas as pd
import numpy as np

(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    structure,
    monotone_constraints,
    curvature_constraints,
    hyperparameters,
) = load_preprocess_return_constraints_LPMC()


def create_and_train_model(
    X_train, y_train, X_val, y_val, X_test, y_test, model_name, file_name, **kwargs
):
    model = ParamB(X_train, y_train, **kwargs)

    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    end_time = time.time() - start_time

    y_train_preds = model.predict(X_train)
    y_val_preds = model.predict(X_val)
    y_test_preds = model.predict(X_test)

    train_score = cross_entropy(y_train, y_train_preds)
    val_score = cross_entropy(y_val, y_val_preds)
    test_score = cross_entropy(y_test, y_test_preds)

    best_iter = model.model.best_iteration

    current_score_df = pd.DataFrame(
        {
            "train_score": train_score,
            "val_score": val_score,
            "test_score": test_score,
            "time": end_time,
            "best_iter": best_iter,
        },
        index=[model_name],
    )
    current_score_df.to_csv(
        f"case_study/results_{file_name}.csv",
        sep=",",
        index=True,
    )

    return model


hyperparameters_normal = copy.deepcopy(hyperparameters)
hyperparameters_normal["continuity"] = -1
model_normal = create_and_train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    "ParamB-$O^3-C^{-1}$",
    "normal",
    **hyperparameters_normal,
)


model_continuous = create_and_train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    "ParamB-$O^3-C^2$",
    "continuous",
    **hyperparameters,
)

model_class_spec = create_and_train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    "ParamB-$O^3-C^2$-Class_spe",
    "class_spe",
    structure=structure,
    **hyperparameters,
)

model_monotonic = create_and_train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    "ParamB-$O^3-C^2$-monotonic",
    "monotonic",
    structure=structure,
    monotone_constraints=monotone_constraints,
    **hyperparameters,
)

model_curvature = create_and_train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    "ParamB-$O^3-C^2$-convex",
    "convex",
    structure=structure,
    monotone_constraints=monotone_constraints,
    curvature_constraints=curvature_constraints,
    **hyperparameters,
)


plot_features(
    [model_normal, model_continuous, model_class_spec, model_monotonic, model_curvature],
    ["ParamB-$O^3-C^{-1}$", "ParamB-$O^3-C^2$", "ParamB-$O^3-C^2$-Class_spe", "ParamB-$O^3-C^2$-monotonic", "ParamB-$O^3-C^2$-curvature"],
    X_train,
    [(0, "dur_walking"), (1, "dur_cycling"), (2, "dur_pt_rail"), (3, "dur_driving")],
    save_name="shape_function_no_constraints_",
)

# plot some features
plot_features_with_ci(
    model_curvature,
    X_train,
    [(0, "dur_walking"), (1, "dur_cycling"), (2, "dur_pt_rail"), (3, "dur_driving")],
    save_name="shape_function_with_constraints_",
)
