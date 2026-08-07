"""
Logistic Regression baseline model for Stage 1.

Untuned, deliberately simple: L2 penalty, C=1.0, no class weighting
(prevalence is ~55/45, close enough to balanced that reweighting would
distort predicted probabilities — directly conflicting with the project's
eventual calibration goal), lbfgs solver. Hyperparameter tuning is
explicitly deferred to a later stage.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.features.preprocessing import build_preprocessor

LR_C = 1.0
LR_MAX_ITER = 1000
LR_SOLVER = "lbfgs"


def build_logistic_pipeline(include_price: bool) -> Pipeline:
    """Build the full preprocessing + Logistic Regression pipeline."""
    preprocessor = build_preprocessor(include_price=include_price)
    clf = LogisticRegression(
        penalty="l2",
        C=LR_C,
        class_weight=None,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf),
    ])


def check_convergence(pipeline: Pipeline) -> None:
    """
    Raise if the fitted LogisticRegression did not converge within
    max_iter. sklearn only warns on non-convergence by default; Stage 1
    treats an unconverged fit as a hard failure rather than a silent
    partial result.
    """
    clf = pipeline.named_steps["clf"]
    n_iter = clf.n_iter_
    n_iter_max = int(n_iter.max()) if hasattr(n_iter, "max") else int(n_iter)
    if n_iter_max >= clf.max_iter:
        raise RuntimeError(
            f"LogisticRegression did not converge: n_iter_={n_iter_max} "
            f">= max_iter={clf.max_iter}. Increase max_iter or investigate "
            "the input features."
        )
