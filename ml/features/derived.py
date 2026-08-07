"""
Derived product-side features for Stage 2.

Only features a tree cannot trivially reconstruct itself are included —
with ~12 categories per field and 1M+ rows, XGBoost learns interactions
like brand x productType through ordinary depth, so no hand-built
interaction terms are added here (see docs/stage2-gbdt.md for the
reasoning).

avgDiscountValue units assumption: it is inferred, with high confidence,
to be a PERCENTAGE rather than a currency amount. Evidence (measured on
product_nodes_training.p): 38.06% of variants have discount > price,
which is nonsensical for an absolute discount; discount is tightly
clustered around a median of ~17.7 with an IQR of [16.4, 19.0], consistent
with a capped rate; and corr(price, discount) = 0.021, whereas an absolute
discount would be expected to scale with price. This is a documented
assumption, not a certainty — avgGbpPrice's pre- vs. post-discount status
is separately UNKNOWN and does not change the recommendation, since the
price x discount product is informative either way.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DerivedPriceFeatures(BaseEstimator, TransformerMixin):
    """
    Adds three numeric columns:

      discount_amount = avgGbpPrice * avgDiscountValue / 100
          Row-wise; fits no statistics.

      price_rel_type_median = avgGbpPrice / median(avgGbpPrice | productType)
          Group medians learned from fit(X) only (the training fold).

      price_rel_brand_median = avgGbpPrice / median(avgGbpPrice | brandDesc)
          Group medians learned from fit(X) only (the training fold).

    A category present at transform() time but absent (or all-NaN-price)
    at fit() time falls back to the train-fold GLOBAL median price. Rows
    with a missing avgGbpPrice (no product node) propagate NaN through
    every derived feature here — this transformer does not impute.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "DerivedPriceFeatures":
        self.type_medians_ = X.groupby("productType")["avgGbpPrice"].median()
        self.brand_medians_ = X.groupby("brandDesc")["avgGbpPrice"].median()
        self.global_median_ = X["avgGbpPrice"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["discount_amount"] = X["avgGbpPrice"] * X["avgDiscountValue"] / 100.0

        type_med = X["productType"].map(self.type_medians_).fillna(self.global_median_)
        brand_med = X["brandDesc"].map(self.brand_medians_).fillna(self.global_median_)

        X["price_rel_type_median"] = X["avgGbpPrice"] / type_med
        X["price_rel_brand_median"] = X["avgGbpPrice"] / brand_med
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + [
            "discount_amount", "price_rel_type_median", "price_rel_brand_median",
        ]
