"""
Trivial baselines for Stage 1.

Both wrap sklearn's DummyClassifier rather than reimplementing the logic,
so every model in Stage 1 (trivial and LR) shares the same fit /
predict_proba interface and can be evaluated with identical code.

Prevalence baseline: predicts a constant probability equal to the TRAIN
FOLD's target prevalence for every row (strategy="prior"). Never computed
from validation data — DummyClassifier.fit() only looks at y_train.

Majority-class baseline: always predicts the train-fold majority class
with a hard probability of 1.0 (strategy="most_frequent"). Included to
make explicit that F1/accuracy alone are not meaningful metrics here: with
~55% positive prevalence, always predicting "returned" already scores
F1 ≈ 0.71.
"""

from sklearn.dummy import DummyClassifier


def build_prevalence_baseline(random_state: int = 42) -> DummyClassifier:
    return DummyClassifier(strategy="prior", random_state=random_state)


def build_majority_baseline(random_state: int = 42) -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent", random_state=random_state)
