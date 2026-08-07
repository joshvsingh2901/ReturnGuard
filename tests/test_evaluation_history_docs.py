"""Regression guards for truthful Stage 3 evaluation-history language."""

from pathlib import Path


ROOT = Path(__file__).parent.parent


def test_evaluation_history_document_exists_and_discloses_test_inspection():
    text = (ROOT / "docs" / "evaluation-history.md").read_text().lower()
    assert "test-statistic inspection" in text
    assert "test model performance" in text
    assert "not pristine" in text


def test_readme_does_not_claim_literal_unopened_test_split():
    text = (ROOT / "README.md").read_text().lower()
    assert "still no test-split access" not in text
    assert "untouched test set" not in text


def test_stage2_doc_does_not_claim_primary_validation_evaluated_once():
    text = (ROOT / "docs" / "stage2-gbdt.md").read_text().lower()
    assert "evaluated **once** on `primary_val`" not in text
