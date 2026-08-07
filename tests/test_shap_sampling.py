"""Fast tests for deterministic Stage 3 explanation sampling."""

from ml.explainability.sampling import representative_validation_sample


def test_representative_sample_is_deterministic_and_has_manifest(synthetic_df):
    train_products = set(synthetic_df["hash(variantID)"].unique()[:50])
    first, first_manifest = representative_validation_sample(
        synthetic_df, train_products, n=400, seed=7
    )
    second, second_manifest = representative_validation_sample(
        synthetic_df, train_products, n=400, seed=7
    )
    assert first.index.tolist() == second.index.tolist()
    assert first_manifest.manifest_hash == second_manifest.manifest_hash
    assert first_manifest.selected_n == 400
    assert sum(first_manifest.stratum_counts.values()) == 400


def test_representative_sample_is_row_order_invariant(synthetic_df):
    train_products = set(synthetic_df["hash(variantID)"].unique()[:50])
    original, _ = representative_validation_sample(synthetic_df, train_products, n=400, seed=19)
    shuffled, _ = representative_validation_sample(
        synthetic_df.sample(frac=1.0, random_state=3), train_products, n=400, seed=19
    )
    assert original.index.tolist() == shuffled.index.tolist()


def test_same_sample_can_be_reused_for_a3_and_a4(synthetic_df):
    train_products = set(synthetic_df["hash(variantID)"].unique()[:50])
    sample, manifest = representative_validation_sample(synthetic_df, train_products, n=250)
    assert sample.index.tolist() == manifest.index_values
