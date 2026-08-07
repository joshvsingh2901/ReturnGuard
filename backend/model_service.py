"""Serving adapter for the governed, frozen ReturnGuard A3 artifact.

This module deliberately builds only a raw event frame and delegates every
model transformation to the fitted MLflow artifact.  It contains no training,
feature fitting, or model-selection logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.config import DEFAULT_MODEL_URI, ROOT, Settings
from backend.errors import ExplanationFailedError, InferenceFailedError, ModelUnavailableError
from backend.schemas import (
    MISSING_CATEGORY_TOKEN,
    OBSERVED_DISCOUNT_MAX,
    OBSERVED_PRICE_RANGE,
    OBSERVED_YEAR_RANGE,
    DataContext,
    ModelInfoResponse,
    ModelProvenance,
    PredictionRequest,
)
from ml.data.schema import EVENT_CUST_COL
from ml.explainability.contracts import (
    METHODOLOGY_NOTE,
    build_local_explanation,
    user_facing_explanation,
)
from ml.explainability.shap_analysis import build_tree_explainer, compute_tree_shap
from ml.lifecycle.reproducibility import MODEL_VERSION, validate_fitted_feature_manifest
from ml.lifecycle.tracking import REGISTERED_MODEL_NAME

LOGGER = logging.getLogger(__name__)
LIFECYCLE_MANIFEST_PATH = ROOT / "artifacts" / "manifests" / f"{MODEL_VERSION}.json"
GOVERNANCE_STATUS = "model_of_record"
_RAW_INPUT_COLUMNS = (
    EVENT_CUST_COL,
    "yearOfBirth",
    "isMale",
    "shippingCountry",
    "premier",
    "productType",
    "brandDesc",
    "avgGbpPrice",
    "avgDiscountValue",
)
_CATEGORY_NAMES = ("shippingCountry", "productType", "brandDesc")


@dataclass(frozen=True)
class ServingProvenance:
    """Safe, validated provenance retained for the lifetime of the process."""

    semantic_version: str
    registry_model: str
    registry_version: str
    mlflow_source_run_id: str
    feature_manifest_hash: str
    dataset_version: str
    governance_status: str

    def concise(self) -> ModelProvenance:
        return ModelProvenance(
            semantic_version=self.semantic_version,
            registry_model=self.registry_model,
            registry_version=self.registry_version,
            feature_manifest_hash=self.feature_manifest_hash,
        )

    def public_model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            semantic_version=self.semantic_version,
            registry_model=self.registry_model,
            registry_version=self.registry_version,
            mlflow_source_run_id=self.mlflow_source_run_id,
            feature_manifest_hash=self.feature_manifest_hash,
            dataset_version=self.dataset_version,
            governance_status=self.governance_status,
            intended_use=(
                "Rank item-purchase return risk within the returner-enriched "
                "ASOS research population."
            ),
            limitations=[
                "Scores are dataset-conditional and are not merchant-wide calibrated return probabilities.",
                "Missing customer profiles use a stable synthetic donor profile whose demographics are not exposed.",
                "Product-missing events had materially weaker evaluation performance.",
            ],
            population_shift_warning=(
                "Use with merchant populations or changing catalogues requires separate validation and calibration."
            ),
        )


class ModelService:
    """One loaded A3 pyfunc model, its fitted composite artifact, and SHAP state."""

    def __init__(
        self,
        *,
        pyfunc_model: Any,
        composite_artifact: Any,
        provenance: ServingProvenance,
        explanations_enabled: bool,
        explainer: Optional[Any],
    ) -> None:
        self._pyfunc_model = pyfunc_model
        self._composite_artifact = composite_artifact
        self.provenance = provenance
        self._explanations_enabled = explanations_enabled
        self._explainer = explainer
        self._category_values = self._read_known_categories(composite_artifact)

    @classmethod
    def load(cls, settings: Settings) -> "ModelService":
        """Load and fail-closed validate the local model-of-record once."""
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_registry_uri(settings.mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient(
                tracking_uri=settings.mlflow_tracking_uri,
                registry_uri=settings.mlflow_tracking_uri,
            )
            registry_version = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME, "model-of-record"
            )
            cls._validate_registry_version(registry_version)
            cls._validate_model_uri(settings.model_uri, str(registry_version.version))

            pyfunc_model = mlflow.pyfunc.load_model(settings.model_uri)
            composite_artifact = cls._extract_composite_artifact(pyfunc_model)
            lifecycle_manifest = cls._load_lifecycle_manifest()
            cls._validate_loaded_artifact(
                pyfunc_model=pyfunc_model,
                composite_artifact=composite_artifact,
                lifecycle_manifest=lifecycle_manifest,
            )

            provenance = ServingProvenance(
                semantic_version=MODEL_VERSION,
                registry_model=REGISTERED_MODEL_NAME,
                registry_version=str(registry_version.version),
                mlflow_source_run_id=cls._source_run_id(registry_version),
                feature_manifest_hash=lifecycle_manifest["feature_manifest_hash"],
                dataset_version=lifecycle_manifest["dataset_version"],
                governance_status=GOVERNANCE_STATUS,
            )
            explainer = build_tree_explainer(composite_artifact.model) if settings.explanations_enabled else None
            service = cls(
                pyfunc_model=pyfunc_model,
                composite_artifact=composite_artifact,
                provenance=provenance,
                explanations_enabled=settings.explanations_enabled,
                explainer=explainer,
            )
            service._startup_smoke_prediction()
            LOGGER.info(
                "event=model_loaded semantic_version=%s registry_version=%s",
                provenance.semantic_version,
                provenance.registry_version,
            )
            return service
        except Exception as exc:
            LOGGER.exception("event=model_load_failed error_code=model_unavailable")
            raise ModelUnavailableError("The governed model could not be loaded") from exc

    @staticmethod
    def _validate_model_uri(model_uri: str, model_of_record_version: str) -> None:
        """Permit only the alias or its exact resolved model-of-record version."""
        version_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_of_record_version}"
        if model_uri not in {DEFAULT_MODEL_URI, version_uri}:
            raise ValueError("Configured model URI does not resolve to the governed model-of-record")

    @staticmethod
    def _validate_registry_version(registry_version: Any) -> None:
        tags = dict(getattr(registry_version, "tags", {}) or {})
        if tags.get("semantic_model_version") != MODEL_VERSION:
            raise ValueError("Registry model semantic version is not the frozen A3 version")
        if tags.get("governance_status") != GOVERNANCE_STATUS:
            raise ValueError("Registry model alias is not explicitly marked model_of_record")

    @staticmethod
    def _source_run_id(registry_version: Any) -> str:
        tags = dict(getattr(registry_version, "tags", {}) or {})
        source_run_id = tags.get("source_run_id") or getattr(registry_version, "run_id", None)
        if not source_run_id:
            raise ValueError("Registry model version does not expose a source run")
        return str(source_run_id)

    @staticmethod
    def _extract_composite_artifact(pyfunc_model: Any) -> Any:
        try:
            python_model = pyfunc_model.unwrap_python_model()
            composite_artifact = python_model._artifact
        except (AttributeError, TypeError) as exc:
            raise ValueError("MLflow model does not contain the governed A3 composite artifact") from exc
        if not hasattr(composite_artifact, "feature_pipeline") or not hasattr(composite_artifact, "model"):
            raise ValueError("Loaded MLflow artifact is not a governed A3 composite")
        return composite_artifact

    @staticmethod
    def _load_lifecycle_manifest() -> dict[str, Any]:
        manifest = json.loads(LIFECYCLE_MANIFEST_PATH.read_text())
        if manifest.get("semantic_model_version") != MODEL_VERSION:
            raise ValueError("Lifecycle manifest semantic version does not match frozen A3")
        if manifest.get("governance_status") != "frozen":
            raise ValueError("Lifecycle manifest is not frozen")
        return manifest

    @staticmethod
    def _validate_loaded_artifact(
        *, pyfunc_model: Any, composite_artifact: Any, lifecycle_manifest: dict[str, Any]
    ) -> None:
        metadata = dict(getattr(getattr(pyfunc_model, "metadata", None), "metadata", {}) or {})
        if metadata.get("semantic_model_version") != MODEL_VERSION:
            raise ValueError("MLflow model metadata lacks the frozen semantic version")
        if metadata.get("feature_manifest_hash") != lifecycle_manifest["feature_manifest_hash"]:
            raise ValueError("MLflow model feature manifest hash does not match the frozen manifest")
        if metadata.get("freeze_spec_hash") != lifecycle_manifest["freeze_spec_hash"]:
            raise ValueError("MLflow model freeze specification hash does not match the frozen manifest")
        if getattr(composite_artifact, "model_version", None) != MODEL_VERSION:
            raise ValueError("Composite artifact semantic version does not match frozen A3")
        fitted_manifest = validate_fitted_feature_manifest(
            list(composite_artifact.feature_names), lifecycle_manifest
        )
        if fitted_manifest["manifest_hash"] != lifecycle_manifest["feature_manifest_hash"]:
            raise ValueError("Composite feature manifest validation failed")
        expected_steps = {"donor_impute", "derived", "preprocess"}
        actual_steps = set(composite_artifact.feature_pipeline.named_steps)
        if actual_steps != expected_steps:
            raise ValueError("Composite preprocessing steps differ from frozen A3")

    @staticmethod
    def _read_known_categories(composite_artifact: Any) -> dict[str, set[str]]:
        preprocessor = composite_artifact.feature_pipeline.named_steps["preprocess"]
        customer_transformer = preprocessor.named_transformers_["cat_cust"]
        customer_encoder = (
            customer_transformer.named_steps["onehot"]
            if hasattr(customer_transformer, "named_steps")
            else customer_transformer
        )
        product_transformer = preprocessor.named_transformers_["cat_prod"]
        product_encoder = (
            product_transformer.named_steps["onehot"]
            if hasattr(product_transformer, "named_steps")
            else product_transformer
        )
        return {
            "shippingCountry": {str(value) for value in customer_encoder.categories_[0]},
            "productType": {str(value) for value in product_encoder.categories_[0]},
            "brandDesc": {str(value) for value in product_encoder.categories_[1]},
        }

    def _startup_smoke_prediction(self) -> None:
        smoke_request = PredictionRequest.model_validate(
            {
                "customer_context_key": "startup-smoke",
                "customer_profile": {
                    "yearOfBirth": 1988,
                    "isMale": False,
                    "shippingCountry": "Country_A",
                    "premier": True,
                },
                "product_profile": {
                    "productType": "Jeans",
                    "brandDesc": "Brand_A",
                    "avgGbpPrice": 54.99,
                    "avgDiscountValue": 15.0,
                },
            }
        )
        prediction = self._predict_probabilities(self._frame_for_requests([smoke_request])[0])
        if prediction.shape != (1,) or not np.isfinite(prediction[0]) or not 0 <= prediction[0] <= 1:
            raise ValueError("Synthetic startup prediction was not a valid probability")

    def predict(self, requests: list[PredictionRequest]) -> list[dict[str, Any]]:
        """Score one vectorized raw frame, preserving request order."""
        frame, contexts = self._frame_for_requests(requests)
        probabilities = self._predict_probabilities(frame)
        if len(probabilities) != len(requests):
            raise InferenceFailedError("The model returned an unexpected prediction count")
        return [
            self._prediction_payload(float(probability), context)
            for probability, context in zip(probabilities, contexts)
        ]

    def explain(self, request: PredictionRequest) -> dict[str, Any]:
        if not self._explanations_enabled or self._explainer is None:
            from backend.errors import ExplanationsDisabledError

            raise ExplanationsDisabledError("Explanation service is disabled")
        try:
            frame, contexts = self._frame_for_requests([request])
            transformed = self._composite_artifact.transform(frame)
            shap_result = compute_tree_shap(
                self._composite_artifact.model, transformed, explainer=self._explainer
            )
            internal = build_local_explanation(
                raw_probability=float(shap_result.probability[0]),
                raw_margin=float(shap_result.raw_margin[0]),
                base_value=shap_result.base_value,
                shap_row=shap_result.values[0],
                feature_names=list(self._composite_artifact.feature_names),
                row=self._explanation_row(frame.iloc[0]),
                customer_profile_imputed=contexts[0].customer_profile_imputed,
                product_profile_available=contexts[0].product_profile_available,
                model_version=self.provenance.semantic_version,
            )
            public = user_facing_explanation(internal)
            payload = self._prediction_payload(float(shap_result.probability[0]), contexts[0])
            payload["top_risk_factors"] = public["top_risk_factors"]
            payload["top_protective_factors"] = public["top_protective_factors"]
            payload["explanation_method"] = "tree_shap_grouped_noncausal"
            return payload
        except ExplanationFailedError:
            raise
        except Exception as exc:
            LOGGER.exception("event=explanation_failed error_code=explanation_failed")
            raise ExplanationFailedError("The explanation could not be generated") from exc

    @staticmethod
    def _explanation_row(row: pd.Series) -> pd.Series:
        """Never send the internal missing-category token into public text."""
        public_row = row.copy()
        for field_name in ("productType", "brandDesc"):
            if public_row[field_name] == MISSING_CATEGORY_TOKEN:
                public_row[field_name] = "unavailable"
        return public_row

    def _predict_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            predictions = np.asarray(self._pyfunc_model.predict(frame), dtype=float).reshape(-1)
        except Exception as exc:
            LOGGER.exception("event=inference_failed error_code=inference_failed")
            raise InferenceFailedError("The frozen model could not score the request") from exc
        if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
            raise InferenceFailedError("The frozen model returned invalid probabilities")
        return predictions

    def _frame_for_requests(
        self, requests: list[PredictionRequest]
    ) -> tuple[pd.DataFrame, list[DataContext]]:
        records: list[dict[str, Any]] = []
        contexts: list[DataContext] = []
        for request in requests:
            record, context = self._raw_record_and_context(request)
            records.append(record)
            contexts.append(context)
        return pd.DataFrame.from_records(records, columns=_RAW_INPUT_COLUMNS), contexts

    def _raw_record_and_context(self, request: PredictionRequest) -> tuple[dict[str, Any], DataContext]:
        customer = request.customer_profile
        product = request.product_profile
        customer_imputed = customer is None
        year_imputed = customer is not None and customer.year_of_birth is None
        missing_product_fields = self._missing_product_fields(product)
        product_available = product is not None and len(missing_product_fields) < 4
        product_complete = product is not None and not missing_product_fields
        unknown_categories = self._unknown_categories(customer, product)
        outside_training_range_fields = self._outside_training_range_fields(customer, product)
        context = DataContext(
            customer_profile_imputed=customer_imputed,
            customer_birth_year_imputed=year_imputed,
            product_profile_available=product_available,
            product_profile_complete=product_complete,
            missing_product_fields=missing_product_fields,
            unknown_categories=unknown_categories,
            outside_training_range_fields=outside_training_range_fields,
        )
        record: dict[str, Any] = {
            # This opaque token is consumed only by DonorImputer's stable routing.
            EVENT_CUST_COL: request.customer_context_key or "known-profile-routing-not-used",
            "yearOfBirth": np.nan if customer is None or customer.year_of_birth is None else customer.year_of_birth,
            "isMale": np.nan if customer is None else customer.is_male,
            "shippingCountry": np.nan if customer is None else customer.shipping_country,
            "premier": np.nan if customer is None else customer.premier,
            "productType": MISSING_CATEGORY_TOKEN if product is None or product.product_type is None else product.product_type,
            "brandDesc": MISSING_CATEGORY_TOKEN if product is None or product.brand_desc is None else product.brand_desc,
            "avgGbpPrice": np.nan if product is None or product.avg_gbp_price is None else product.avg_gbp_price,
            "avgDiscountValue": (
                np.nan if product is None or product.avg_discount_value is None else product.avg_discount_value
            ),
        }
        return record, context

    @staticmethod
    def _missing_product_fields(product: Any) -> list[str]:
        if product is None:
            return ["productType", "brandDesc", "avgGbpPrice", "avgDiscountValue"]
        fields = {
            "productType": product.product_type,
            "brandDesc": product.brand_desc,
            "avgGbpPrice": product.avg_gbp_price,
            "avgDiscountValue": product.avg_discount_value,
        }
        return [name for name, value in fields.items() if value is None]

    def _unknown_categories(self, customer: Any, product: Any) -> list[str]:
        raw_categories = {
            "shippingCountry": None if customer is None else customer.shipping_country,
            "productType": None if product is None else product.product_type,
            "brandDesc": None if product is None else product.brand_desc,
        }
        return [
            field_name
            for field_name in _CATEGORY_NAMES
            if raw_categories[field_name] is not None
            and str(raw_categories[field_name]) not in self._category_values[field_name]
        ]

    @staticmethod
    def _outside_training_range_fields(customer: Any, product: Any) -> list[str]:
        fields: list[str] = []
        if customer is not None and customer.year_of_birth is not None:
            if not OBSERVED_YEAR_RANGE[0] <= customer.year_of_birth <= OBSERVED_YEAR_RANGE[1]:
                fields.append("yearOfBirth")
        if product is not None and product.avg_gbp_price is not None:
            if not OBSERVED_PRICE_RANGE[0] <= product.avg_gbp_price <= OBSERVED_PRICE_RANGE[1]:
                fields.append("avgGbpPrice")
        if product is not None and product.avg_discount_value is not None:
            if product.avg_discount_value > OBSERVED_DISCOUNT_MAX:
                fields.append("avgDiscountValue")
        return fields

    def _prediction_payload(self, probability: float, context: DataContext) -> dict[str, Any]:
        warnings = self._warnings_for_context(context)
        return {
            "risk_score": int(np.clip(round(probability * 100), 0, 100)),
            "risk_score_scale": "0-100",
            "score_type": "dataset_conditional_return_risk",
            "data_context": context,
            "warnings": warnings,
        }

    @staticmethod
    def _warnings_for_context(context: DataContext) -> list[str]:
        warnings: list[str] = []
        if context.customer_profile_imputed:
            warnings.append(
                "Customer profile was unavailable; frozen preprocessing used a stable donor profile."
            )
        elif context.customer_birth_year_imputed:
            warnings.append(
                "Customer birth year was unavailable; frozen preprocessing applied its training-derived handling."
            )
        if not context.product_profile_complete:
            warnings.append(
                "Product information is incomplete; product-missing events had materially weaker evaluation performance."
            )
        if context.unknown_categories:
            warnings.append(
                "One or more categorical values were unseen in training and are ignored by the frozen encoder."
            )
        if context.outside_training_range_fields:
            warnings.append(
                "One or more valid values fall outside observed training support."
            )
        return warnings

    def model_info(self) -> ModelInfoResponse:
        return self.provenance.public_model_info()

    def concise_provenance(self) -> ModelProvenance:
        return self.provenance.concise()

    @property
    def methodology_note(self) -> str:
        return METHODOLOGY_NOTE


__all__ = ["ModelService", "ServingProvenance", "DEFAULT_MODEL_URI"]
