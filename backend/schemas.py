"""Raw API contracts for ReturnGuard's frozen inference artifact."""

from __future__ import annotations

import math
from typing import Annotated, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

MISSING_CATEGORY_TOKEN = "__MISSING__"
OBSERVED_YEAR_RANGE = (1890, 2020)
OBSERVED_PRICE_RANGE = (1.25, 518.0)
OBSERVED_DISCOUNT_MAX = 45.3
PLAUSIBLE_MIN_BIRTH_YEAR = 1880
PLAUSIBLE_MAX_BIRTH_YEAR = 2026

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CustomerProfile(ContractModel):
    """Known customer attributes; all but birth year are required when present."""

    year_of_birth: Optional[int] = Field(
        default=None,
        alias="yearOfBirth",
        description="Birth year; use null when unknown. The historical 1900 sentinel is not accepted.",
        examples=[1988],
    )
    is_male: Optional[bool] = Field(
        default=None,
        alias="isMale",
        description="Required for a known customer profile; use customer_profile=null when the profile is absent.",
        examples=[False],
    )
    shipping_country: Optional[NonEmptyString] = Field(
        default=None, alias="shippingCountry", examples=["Country_A"]
    )
    premier: Optional[bool] = Field(
        default=None,
        description="Required for a known customer profile.",
        examples=[True],
    )

    @field_validator("year_of_birth")
    @classmethod
    def validate_year(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value == 1900:
            raise ValueError("yearOfBirth=1900 is a historical missing-value sentinel; send null instead")
        if not PLAUSIBLE_MIN_BIRTH_YEAR <= value <= PLAUSIBLE_MAX_BIRTH_YEAR:
            raise ValueError("yearOfBirth must be a plausible calendar year")
        return value

    @model_validator(mode="after")
    def require_complete_customer_profile(self) -> "CustomerProfile":
        missing = []
        if self.is_male is None:
            missing.append("isMale")
        if self.shipping_country is None:
            missing.append("shippingCountry")
        if self.premier is None:
            missing.append("premier")
        if missing:
            raise ValueError(f"known customer_profile requires: {', '.join(missing)}")
        return self


class ProductProfile(ContractModel):
    """Product fields accepted by frozen A3 before its fitted preprocessing."""

    product_type: Optional[NonEmptyString] = Field(
        default=None,
        alias="productType",
        description="Product type. Use null, never the reserved __MISSING__ token, when unavailable.",
        examples=["Jeans"],
    )
    brand_desc: Optional[NonEmptyString] = Field(
        default=None,
        alias="brandDesc",
        description="Brand descriptor. Unknown categories are accepted and ignored by the frozen encoder.",
        examples=["Brand_A"],
    )
    avg_gbp_price: Optional[float] = Field(
        default=None,
        alias="avgGbpPrice",
        description="Positive GBP price. Valid values outside observed training support remain scoreable with a warning.",
        examples=[54.99],
    )
    avg_discount_value: Optional[float] = Field(
        default=None,
        alias="avgDiscountValue",
        description="Discount percentage from 0 through 100. Values above observed support receive a warning.",
        examples=[15.0],
    )

    @field_validator("product_type", "brand_desc")
    @classmethod
    def reject_reserved_missing_token(cls, value: Optional[str]) -> Optional[str]:
        if value == MISSING_CATEGORY_TOKEN:
            raise ValueError("Use null for a missing product category; __MISSING__ is reserved")
        return value

    @field_validator("avg_gbp_price")
    @classmethod
    def validate_price(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not math.isfinite(value) or value <= 0:
            raise ValueError("avgGbpPrice must be finite and greater than zero")
        return value

    @field_validator("avg_discount_value")
    @classmethod
    def validate_discount(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("avgDiscountValue must be finite and between 0 and 100")
        return value


class PredictionRequest(ContractModel):
    """One customer × product-variant purchase event before preprocessing."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "customer_context_key": "opaque-stable-token",
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
                },
                {
                    "customer_context_key": "stable-missing-customer",
                    "customer_profile": None,
                    "product_profile": {
                        "productType": "Jeans",
                        "brandDesc": None,
                        "avgGbpPrice": None,
                        "avgDiscountValue": 15.0,
                    },
                },
            ]
        },
    )

    customer_context_key: Optional[NonEmptyString] = Field(
        default=None,
        description="Opaque stable token required only when customer_profile is null. It is used for donor routing, not prediction.",
        examples=["merchant-token-42"],
    )
    customer_profile: Optional[CustomerProfile] = Field(
        ...,
        description="Known customer attributes, or null to use the frozen donor-imputation pathway.",
    )
    product_profile: Optional[ProductProfile] = Field(
        ...,
        description="Complete, partial, or null product attributes. Missing values remain supported by the frozen artifact.",
    )

    @model_validator(mode="after")
    def require_context_key_for_missing_customer(self) -> "PredictionRequest":
        if self.customer_profile is None and self.customer_context_key is None:
            raise ValueError("customer_context_key is required when customer_profile is null")
        return self


class BatchPredictionRequest(ContractModel):
    events: list[PredictionRequest] = Field(min_length=1, max_length=100)


class DataContext(ContractModel):
    customer_profile_imputed: bool
    customer_birth_year_imputed: bool
    product_profile_available: bool
    product_profile_complete: bool
    missing_product_fields: list[str]
    unknown_categories: list[str]
    outside_training_range_fields: list[str]


class ModelProvenance(ContractModel):
    semantic_version: str
    registry_model: str
    registry_version: str
    feature_manifest_hash: str


class PredictionResult(ContractModel):
    risk_score: int = Field(ge=0, le=100)
    risk_score_scale: str = "0-100"
    score_type: str = "dataset_conditional_return_risk"
    data_context: DataContext
    warnings: list[str]


class PredictionResponse(PredictionResult):
    request_id: str
    model: ModelProvenance
    methodology_note: str


class BatchPredictionResponse(ContractModel):
    request_id: str
    results: list[PredictionResult]
    model: ModelProvenance
    methodology_note: str


class ExplanationFactor(ContractModel):
    display_name: str
    direction: str


class ExplanationResponse(PredictionResponse):
    top_risk_factors: list[ExplanationFactor]
    top_protective_factors: list[ExplanationFactor]
    explanation_method: str = "tree_shap_grouped_noncausal"


class ReadinessResponse(ContractModel):
    status: str
    code: Optional[str] = None


class ModelInfoResponse(ContractModel):
    semantic_version: str
    registry_model: str
    registry_version: str
    mlflow_source_run_id: str
    feature_manifest_hash: str
    dataset_version: str
    governance_status: str
    intended_use: str
    limitations: list[str]
    population_shift_warning: str
