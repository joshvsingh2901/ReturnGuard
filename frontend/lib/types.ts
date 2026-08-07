export type CustomerProfile = {
  yearOfBirth: number | null;
  isMale: boolean;
  shippingCountry: string;
  premier: boolean;
};

export type ProductProfile = {
  productType: string | null;
  brandDesc: string | null;
  avgGbpPrice: number | null;
  avgDiscountValue: number | null;
};

export type PredictionRequest = {
  customer_context_key?: string;
  customer_profile: CustomerProfile | null;
  product_profile: ProductProfile | null;
};

export type DataContext = {
  customer_profile_imputed: boolean;
  customer_birth_year_imputed: boolean;
  product_profile_available: boolean;
  product_profile_complete: boolean;
  missing_product_fields: string[];
  unknown_categories: Array<{ field: string; value: string }>;
  outside_training_range_fields: string[];
};

export type ModelProvenance = {
  semantic_version: string;
  registry_model: string;
  registry_version: string;
  feature_manifest_hash: string;
};

export type PredictionResponse = {
  risk_score: number;
  risk_score_scale: "0-100";
  score_type: "dataset_conditional_return_risk";
  data_context: DataContext;
  warnings: string[];
  request_id: string;
  model: ModelProvenance;
  methodology_note: string;
};

export type ExplanationFactor = { display_name: string; direction: "higher" | "lower" };

export type ExplanationResponse = PredictionResponse & {
  top_risk_factors: ExplanationFactor[];
  top_protective_factors: ExplanationFactor[];
  explanation_method: "tree_shap_grouped_noncausal";
};

export type ModelInfo = {
  semantic_version: string;
  registry_model: string;
  registry_version: string;
  feature_manifest_hash: string;
  dataset_version: string;
  governance_status: string;
  intended_use: string;
  limitations: string[];
  population_shift_warning: string;
};

export type ApiError = {
  code: string;
  request_id?: string;
  errors?: Array<{ field: string; message: string; type: string }>;
};
