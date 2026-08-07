import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PredictionWorkspace } from "@/components/PredictionWorkspace";
import { api } from "@/lib/api";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: { health: vi.fn(), ready: vi.fn(), predict: vi.fn(), explain: vi.fn(), model: vi.fn() },
  };
});

const apiMock = api as unknown as {
  health: ReturnType<typeof vi.fn>;
  ready: ReturnType<typeof vi.fn>;
  predict: ReturnType<typeof vi.fn>;
  explain: ReturnType<typeof vi.fn>;
  model: ReturnType<typeof vi.fn>;
};

const score = {
  risk_score: 67,
  risk_score_scale: "0-100" as const,
  score_type: "dataset_conditional_return_risk" as const,
  data_context: {
    customer_profile_imputed: true,
    customer_birth_year_imputed: false,
    product_profile_available: true,
    product_profile_complete: true,
    missing_product_fields: [],
    unknown_categories: [],
    outside_training_range_fields: [],
  },
  warnings: [],
  request_id: "request-123",
  model: { semantic_version: "returnguard-a3-v1", registry_model: "returnguard-a3", registry_version: "1", feature_manifest_hash: "feature-hash" },
  methodology_note: "Research sample warning.",
};

describe("PredictionWorkspace", () => {
  afterEach(() => cleanup());

  beforeEach(() => {
    vi.clearAllMocks();
    apiMock.health.mockResolvedValue({ status: "ok" });
    apiMock.ready.mockResolvedValue({ status: "ready", code: null });
    apiMock.predict.mockResolvedValue(score);
    apiMock.explain.mockResolvedValue({
      ...score,
      top_risk_factors: [{ display_name: "Product type: Jeans", direction: "higher" }],
      top_protective_factors: [{ display_name: "Price and discount", direction: "lower" }],
      explanation_method: "tree_shap_grouped_noncausal",
    });
  });

  it("loads a synthetic preset, scores it, and keeps its opaque key out of the UI", async () => {
    const user = userEvent.setup();
    render(<PredictionWorkspace />);
    await screen.findByText("API ready");

    await user.selectOptions(screen.getByLabelText("Load example"), "customer-missing");
    await user.click(screen.getByRole("button", { name: "Calculate return-risk score" }));

    await screen.findByText("67");
    expect(apiMock.predict).toHaveBeenCalledOnce();
    const payload = apiMock.predict.mock.calls[0][0];
    expect(payload.customer_context_key).toBeTruthy();
    expect(screen.queryByText(payload.customer_context_key)).not.toBeInTheDocument();
    expect(screen.getByText(/governed donor profile/i)).toBeInTheDocument();
  });

  it("requests an explanation only for the saved score snapshot", async () => {
    const user = userEvent.setup();
    render(<PredictionWorkspace />);
    await screen.findByText("API ready");
    await user.selectOptions(screen.getByLabelText("Load example"), "complete");
    await user.click(screen.getByRole("button", { name: "Calculate return-risk score" }));
    await screen.findByText("67");

    await user.click(screen.getByRole("button", { name: "Show explanation" }));
    await screen.findByText("Associated with higher score");
    expect(apiMock.explain).toHaveBeenCalledWith(apiMock.predict.mock.calls[0][0]);

    fireEvent.change(screen.getByLabelText(/Year of birth/i), { target: { value: "1989" } });
    await waitFor(() => expect(screen.getByText(/form has changed/i)).toBeInTheDocument());
  });
});
