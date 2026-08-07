import type { ApiError, ExplanationResponse, ModelInfo, PredictionRequest, PredictionResponse } from "@/lib/types";

const apiBaseUrl = (process.env.NEXT_PUBLIC_RETURNGUARD_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ReturnGuardApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly detail: ApiError | null = null,
  ) {
    super(message);
    this.name = "ReturnGuardApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ReturnGuardApiError("The API could not be reached. Check that the local backend is running.", 0);
  }

  const body = (await response.json().catch(() => null)) as T | ApiError | null;
  if (!response.ok) {
    const detail = body as ApiError | null;
    throw new ReturnGuardApiError(
      detail?.code === "model_unavailable"
        ? "The frozen model is not ready yet. Try again shortly."
        : "The API could not process this request.",
      response.status,
      detail,
    );
  }
  return body as T;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  ready: () => request<{ status: string; code: string | null }>("/ready"),
  predict: (payload: PredictionRequest) =>
    request<PredictionResponse>("/predict", { method: "POST", body: JSON.stringify(payload) }),
  explain: (payload: PredictionRequest) =>
    request<ExplanationResponse>("/explain", { method: "POST", body: JSON.stringify(payload) }),
  model: () => request<ModelInfo>("/model"),
};

export { apiBaseUrl };
