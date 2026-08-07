# Stage 6 — ReturnGuard frontend

Stage 6 adds a small, accessible Next.js demonstrator for the frozen
`returnguard-a3-v1` model-of-record. It is a consumer of the Stage 5 API, not a
new model-development surface: it does not train, tune, calibrate, promote a
model, read raw ASOS data, or run the official test evaluation.

## Run locally

Start the already-registered frozen API in one terminal:

```bash
.venv/bin/uvicorn backend.main:app --reload
```

Then start the frontend in another:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. The default API base URL is
`http://localhost:8000`; set `NEXT_PUBLIC_RETURNGUARD_API_URL` only when the
backend is served from another explicit origin. The backend allowlist defaults
to `http://localhost:3000` and can be set as a comma-separated list with
`RETURNGUARD_CORS_ORIGINS`.

## Interaction contract

The page calls only `GET /health`, `GET /ready`, `GET /model`, `POST /predict`,
and user-initiated `POST /explain`. It never uses `/predict/batch`, accesses
MLflow, transforms model features, creates derived features, or decides a
business action.

The client polls health/readiness every three seconds. A score button remains
disabled until the API reports ready, while the form remains editable. API
validation errors are associated with their form fields where the public API
provides one. Network and readiness failures stay user-readable and do not
expose technical details.

The three supplied scenarios are entirely synthetic: a complete profile, an
unavailable customer profile, and incomplete product data. An opaque
`customer_context_key` is generated only in browser memory when the customer
profile is unavailable. It is sent only to satisfy the API’s stable donor
routing contract; it is not displayed or added to results, provenance, or
explanations. Each loaded example starts a fresh key on its next score request.

## Score and explanation presentation

The result renders the API’s deterministic 0–100 score as a continuous meter.
It intentionally has no low/medium/high bands, operational recommendation, or
merchant-wide probability wording. The page repeats the returner-enriched
sample warning and displays structured missingness, unknown-category, and
out-of-range context as cautious evidence warnings.

Explanations are strictly on demand and use the saved score request, rather
than current form values. Editing the form marks a score stale and disables a
new explanation request until it is recalculated. The API’s grouped factor
directions are shown without raw SHAP values, encoded feature names, customer
donor details, or causal claims. The Model details disclosure fetches only the
safe `/model` provenance contract.

## Verification

The client unit suite has no raw data, MLflow, or final-test dependency:

```bash
cd frontend
npm run lint
npm run typecheck
npm test
npm run build
```

The repository CI runs these checks alongside the existing synthetic Python
suite. It uses Node 23 and no frontend environment file. Full local browser
integration requires an already-created local model-of-record, but it does not
retrain or invoke the official test evaluation.

## Scope boundary

This stage intentionally does not add authentication, persistence, batch UI,
analytics, monitoring, Docker, or deployment infrastructure. It preserves the
frozen A3 governance and treats all model output as research-context evidence.
