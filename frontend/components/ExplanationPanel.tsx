import type { ExplanationResponse } from "@/lib/types";

type Props = {
  explanation: ExplanationResponse | null;
  available: boolean;
  loading: boolean;
  disabled: boolean;
  onRequest: () => void;
};

function FactorList({ title, factors, modifier }: { title: string; factors: ExplanationResponse["top_risk_factors"]; modifier: string }) {
  return (
    <div className={`factor-list ${modifier}`}>
      <h4>{title}</h4>
      {factors.length ? <ul>{factors.map((factor) => <li key={`${factor.display_name}-${factor.direction}`}>{factor.display_name}</li>)}</ul> : <p>No grouped factors were returned.</p>}
    </div>
  );
}

export function ExplanationPanel({ explanation, available, loading, disabled, onRequest }: Props) {
  return (
    <section className="explanation-panel" aria-labelledby="explanation-heading">
      <div>
        <p className="eyebrow">Model behavior</p>
        <h3 id="explanation-heading">Why this score?</h3>
      </div>
      {!explanation ? (
        <>
          <p className="quiet-note">Grouped factors are available only for the saved score. They describe model behavior, not causes.</p>
          <button className="secondary-button" type="button" disabled={!available || disabled || loading} onClick={onRequest}>
            {loading ? "Loading explanation…" : "Show explanation"}
          </button>
        </>
      ) : (
        <>
          <div className="factor-grid">
            <FactorList title="Associated with higher score" factors={explanation.top_risk_factors} modifier="factor-list--higher" />
            <FactorList title="Associated with lower score" factors={explanation.top_protective_factors} modifier="factor-list--lower" />
          </div>
          <p className="explanation-disclaimer">These grouped factors describe the frozen model’s behavior for this request. They are not causal explanations.</p>
        </>
      )}
    </section>
  );
}
