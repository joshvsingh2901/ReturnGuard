import { useState } from "react";

import type { ExplanationResponse, ModelInfo, PredictionResponse } from "@/lib/types";
import { presentWarnings } from "@/lib/warnings";

import { ExplanationPanel } from "@/components/ExplanationPanel";

type Props = {
  prediction: PredictionResponse | null;
  explanation: ExplanationResponse | null;
  explanationLoading: boolean;
  predictionStale: boolean;
  onExplain: () => void;
  loadModelInfo: () => Promise<ModelInfo>;
};

function ScoreMeter({ score }: { score: number }) {
  return (
    <div className="score-meter" role="meter" aria-label="Return-risk score" aria-valuemin={0} aria-valuemax={100} aria-valuenow={score}>
      <div className="score-meter__track"><span style={{ width: `${score}%` }} /></div>
      <div className="score-meter__labels"><span>Lower</span><span>Higher</span></div>
    </div>
  );
}

function ModelDetails({ prediction, loadModelInfo }: { prediction: PredictionResponse; loadModelInfo: () => Promise<ModelInfo> }) {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleToggle = async (open: boolean) => {
    if (!open || info || error) return;
    try {
      setInfo(await loadModelInfo());
    } catch {
      setError("Model details are temporarily unavailable.");
    }
  };

  return (
    <details className="model-details" onToggle={(event) => void handleToggle(event.currentTarget.open)}>
      <summary>Model details</summary>
      <dl>
        <div><dt>Version</dt><dd>{prediction.model.semantic_version}</dd></div>
        <div><dt>Registry</dt><dd>{prediction.model.registry_model} · v{prediction.model.registry_version}</dd></div>
        <div><dt>Feature manifest</dt><dd className="hash">{prediction.model.feature_manifest_hash}</dd></div>
        {info && <><div><dt>Dataset</dt><dd>{info.dataset_version}</dd></div><div><dt>Governance</dt><dd>{info.governance_status.replaceAll("_", " ")}</dd></div></>}
      </dl>
      {error && <p className="field-error">{error}</p>}
    </details>
  );
}

export function ResultPanel({ prediction, explanation, explanationLoading, predictionStale, onExplain, loadModelInfo }: Props) {
  if (!prediction) {
    return (
      <aside className="result-panel result-panel--empty" aria-labelledby="result-heading">
        <p className="eyebrow">Result</p>
        <h2 id="result-heading">Your return-risk score will appear here.</h2>
        <p>Enter a synthetic purchase scenario, then calculate a score from the frozen A3 model.</p>
      </aside>
    );
  }

  const warnings = presentWarnings(prediction.data_context, prediction.warnings);
  return (
    <aside className="result-panel" aria-live="polite" aria-labelledby="result-heading">
      <p className="eyebrow">Frozen A3 result</p>
      <h2 id="result-heading">Dataset-conditional return-risk score</h2>
      <div className="score-value"><strong>{Math.round(prediction.risk_score)}</strong><span>/ 100</span></div>
      <ScoreMeter score={prediction.risk_score} />
      <p className="score-copy">Use this for relative comparison within the research context—not as a merchant-wide return probability.</p>

      {predictionStale && <p className="stale-note">The form has changed. Calculate a new score before requesting an explanation.</p>}
      {warnings.length > 0 && (
        <section className="warning-list" aria-label="Data context warnings">
          <h3>Data context</h3>
          <ul>{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </section>
      )}

      <ExplanationPanel
        explanation={explanation}
        available={!predictionStale}
        loading={explanationLoading}
        disabled={false}
        onRequest={onExplain}
      />
      <ModelDetails prediction={prediction} loadModelInfo={loadModelInfo} />
    </aside>
  );
}
