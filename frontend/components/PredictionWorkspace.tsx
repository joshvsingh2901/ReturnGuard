"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useCallback, useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";

import { ApiStatusBadge, type ApiStatus } from "@/components/ApiStatus";
import { MethodologyNotice } from "@/components/MethodologyNotice";
import { PredictionForm } from "@/components/PredictionForm";
import { ResultPanel } from "@/components/ResultPanel";
import { api, ReturnGuardApiError } from "@/lib/api";
import { defaultScenario, scenarioSchema, toPredictionRequest, type ScenarioFormValues } from "@/lib/form";
import { demoPresets } from "@/lib/presets";
import type { ExplanationResponse, PredictionRequest, PredictionResponse } from "@/lib/types";

function newContextKey(): string {
  return globalThis.crypto?.randomUUID?.() ?? `returnguard-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function PredictionWorkspace() {
  const form = useForm<ScenarioFormValues>({
    resolver: zodResolver(scenarioSchema),
    defaultValues: defaultScenario,
    mode: "onSubmit",
  });
  const contextKey = useRef<string | null>(null);
  const requestSnapshot = useRef<PredictionRequest | null>(null);
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState("");

  const checkReadiness = useCallback(async () => {
    try {
      await api.health();
      const ready = await api.ready();
      setApiStatus(ready.status === "ready" ? "ready" : "unavailable");
    } catch {
      setApiStatus("unavailable");
    }
  }, []);

  useEffect(() => {
    const initialCheck = window.setTimeout(() => void checkReadiness(), 0);
    const interval = window.setInterval(() => void checkReadiness(), 3000);
    return () => {
      window.clearTimeout(initialCheck);
      window.clearInterval(interval);
    };
  }, [checkReadiness]);

  const clearResultIfEditing = () => {
    setExplanation(null);
    setSubmitError(null);
  };

  const onPresetChange = (id: string) => {
    const preset = demoPresets.find((item) => item.id === id);
    setSelectedPreset(id);
    if (!preset) return;
    contextKey.current = null;
    requestSnapshot.current = null;
    setPrediction(null);
    clearResultIfEditing();
    form.reset(preset.values);
  };

  const submit = async () => {
    if (apiStatus !== "ready") return;
    const values = form.getValues();
    contextKey.current ??= newContextKey();
    const payload = toPredictionRequest(values, contextKey.current);
    setSubmitting(true);
    setSubmitError(null);
    setExplanation(null);
    try {
      const result = await api.predict(payload);
      requestSnapshot.current = payload;
      setPrediction(result);
      form.reset(values);
    } catch (error) {
      if (error instanceof ReturnGuardApiError && error.status === 422 && error.detail?.errors) {
        for (const issue of error.detail.errors) {
          const field = issue.field.split(".").at(-1);
          const frontendField = field === "year_of_birth" ? "yearOfBirth" : field;
          if (frontendField && frontendField in defaultScenario) {
            form.setError(frontendField as keyof ScenarioFormValues, { message: issue.message });
          }
        }
      }
      setSubmitError(error instanceof Error ? error.message : "Unable to calculate a score right now.");
    } finally {
      setSubmitting(false);
    }
  };

  const requestExplanation = async () => {
    if (!requestSnapshot.current || form.formState.isDirty) return;
    setExplaining(true);
    setSubmitError(null);
    try {
      setExplanation(await api.explain(requestSnapshot.current));
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Unable to load an explanation right now.");
    } finally {
      setExplaining(false);
    }
  };

  const predictionStale = Boolean(prediction && form.formState.isDirty);
  return (
    <main>
      <header className="site-header">
        <a className="wordmark" href="#workspace" aria-label="ReturnGuard home">ReturnGuard</a>
        <ApiStatusBadge status={apiStatus} />
      </header>
      <section className="hero" aria-labelledby="page-title">
        <p className="eyebrow">Frozen model demonstrator</p>
        <h1 id="page-title">Understand dataset-conditional return risk before a purchase.</h1>
        <p>Explore one synthetic purchase scenario at a time with the governed ReturnGuard A3 model.</p>
      </section>
      <MethodologyNotice />
      <section className="workspace" id="workspace" aria-label="Return-risk scoring workspace">
        <div className="form-card">
          <PredictionForm form={form} apiReady={apiStatus === "ready"} submitting={submitting} selectedPreset={selectedPreset} presets={demoPresets} onPresetChange={onPresetChange} onSubmit={submit} />
          {submitError && <p className="request-error" role="alert">{submitError}</p>}
        </div>
        <ResultPanel prediction={prediction} explanation={explanation} explanationLoading={explaining} predictionStale={predictionStale} onExplain={() => void requestExplanation()} loadModelInfo={api.model} />
      </section>
      <footer>ReturnGuard portfolio demonstrator · frozen model lifecycle preserved</footer>
    </main>
  );
}
