export function MethodologyNotice() {
  return (
    <section className="methodology-notice" aria-label="How to interpret this score">
      <p>
        <strong>Interpret with care.</strong> This score comes from a returner-enriched ASOS research sample. It supports relative risk ranking, not a merchant-wide probability of return.
      </p>
      <details>
        <summary>Read the methodology note</summary>
        <p>
          ReturnGuard scores one purchase event using the frozen A3 model and information available at purchase time. Model factors describe model behavior; they are not causal explanations or a decision about an individual customer.
        </p>
      </details>
    </section>
  );
}
