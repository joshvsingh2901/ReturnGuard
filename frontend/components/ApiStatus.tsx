export type ApiStatus = "checking" | "ready" | "unavailable";

export function ApiStatusBadge({ status }: { status: ApiStatus }) {
  const content = {
    checking: ["Checking service", "status--checking"],
    ready: ["API ready", "status--ready"],
    unavailable: ["API unavailable", "status--unavailable"],
  }[status];

  return (
    <span className={`status ${content[1]}`} role="status" aria-live="polite">
      <span className="status__dot" aria-hidden="true" />
      {content[0]}
    </span>
  );
}
