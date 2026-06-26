import { Button, Card } from "@heroui/react";
import { useState } from "react";
import type { ReactNode } from "react";

import { ApiError } from "@/api/client";

export function StatCard({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <Card className="flex-1 min-w-[180px]">
      <Card.Content className="flex flex-col gap-1 p-5">
        <span className="text-xs font-medium uppercase tracking-wide text-[var(--otari-muted)]">{label}</span>
        <span className="text-2xl font-semibold text-[var(--otari-ink)]">{value}</span>
        {hint ? <span className="text-xs text-[var(--otari-muted)]">{hint}</span> : null}
      </Card.Content>
    </Card>
  );
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Something went wrong.";
}

export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) {
    return null;
  }
  return (
    <div
      role="alert"
      className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
    >
      {errorMessage(error)}
    </div>
  );
}

export function InfoBanner({ tone = "info", children }: { tone?: "info" | "warning"; children: ReactNode }) {
  const styles =
    tone === "warning"
      ? "border-amber-200 bg-amber-50 text-amber-800"
      : "border-[var(--otari-brand)] bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]";
  return <div className={`rounded-lg border px-4 py-3 text-sm ${styles}`}>{children}</div>;
}

export function PageHeader({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold text-[var(--otari-ink)]">{title}</h1>
        {description ? <p className="mt-1 text-sm text-[var(--otari-muted)]">{description}</p> : null}
      </div>
      {action}
    </div>
  );
}

// A destructive button that requires a second click to confirm, avoiding a
// modal dependency for revoke/delete actions.
export function ConfirmButton({
  children,
  confirmLabel,
  onConfirm,
  isPending,
}: {
  children: ReactNode;
  confirmLabel: string;
  onConfirm: () => void;
  isPending?: boolean;
}) {
  const [armed, setArmed] = useState(false);

  if (armed) {
    return (
      <span className="inline-flex items-center gap-1">
        <Button size="sm" variant="danger" isDisabled={isPending} onPress={onConfirm}>
          {confirmLabel}
        </Button>
        <Button size="sm" variant="ghost" isDisabled={isPending} onPress={() => setArmed(false)}>
          Cancel
        </Button>
      </span>
    );
  }

  return (
    <Button size="sm" variant="danger-soft" onPress={() => setArmed(true)}>
      {children}
    </Button>
  );
}
