import { Card } from "@heroui/react";
import type { ReactNode } from "react";

import { useSettings, useUpdateSettings } from "@/api/hooks";
import type { UpdateSettingsRequest } from "@/api/types";
import { ErrorBanner, PageHeader } from "@/components/ui";

// A small on/off switch. role="switch" so it reads correctly to assistive tech
// and can be targeted by its accessible name.
function Toggle({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors disabled:opacity-50 ${
        checked ? "bg-[var(--otari-brand)]" : "bg-[var(--otari-line)]"
      }`}
    >
      <span
        className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
          checked ? "translate-x-5" : "translate-x-0.5"
        }`}
      />
    </button>
  );
}

function SettingRow({
  title,
  description,
  checked,
  onChange,
  label,
  disabled,
}: {
  title: string;
  description: ReactNode;
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-6 py-4">
      <div className="min-w-0">
        <div className="text-sm font-medium text-[var(--otari-ink)]">{title}</div>
        <p className="mt-1 text-sm text-[var(--otari-muted)]">{description}</p>
      </div>
      <Toggle checked={checked} onChange={onChange} label={label} disabled={disabled} />
    </div>
  );
}

export function SettingsPage() {
  const settings = useSettings();
  const updateSettings = useUpdateSettings();

  const data = settings.data;
  const pending = updateSettings.isPending;

  const patch = (body: UpdateSettingsRequest) => updateSettings.mutate(body);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Settings"
        description="Control how the gateway meters and lists models. Changes apply immediately and persist across restarts."
      />

      <ErrorBanner error={settings.error ?? updateSettings.error} />

      <Card>
        <Card.Content className="flex flex-col divide-y divide-[var(--otari-line)] px-5 py-1">
          <SettingRow
            title="Model discovery"
            label="Model discovery"
            description={
              <>
                Auto-discover models from your configured providers and list them at <code>GET /v1/models</code>. When
                off, only models you have priced are listed.
              </>
            }
            checked={data?.model_discovery ?? false}
            onChange={(next) => patch({ model_discovery: next })}
            disabled={!data || pending}
          />
          <SettingRow
            title="Default pricing"
            label="Default pricing"
            description={
              <>
                Meter models without a configured price using community-maintained rates (the bundled genai-prices
                dataset). When off, only models with a configured price are metered
                {data?.require_pricing ? "; requests for anything else are rejected because require_pricing is on" : ""}.
              </>
            }
            checked={data?.default_pricing ?? false}
            onChange={(next) => patch({ default_pricing: next })}
            disabled={!data || pending}
          />
        </Card.Content>
      </Card>

      {data ? (
        <p className="text-xs text-[var(--otari-muted)]">
          Mode: {data.mode} · Version {data.version}
          {data.require_pricing ? " · require_pricing on" : ""}
        </p>
      ) : null}
    </div>
  );
}
