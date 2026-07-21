import { ComboBox, Input, Label, ListBox, ListBoxItem } from "@heroui/react";
import { type ReactNode, useMemo } from "react";

import { useDiscoverableModels } from "@/api/hooks";
import type { DiscoverableModel } from "@/api/types";

// How many matches to render at once. A single provider can report a few hundred
// models, and past this the popover is a wall of text nobody scrolls; typing one
// more character is faster. How many were withheld is reported rather than the
// list silently ending.
const MAX_VISIBLE = 50;

/**
 * Model selector backed by GET /v1/models/discoverable.
 *
 * Free text is always allowed: discovery only sees what the configured
 * credentials expose, so a model behind an unconfigured provider, a brand-new
 * release, or a backend that cannot list must stay typeable. The dropdown is a
 * shortcut, never a whitelist.
 *
 * Everything a provider reports is offered, unfiltered. any-llm's model type
 * carries no capability field, so there is no honest way to tell a chat model
 * from an embedding or TTS one here, and guessing from name patterns would hide
 * real models. Search is the answer instead.
 *
 * Rows are flat, each showing the full `provider:model` selector, rather than
 * grouped under provider headers. React Aria's sectioned collections
 * (ListBoxSection/Collection) throw "childNodes is not supported" under React
 * 19.2's development build, whose performance-track logging walks props and
 * trips over React Aria's synthetic collection DOM. The uncaught error breaks
 * the commit, so picking an option silently fails to update the field.
 * Production builds are unaffected, but a picker that only works in prod is not
 * worth the headers, and the provider is legible on every row regardless.
 */
export function ModelComboBox({
  label,
  value,
  onChange,
  description,
  placeholder = "provider:model",
  autoFocus,
  isRequired,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  description?: ReactNode;
  placeholder?: string;
  autoFocus?: boolean;
  isRequired?: boolean;
}) {
  const discoverable = useDiscoverableModels();

  const { visible, total, failed } = useMemo(() => {
    const query = value.trim().toLowerCase();
    const providers = discoverable.data?.providers ?? [];
    // Provider order is preserved, so rows still cluster by provider even
    // without section headers.
    const all: DiscoverableModel[] = providers.flatMap((provider) => provider.models);
    const hits = query ? all.filter((model) => model.key.toLowerCase().includes(query)) : all;
    return {
      visible: hits.slice(0, MAX_VISIBLE),
      total: hits.length,
      failed: providers.filter((provider) => !provider.ok),
    };
  }, [discoverable.data, value]);

  const hint = ((): ReactNode => {
    if (discoverable.isLoading) {
      return "Loading models from your providers…";
    }
    // A failed provider is worth saying out loud: its models are simply absent
    // from the list, which is indistinguishable from a provider that has none.
    if (failed.length > 0) {
      const names = failed.map((provider) => provider.provider).join(", ");
      return `Could not list models for ${names}. Check that provider's credentials, or type the model key directly.`;
    }
    if (total > visible.length) {
      return `Showing ${visible.length} of ${total} matches. Keep typing to narrow them.`;
    }
    return description;
  })();

  return (
    <ComboBox.Root
      // Discovery is not authoritative, so anything typed stands on its own.
      allowsCustomValue
      // Otherwise a query matching nothing closes the popover, which reads as
      // "the field broke" rather than "no matches".
      allowsEmptyCollection
      // HeroUI defaults this to "focus", which with autoFocus drops the whole
      // list open as soon as the form appears. React Aria marks everything
      // outside an open popover aria-hidden, so the price fields and Save button
      // would be unreachable to a screen reader before a single keystroke. Open
      // on typing (or the trigger) instead.
      menuTrigger="input"
      inputValue={value}
      onInputChange={onChange}
      onSelectionChange={(key) => {
        if (key != null) {
          onChange(String(key));
        }
      }}
      isRequired={isRequired}
      // Cap the width so the field and its dropdown trigger stay within easy
      // reach instead of stretching across a wide form.
      className="flex max-w-md flex-col gap-1"
    >
      {/* HeroUI marks a required field's label through CSS; see Field. */}
      <Label className="text-sm font-medium text-[var(--otari-ink)]">{label}</Label>
      <ComboBox.InputGroup>
        <Input placeholder={placeholder} autoFocus={autoFocus} />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover>
        <ListBox items={visible} className="max-h-72 overflow-auto">
          {(model: DiscoverableModel) => (
            // id and textValue are the full selector, so picking a row puts what
            // the API expects into the field.
            <ListBoxItem id={model.key} textValue={model.key}>
              {model.key}
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
      {hint ? <span className="text-xs text-[var(--otari-muted)]">{hint}</span> : null}
    </ComboBox.Root>
  );
}
