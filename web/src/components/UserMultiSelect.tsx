import { ComboBox, Input, ListBox, ListBoxItem } from "@heroui/react";
import { type ReactNode, useMemo, useState } from "react";

import type { User } from "@/api/types";

interface Option {
  id: string;
  label: string;
}

const MAX_VISIBLE = 50;

// A chip picker of existing users, for assigning a budget to several at once. Only
// lists named users (virtual apikey-* shadows are excluded); it never creates a
// user, matching the user-first model where users exist before they are assigned.
export function UserMultiSelect({
  value,
  onChange,
  users,
  label,
  description,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  users: User[];
  label: string;
  description?: ReactNode;
}) {
  const [query, setQuery] = useState("");

  const options = useMemo<Option[]>(
    () =>
      users
        .filter((u) => !u.user_id.startsWith("apikey-"))
        .map((u) => ({ id: u.user_id, label: u.alias ? `${u.user_id} (${u.alias})` : u.user_id })),
    [users],
  );

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return options
      .filter((o) => !value.includes(o.id))
      .filter((o) => !q || o.id.toLowerCase().includes(q) || o.label.toLowerCase().includes(q))
      .slice(0, MAX_VISIBLE);
  }, [options, value, query]);

  const add = (id: string) => {
    if (!value.includes(id)) onChange([...value, id]);
    setQuery("");
  };
  const remove = (id: string) => onChange(value.filter((v) => v !== id));

  return (
    <div className="flex flex-col gap-2">
      <div>
        <span className="text-sm font-medium text-[var(--otari-ink)]">{label}</span>
        {description ? <p className="text-xs text-[var(--otari-muted)]">{description}</p> : null}
      </div>
      {value.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {value.map((id) => (
            <span
              key={id}
              className="inline-flex items-center gap-1 rounded-full bg-[var(--otari-brand-tint)] px-2.5 py-1 font-mono text-xs text-[var(--otari-brand-dark)]"
            >
              {id}
              <button
                type="button"
                aria-label={`Remove ${id}`}
                onClick={() => remove(id)}
                className="text-[var(--otari-brand-dark)] hover:text-red-700"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      ) : null}
      {options.length === 0 ? (
        <span className="text-xs text-[var(--otari-muted)]">
          No users yet. Create users first, then assign them here or from the Users page.
        </span>
      ) : (
        <ComboBox.Root
          allowsEmptyCollection
          menuTrigger="input"
          inputValue={query}
          onInputChange={setQuery}
          selectedKey={null}
          onSelectionChange={(key) => {
            if (key != null) add(String(key));
          }}
          className="flex flex-col gap-1"
        >
          <ComboBox.InputGroup>
            <Input aria-label="Add a user" placeholder="Search users…" autoComplete="off" />
            <ComboBox.Trigger />
          </ComboBox.InputGroup>
          <ComboBox.Popover>
            <ListBox items={visible} className="max-h-72 overflow-auto">
              {(option: Option) => (
                <ListBoxItem id={option.id} textValue={option.label}>
                  {option.label}
                </ListBoxItem>
              )}
            </ListBox>
          </ComboBox.Popover>
        </ComboBox.Root>
      )}
    </div>
  );
}
