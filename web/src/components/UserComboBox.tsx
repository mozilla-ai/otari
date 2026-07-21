import { ComboBox, Description, Input, Label, ListBox, ListBoxItem } from "@heroui/react";
import { useState } from "react";
import type { ReactNode } from "react";

import type { User } from "@/api/types";

interface Option {
  id: string;
  name: string;
}

// A required "owner" picker for a new API key: choose an existing user or type a
// new id to create one (the keys API creates a named user for any id it does not
// know). This is the dashboard's user-first gate; it never mints an anonymous
// virtual user the way an omitted id at the API would. Virtual users are left out
// of the options: you attach keys to people/teams you name, not to key-shadows.
export function UserComboBox({
  value,
  onChange,
  users,
  description,
}: {
  value: string;
  onChange: (userId: string) => void;
  users: User[];
  description?: ReactNode;
}) {
  const options: Option[] = users
    .filter((u) => !u.user_id.startsWith("apikey-"))
    .map((u) => ({ id: u.user_id, name: u.alias ? `${u.user_id} (${u.alias})` : u.user_id }));

  const [text, setText] = useState(value);
  const query = text.trim().toLowerCase();
  const visible = options
    .filter((o) => !query || o.id.toLowerCase().includes(query) || o.name.toLowerCase().includes(query))
    .slice(0, 50);

  // What the input holds is not necessarily the user_id: when an option is picked
  // the ComboBox writes that option's display text back into the input, which
  // re-fires onInputChange. Resolve either form (raw id, or "id (alias)" label) to
  // the canonical id, or the submitted owner would be the label and the keys API
  // would silently create a second user named after it.
  const resolveId = (raw: string): string => {
    const trimmed = raw.trim();
    const match = options.find((o) => o.id === trimmed || o.name === trimmed);
    return match ? match.id : trimmed;
  };

  const selectedId = resolveId(text);
  const known = options.some((o) => o.id === selectedId);
  const creatingHint =
    selectedId !== "" && !known ? (
      <span>
        Creates a new user <code>{selectedId}</code>.
      </span>
    ) : (
      (description ?? "Spend and budgets track against this user.")
    );

  return (
    <ComboBox.Root
      allowsCustomValue
      allowsEmptyCollection
      menuTrigger="focus"
      inputValue={text}
      onInputChange={(next) => {
        setText(next);
        onChange(resolveId(next));
      }}
      onSelectionChange={(key) => {
        if (key != null) {
          const id = String(key);
          setText(id);
          onChange(id);
        }
      }}
      className="flex flex-col gap-1"
    >
      <Label className="text-sm font-medium text-[var(--otari-ink)]">Owner</Label>
      <ComboBox.InputGroup>
        {/* Not a credential field: keep password managers out, and select on focus
            so typing replaces the current value rather than appending. */}
        <Input
          placeholder="Pick a user, or type a new id…"
          autoComplete="off"
          data-1p-ignore
          data-lpignore="true"
          onFocus={(event) => event.currentTarget.select()}
        />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover>
        <ListBox items={visible} className="max-h-72 overflow-auto">
          {(option: Option) => (
            <ListBoxItem id={option.id} textValue={option.name}>
              {option.name}
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
      <Description className="text-xs text-[var(--otari-muted)]">{creatingHint}</Description>
    </ComboBox.Root>
  );
}
