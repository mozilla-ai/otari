import { Input, Label, TextField } from "@heroui/react";
import type { ReactNode } from "react";

interface FieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: "text" | "datetime-local";
  isRequired?: boolean;
  description?: ReactNode;
  autoFocus?: boolean;
}

// A labelled single-line text input built from HeroUI's TextField primitives.
export function Field({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  isRequired,
  description,
  autoFocus,
}: FieldProps) {
  return (
    <TextField value={value} onChange={onChange} isRequired={isRequired} className="flex flex-col gap-1">
      <Label className="text-sm font-medium text-[var(--otari-ink)]">
        {label}
        {isRequired ? <span className="text-[var(--otari-brand-dark)]"> *</span> : null}
      </Label>
      <Input type={type} placeholder={placeholder} autoFocus={autoFocus} />
      {description ? <span className="text-xs text-[var(--otari-muted)]">{description}</span> : null}
    </TextField>
  );
}
