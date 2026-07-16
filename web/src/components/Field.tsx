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
      {/* No manual "*": HeroUI marks a required field's label through CSS
          ([data-required=true] > .label::after), so adding one renders two. */}
      <Label className="text-sm font-medium text-[var(--otari-ink)]">{label}</Label>
      <Input type={type} placeholder={placeholder} autoFocus={autoFocus} />
      {description ? <span className="text-xs text-[var(--otari-muted)]">{description}</span> : null}
    </TextField>
  );
}
