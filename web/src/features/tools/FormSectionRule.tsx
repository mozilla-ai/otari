/**
 * A hairline rule with a caption over it, dividing one dialog form into the
 * parts an operator decides separately.
 *
 * The negative margin closes the form's own gap below the rule, so the caption
 * sits with the fields it introduces rather than floating between two groups.
 */
export function FormSectionRule({ label }: { label: string }) {
  return (
    <div className="-mb-2 border-border-subtle border-t pt-4">
      <span className="text-mono-overline text-subtle">{label}</span>
    </div>
  )
}
