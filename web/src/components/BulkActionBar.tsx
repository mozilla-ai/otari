import { Button } from "@heroui/react";
import type { ReactNode } from "react";

// Contextual bar shown when a table has a selection. Reads "{n} selected" with a
// Clear, the action buttons the page supplies, and, when the whole visible page
// is selected and more rows match the filter, a "Select all N matching this
// filter" affordance so a bulk op can target the full filtered set.
//
// Floats fixed near the bottom of the viewport instead of rendering in flow:
// an in-flow bar appeared above the table on first selection and shifted every
// row down under the operator's cursor mid-click-spree, and it scrolled out of
// view on long pages. Fixed positioning gives zero layout shift and keeps the
// actions reachable however deep the selection goes.

export interface BulkActionBarProps {
  /** Rows selected on the current page (expanded "all" sentinel). */
  selectedCount: number;
  /** True once the operator opted into "all matching this filter". */
  allMatching: boolean;
  /** Total rows matching the filter, for the select-all affordance and label. */
  matchingTotal: number | null;
  /** Show the "select all N matching" prompt (page fully selected and more exist). */
  canSelectAllMatching: boolean;
  onSelectAllMatching: () => void;
  onClear: () => void;
  /** Action buttons (Delete, Set price, …). */
  children: ReactNode;
}

export function BulkActionBar({
  selectedCount,
  allMatching,
  matchingTotal,
  canSelectAllMatching,
  onSelectAllMatching,
  onClear,
  children,
}: BulkActionBarProps) {
  const label = allMatching
    ? `All ${(matchingTotal ?? selectedCount).toLocaleString()} matching rows selected`
    : `${selectedCount.toLocaleString()} selected`;

  return (
    <div
      role="toolbar"
      aria-label="Bulk actions"
      className="otari-bulk-bar fixed bottom-4 left-1/2 z-40 flex w-[calc(100%-2rem)] max-w-3xl -translate-x-1/2 flex-wrap items-center gap-3 rounded-xl border border-[var(--otari-brand)] bg-[var(--otari-surface)] px-4 py-2.5 shadow-lg"
    >
      <span className="text-sm font-medium text-[var(--otari-brand-dark)]">{label}</span>
      {!allMatching && canSelectAllMatching && matchingTotal != null ? (
        <Button size="sm" variant="ghost" onPress={onSelectAllMatching}>
          Select all {matchingTotal.toLocaleString()} matching this filter
        </Button>
      ) : null}
      <div className="ml-auto flex flex-wrap items-center gap-2">
        {children}
        <Button size="sm" variant="ghost" onPress={onClear}>
          Clear
        </Button>
      </div>
    </div>
  );
}
