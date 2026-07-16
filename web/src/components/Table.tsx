import { Spinner } from "@heroui/react";
import type { ReactNode } from "react";

// Lightweight semantic table primitives. HeroUI v3 ships a full react-aria
// Table, but the dashboard's tables are read-mostly and don't need selection or
// keyboard grid navigation, so a styled <table> keeps the markup simple.

export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--otari-line)] bg-[var(--otari-surface)]">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  );
}

export function THead({ children }: { children: ReactNode }) {
  return (
    <thead className="border-b border-[var(--otari-line)] bg-[var(--otari-brand-tint)] text-left">
      {children}
    </thead>
  );
}

export function Th({
  children,
  className = "",
  ariaSort,
}: {
  children: ReactNode;
  className?: string;
  // Announces sort state to assistive tech on a sortable column header.
  ariaSort?: "ascending" | "descending" | "none";
}) {
  return (
    <th
      aria-sort={ariaSort}
      className={`px-4 py-2.5 font-semibold text-[var(--otari-ink)] whitespace-nowrap ${className}`}
    >
      {children}
    </th>
  );
}

export function Td({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <td className={`px-4 py-2.5 align-middle ${className}`}>{children}</td>;
}

export function Tr({
  children,
  className = "",
  onClick,
  selected,
}: {
  children: ReactNode;
  className?: string;
  // When set, the row is clickable (used for row-selection tables).
  onClick?: () => void;
  selected?: boolean;
}) {
  return (
    <tr
      onClick={onClick}
      // A clickable row is the only path to the detail panel, so it must be
      // reachable and operable by keyboard (WCAG 2.1.1), not just the mouse.
      onKeyDown={
        onClick
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onClick();
              }
            }
          : undefined
      }
      tabIndex={onClick ? 0 : undefined}
      aria-selected={selected}
      className={`border-b border-[var(--otari-line)] last:border-b-0 ${
        onClick
          ? "cursor-pointer focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-[var(--otari-brand)]"
          : ""
      } ${selected ? "bg-[var(--otari-brand-tint)]" : "hover:bg-[var(--otari-bg)]"} ${className}`}
    >
      {children}
    </tr>
  );
}

export function TableMessage({ colSpan, children }: { colSpan: number; children: ReactNode }) {
  return (
    <tr>
      <td colSpan={colSpan} className="px-4 py-10 text-center text-[var(--otari-muted)]">
        {children}
      </td>
    </tr>
  );
}

export function LoadingRow({ colSpan }: { colSpan: number }) {
  return (
    <TableMessage colSpan={colSpan}>
      <span className="inline-flex items-center gap-2">
        <Spinner size="sm" /> Loading…
      </span>
    </TableMessage>
  );
}
