/** A 6px square. The page's one status mark, in every place it appears. */
export function Dot({ className }: { className: string }) {
  return <span aria-hidden className={`h-1.5 w-1.5 shrink-0 ${className}`} />
}
