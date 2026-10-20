// A number in a lane sized for "999,999" or "$9,999.99", with `detail` (the
// cache split) as its title and read after it. Past the lane's width it
// shortens ("1.23B", "$12.3M"); the exact figure then leads the title and is
// all a screen reader hears, so the shortening is visual alone.
export function LaneValue({
  display,
  exact,
  detail,
}: {
  display: string
  exact: string
  detail?: string
}) {
  if (display === exact) {
    return (
      <span title={detail}>
        {display}
        {detail ? <span className="sr-only">, {detail}</span> : null}
      </span>
    )
  }
  return (
    <span title={detail ? `${exact}: ${detail}` : exact}>
      <span aria-hidden>{display}</span>
      <span className="sr-only">
        {exact}
        {detail ? `, ${detail}` : ""}
      </span>
    </span>
  )
}
