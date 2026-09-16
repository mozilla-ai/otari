/**
 * What a price row's rates are per.
 *
 * Every rate column is headed "/ 1M", which is only true of a token rate: a
 * gateway-run tool is priced per million requests and an image endpoint per
 * image. The unit is the lane that keeps those headings honest, so a table of
 * rates shows it beside them.
 */
export const UNIT_LABELS: Record<string, string> = {
  tokens: "tokens",
  requests: "requests",
  images: "images",
}
