/**
 * The Otari mark, inline so it can take its color from the text around it.
 *
 * `currentColor` and not a fill of its own: the accent has already moved once
 * in this product's life, and a copy of its value inside an asset is a copy
 * nobody finds when it moves again. A call site sets the color the way it sets
 * any other ink, with a text utility.
 *
 * The mark is the one further sanctioned use of the accent beyond data ink,
 * graphic fills and the status dot. That is a licence for this one thing, not
 * a loosening of the rule.
 *
 * 273 by 250, which is not square, so it is given a width and left to find its
 * own height. Forcing it into a square box stretches it about 9%, which is the
 * kind of wrong that reads as "something is off" without anybody being able to
 * say what.
 *
 * A caller has to pass `h-auto` with its width, and that is not pedantry: a
 * width alone left this rendering 28 by 16 inside a flex row, a ratio of 1.75
 * against the artwork's 1.09, because an `<svg>` is a replaced element and its
 * height came from the box rather than from the viewBox. Measured, not assumed.
 *
 * `fill-rule="evenodd"` is here for intent rather than for repair, and that is
 * worth saying because it was added expecting to fix something. The small shape
 * at the top right is a counter, a hole and not a dot, and a compound path can
 * lose a counter under the default `nonzero` rule. This one does not: rendered
 * side by side at 180px under both rules, the counter is a hole either way,
 * because its subpath is wound against the shape around it. The rule stays
 * because the mark's holes are holes by construction and saying so costs
 * nothing, not because removing it would break the artwork.
 */
export function ProductMark({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 273 250"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-hidden="true"
      focusable="false"
      className={className}
    >
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M164.338 214.396C164.706 213.589 165.069 212.785 165.424 211.989C167.697 206.885 169.694 202.029 171.446 197.474C180.615 173.642 183.065 158.027 183.087 157.885L165.645 187.422L164.75 188.922L149.823 214.778C149.529 215.288 149.221 215.805 148.915 216.305C148.718 216.627 148.517 216.949 148.315 217.268C146.076 220.799 143.563 224.061 140.818 227.041C136.254 231.996 131.052 236.174 125.417 239.517C104.732 251.788 78.201 252.828 55.89 239.947L56.138 239.517L82.905 193.153C86.775 186.452 91.603 180.647 97.115 175.813C95.998 175.929 85.362 177.967 56.612 213.759L56.58 213.802L56.425 213.993L56.438 213.663C25.15 196.444 3.284 164.217 0.554 126.677C0.189 124.473 0 122.209 0 119.901L0 26.599C22.738 26.599 41.171 45.03 41.171 67.769C41.171 45.03 59.595 26.599 82.333 26.599L82.333 77.714C82.333 88.784 77.963 98.835 70.855 106.234L69.576 107.505C63.053 113.728 54.491 117.833 44.993 118.71L124.948 118.71L154.895 66.743C125.874 59.437 104.39 33.169 104.389 1.881L104.389 0L193.598 0.02C219.712 -0.019 240.934 21.142 240.935 47.288L240.935 134.572C240.934 142.999 239.806 151.163 237.693 158.921C236.494 163.324 234.977 167.597 233.17 171.713C223.865 192.902 206.842 209.935 185.658 219.252C181.625 221.026 177.441 222.52 173.131 223.71C167.946 225.141 162.578 226.132 157.07 226.641C158.32 224.831 159.506 222.954 160.625 221.016L163.959 215.241C164.018 215.108 164.08 214.973 164.142 214.835C164.169 214.774 164.196 214.712 164.224 214.65C164.261 214.567 164.299 214.482 164.338 214.396ZM241.987 182.229L242.71 183.507L242.704 183.529C243.272 184.418 243.832 185.319 244.364 186.241L271.383 233.036C246.73 247.269 216.925 244.503 195.547 228.374C195.685 228.304 195.826 228.234 195.964 228.164L195.489 227.837C195.523 227.819 219.193 215.667 229.097 191.177C232.122 183.696 233.817 176.917 234.618 173.188L235.064 170.966C235.157 170.462 235.199 170.185 235.201 170.173L241.987 182.229ZM171.211 12.059C165.466 12.059 160.811 16.717 160.811 22.462C160.811 28.206 165.466 32.862 171.211 32.863C176.956 32.863 181.615 28.207 181.615 22.462C181.615 16.716 176.956 12.059 171.211 12.059Z"
      />
    </svg>
  )
}
