import { FiCheck } from "react-icons/fi"

/** The success check beside a payoff's title. */
export function SetupSuccessMark() {
  return (
    <span className="mt-0.5 flex shrink-0">
      <FiCheck aria-hidden className="text-success size-6" />
    </span>
  )
}
