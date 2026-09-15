import { FiThumbsUp } from "react-icons/fi"
import type { PlaygroundComparisonPreference } from "@/client"
import { Button } from "@/design-system/actions/Button"

/**
 * "Which is better?", under a comparison both models have answered.
 *
 * The bar is what makes a comparison a recorded judgment rather than two
 * answers somebody read: the table it writes to is only worth having if every
 * row is a deliberate press, which is why there is no Save that rates by
 * default.
 *
 * Each control sits under the column it votes for, centred in that column's
 * half, so which model a press is about is read from position rather than from
 * the label. The labels still say it, for anyone not reading positionally.
 *
 * Three verdicts, not two: the stored vocabulary has always had a tie, and the
 * history list renders one, so offering only the two thumbs made an outcome the
 * dataset can hold unreachable from the page that fills it. It sits in the
 * middle, where the question is, because a tie is not about either column.
 */
export function ComparisonRatingBar({
  isAcknowledged,
  isPending,
  onRate,
}: {
  /** Show the confirmation line instead of the controls. */
  isAcknowledged: boolean
  isPending: boolean
  onRate: (preference: PlaygroundComparisonPreference) => void
}) {
  if (isAcknowledged) {
    return (
      <p className="shrink-0 py-2 text-center text-sm text-success">
        Recorded. Thanks for the feedback.
      </p>
    )
  }

  return (
    <div className="flex shrink-0 items-center gap-4 py-2">
      <div className="flex min-w-0 flex-1 justify-center">
        <Button
          size="sm"
          aria-label="Model A answered better"
          onPress={() => onRate("model_a")}
          isDisabled={isPending}
        >
          <FiThumbsUp aria-hidden className="size-4" />
        </Button>
      </div>
      <div className="flex shrink-0 flex-col items-center gap-1">
        <span className="text-caption">Which is better?</span>
        <Button
          size="sm"
          aria-label="Both answered equally well"
          onPress={() => onRate("tie")}
          isDisabled={isPending}
        >
          Tie
        </Button>
      </div>
      <div className="flex min-w-0 flex-1 justify-center">
        <Button
          size="sm"
          aria-label="Model B answered better"
          onPress={() => onRate("model_b")}
          isDisabled={isPending}
        >
          <FiThumbsUp aria-hidden className="size-4" />
        </Button>
      </div>
    </div>
  )
}
