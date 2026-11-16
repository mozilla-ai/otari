import { play, type SoundName, unlock } from "cuelume"

/**
 * The picker's sounds, played through cuelume, which synthesizes each one with
 * Web Audio. One material for the whole picker, its water theme, every cue at
 * its quietest arrangement: the picker is the one playful moment in the
 * dashboard, and it should still sound like part of it.
 */
export type PickerMoment =
  | "open"
  | "pass"
  | "land"
  | "key"
  | "shuffle"
  | "save"
  | "cancel"

const CUES: Record<PickerMoment, SoundName> = {
  open: "open",
  // A face sliding through the frame, the face that settles there, and an
  // arrow key's step are all the same act: choosing the next option.
  pass: "select",
  land: "select",
  key: "select",
  shuffle: "navigate",
  save: "success",
  cancel: "close",
}

const VOLUME = 0.8

// A flick carries about twenty faces a second through the frame. One tick per
// interval is enough to feel each one go by without a buzz.
const PASS_GAP_MS = 45
let lastPassAt = Number.NEGATIVE_INFINITY

/**
 * Plays the cue for `moment`. A later face sounds a hair higher than an
 * earlier one, so `direction` is which way the frame moved.
 */
export function playPickerSound(
  moment: PickerMoment,
  direction?: "forward" | "back",
) {
  if (moment === "pass") {
    const now = performance.now()
    if (now - lastPassAt < PASS_GAP_MS) return
    lastPassAt = now
  }
  play(CUES[moment], {
    theme: "bubble",
    emphasis: "subtle",
    volume: VOLUME,
    direction,
  })
}

/**
 * Opens the picker's audio. Call it from the press that opens the picker: some
 * browsers start audio only inside a gesture, and a scroll is not one.
 */
export function startPickerSound() {
  void unlock()
  playPickerSound("open")
}
