/**
 * The operator-facing text for something that was thrown.
 *
 * One branch, where the application's own version had two: it tested
 * `ApiError` before `Error` and returned `error.message` from both, and
 * `ApiError extends Error`, so the first branch could never change an answer.
 * Dropping it is what let this module leave `src/shared`, since naming
 * `ApiError` meant importing the transport.
 *
 * A thrown value that is not an `Error` deliberately does not reach the screen.
 * It is usually a rejected fetch body or a library's bare string, and putting
 * either in front of an operator is how internals leak into a UI.
 */
export function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  return "Something went wrong."
}
