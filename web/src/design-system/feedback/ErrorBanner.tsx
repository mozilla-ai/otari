import { errorMessage } from "./errorMessage"

export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) {
    return null
  }
  return (
    <div
      role="alert"
      className="rounded-lg border border-danger bg-danger-subtle px-4 py-3 text-sm text-danger"
    >
      {errorMessage(error)}
    </div>
  )
}
