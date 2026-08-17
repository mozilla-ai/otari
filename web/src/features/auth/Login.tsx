import { Button, Card, Input, Label, Link, TextField } from "@heroui/react"
import { useState } from "react"
import { useAuth } from "@/features/auth/AuthContext"
import { createSession } from "@/shared/api/client"
import { ErrorBanner } from "@/shared/components/ui"

export function Login() {
  const { login, isSigningOut } = useAuth()
  const [value, setValue] = useState("")
  const [error, setError] = useState<unknown>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const submit = async () => {
    const trimmed = value.trim()
    // isSigningOut blocks a new sign-in until a prior sign-out's server-side
    // revocation has finished (or timed out): otherwise its expiring cookie
    // could land after this one mints a fresh session and clobber it (#557).
    if (!trimmed || isSubmitting || isSigningOut) {
      return
    }
    setIsSubmitting(true)
    setError(null)
    try {
      const valid = await createSession(trimmed)
      if (valid) {
        login()
      } else {
        setError(new Error("Invalid master key."))
      }
    } catch (caught) {
      setError(caught)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <Card.Content className="flex flex-col gap-5 p-7">
          <div className="flex flex-col items-center gap-3 text-center">
            <img src="/favicon.svg" alt="Otari" className="h-12 w-12" />
            <div>
              <h1 className="text-lg font-semibold text-foreground">
                Otari Dashboard
              </h1>
              <p className="mt-1 text-sm text-muted">
                Sign in with your master key to browse models, set pricing, and
                manage settings.
              </p>
            </div>
          </div>

          <form
            className="flex flex-col gap-4"
            onSubmit={(event) => {
              event.preventDefault()
              void submit()
            }}
          >
            <TextField
              value={value}
              onChange={(next) => {
                setValue(next)
                if (error) {
                  setError(null)
                }
              }}
              type="password"
              isRequired
              className="flex flex-col gap-1"
            >
              <Label className="text-sm font-medium text-foreground">
                Master key
              </Label>
              <Input
                placeholder="otari-mk-… or your master key"
                autoFocus
                autoComplete="off"
              />
            </TextField>
            <details className="text-xs text-muted">
              <summary className="cursor-pointer font-medium text-link hover:text-link-hover">
                First run? Where to find your key
              </summary>
              <p className="mt-2 leading-relaxed">
                If you did not set <code>OTARI_MASTER_KEY</code>, Otari
                generated one and printed it to the server logs on startup. Look
                for the line <code>Your master key:</code> (for example, run{" "}
                <code>docker logs &lt;container&gt;</code>) and paste it above.
              </p>
            </details>
            <ErrorBanner error={error} />
            <Button
              type="submit"
              variant="primary"
              fullWidth
              isDisabled={!value.trim() || isSubmitting || isSigningOut}
            >
              {isSigningOut
                ? "Finishing sign-out…"
                : isSubmitting
                  ? "Signing in…"
                  : "Sign in"}
            </Button>
          </form>

          <p className="text-center text-xs text-muted">
            The key is sent once to this gateway and exchanged for a session
            cookie; it is never stored in the browser.
          </p>

          <div className="border-t border-border pt-4 text-center">
            <Link
              href="/welcome"
              className="text-sm font-medium text-link hover:text-link-hover"
            >
              New to Otari? Open the welcome guide
            </Link>
          </div>
        </Card.Content>
      </Card>
    </div>
  )
}
