import { Button, Card, Input, Label, Link, TextField } from "@heroui/react";
import { useState } from "react";

import { validateMasterKey } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { ErrorBanner } from "@/components/ui";

export function Login() {
  const { login } = useAuth();
  const [value, setValue] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const submit = async () => {
    const trimmed = value.trim();
    if (!trimmed || isSubmitting) {
      return;
    }
    setIsSubmitting(true);
    setError(null);
    try {
      const valid = await validateMasterKey(trimmed);
      if (valid) {
        login(trimmed);
      } else {
        setError(new Error("Invalid master key."));
      }
    } catch (caught) {
      setError(caught);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <Card.Content className="flex flex-col gap-5 p-7">
          <div className="flex flex-col items-center gap-3 text-center">
            <img src="/favicon.svg" alt="Otari" className="h-12 w-12" />
            <div>
              <h1 className="text-lg font-semibold text-[var(--otari-ink)]">Otari Dashboard</h1>
              <p className="mt-1 text-sm text-[var(--otari-muted)]">
                Sign in with your master key to manage keys, users, and usage.
              </p>
            </div>
          </div>

          <form
            className="flex flex-col gap-4"
            onSubmit={(event) => {
              event.preventDefault();
              void submit();
            }}
          >
            <TextField
              value={value}
              onChange={(next) => {
                setValue(next);
                if (error) {
                  setError(null);
                }
              }}
              type="password"
              isRequired
              className="flex flex-col gap-1"
            >
              <Label className="text-sm font-medium text-[var(--otari-ink)]">Master key</Label>
              <Input placeholder="sk-…" autoFocus autoComplete="off" />
            </TextField>
            <ErrorBanner error={error} />
            <Button type="submit" variant="primary" fullWidth isDisabled={!value.trim() || isSubmitting}>
              {isSubmitting ? "Signing in…" : "Sign in"}
            </Button>
          </form>

          <p className="text-center text-xs text-[var(--otari-muted)]">
            The key is held only in this browser tab (session storage) and sent directly to this gateway.
          </p>

          <div className="border-t border-[var(--otari-line)] pt-4 text-center">
            <Link href="/welcome" className="text-sm font-medium text-[var(--otari-brand-dark)]">
              New to Otari? Open the welcome guide
            </Link>
          </div>
        </Card.Content>
      </Card>
    </div>
  );
}
