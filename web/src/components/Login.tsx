import { Button, Card, Input, Label, TextField } from "@heroui/react";
import { useState } from "react";

import { useAuth } from "@/auth/AuthContext";

export function Login() {
  const { login } = useAuth();
  const [value, setValue] = useState("");

  const submit = () => {
    if (value.trim()) {
      login(value);
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
              submit();
            }}
          >
            <TextField
              value={value}
              onChange={setValue}
              type="password"
              isRequired
              className="flex flex-col gap-1"
            >
              <Label className="text-sm font-medium text-[var(--otari-ink)]">Master key</Label>
              <Input placeholder="sk-…" autoFocus autoComplete="off" />
            </TextField>
            <Button type="submit" variant="primary" fullWidth isDisabled={!value.trim()}>
              Sign in
            </Button>
          </form>

          <p className="text-center text-xs text-[var(--otari-muted)]">
            The key is held only in this browser tab (session storage) and sent directly to this gateway.
          </p>
        </Card.Content>
      </Card>
    </div>
  );
}
