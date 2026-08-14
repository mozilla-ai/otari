import { RouterProvider } from "@tanstack/react-router";

import { useAuth } from "@/auth/AuthContext";
import { Login } from "@/components/Login";
import { router } from "@/router";

export default function App() {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Login />;
  }

  // The route table, the shell it renders into (AppShell, the root route's
  // component) and the per-page code splitting all live in src/routes and are
  // wired up in src/router.tsx. What is left here is the one thing routing does
  // not decide: an unauthenticated visitor sees the sign-in screen instead of
  // any route at all.
  return <RouterProvider router={router} />;
}
