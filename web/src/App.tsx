import { RouterProvider } from "@tanstack/react-router";

import { useAuth } from "@/auth/AuthContext";
import { Login } from "@/components/Login";
import { router } from "@/router";

export default function App() {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Login />;
  }

  // Auth gates the router rather than living inside it: signing in is the one
  // decision no route gets to make. The route table and the shell it renders
  // into are in src/routes, wired up in src/router.tsx.
  return <RouterProvider router={router} />;
}
