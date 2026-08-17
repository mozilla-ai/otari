import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "@/app/App";
import { Provider } from "@/app/provider";
import "@/styles/globals.css";

const container = document.getElementById("root");
if (!container) {
  throw new Error("Root element #root not found");
}

createRoot(container).render(
  <StrictMode>
    <Provider>
      <App />
    </Provider>
  </StrictMode>,
);
