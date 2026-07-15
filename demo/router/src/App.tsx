import { useState } from "react";
import { Chip } from "@heroui/react";
import { Walkthrough } from "./components/Walkthrough";
import { TeachWalkthrough } from "./components/TeachWalkthrough";
import type { DemoData } from "./types";
import rawDemo from "./demo_prompts.json";

// Fully offline demo: everything runs against the bundled prompts, answers, and
// embeddings in demo_prompts.json. No gateway connection, no OpenAI at runtime.
const demo = rawDemo as unknown as DemoData;

const VIEWS = [
  ["teach", "How it learns"],
  ["showcase", "See it route"],
] as const;

export function App() {
  const [view, setView] = useState<"teach" | "showcase">("teach");

  return (
    <div className={`mx-auto px-5 pb-24 pt-6 ${view === "showcase" ? "max-w-6xl" : "max-w-5xl"}`}>
      <header className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">Otari Router Demo</h1>
          <p className="text-sm text-muted">Learn which prompts a cheaper model handles just as well, then route them there.</p>
        </div>
        <Chip size="sm">offline demo</Chip>
      </header>

      {/* Segmented view switch (selected state is explicitly highlighted). */}
      <div className="mb-4 inline-flex rounded-xl border border-border p-0.5 text-sm">
        {VIEWS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setView(key)}
            aria-pressed={view === key}
            className={`rounded-lg px-4 py-1.5 font-medium transition ${
              view === key ? "bg-accent text-accent-foreground shadow-sm" : "text-muted hover:text-foreground"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {view === "teach" && <TeachWalkthrough demo={demo} onSeeItRoute={() => setView("showcase")} />}
      {view === "showcase" && <Walkthrough demo={demo} />}
    </div>
  );
}
