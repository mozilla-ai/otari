import { useEffect, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Button } from "./ui";

export interface StepDef {
  id: string;
  title: string;
}

/**
 * Shared chrome for the two step-by-step walkthroughs: a header, the progress
 * rail, the animated step body, Back/Next navigation, and arrow-key support. It
 * owns the current step index; the body is a render function of that step.
 */
export function StepWalkthrough({
  title,
  description,
  steps,
  footer,
  children,
}: {
  title: string;
  description: ReactNode;
  steps: readonly StepDef[];
  footer?: ReactNode;
  children: (step: number) => ReactNode;
}) {
  const [step, setStep] = useState(0);
  const last = steps.length - 1;
  const next = () => setStep((s) => Math.min(last, s + 1));
  const prev = () => setStep((s) => Math.max(0, s - 1));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") next();
      else if (e.key === "ArrowLeft") prev();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        <p className="text-sm text-muted">{description}</p>
      </div>

      {/* Progress rail */}
      <div className="flex items-center gap-2">
        {steps.map((s, i) => (
          <button
            key={s.id}
            type="button"
            onClick={() => setStep(i)}
            aria-label={`Step ${i + 1}: ${s.title}`}
            className={`h-1.5 flex-1 rounded-full transition-colors ${i <= step ? "bg-accent" : "bg-surface-tertiary"}`}
          />
        ))}
      </div>
      <div className="-mt-2 text-xs font-medium uppercase tracking-wide text-muted">
        Step {step + 1} of {steps.length} · {steps[step].title}
      </div>

      {/* Animated step body */}
      <div className="min-h-[24rem]">
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
          >
            {children(step)}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Nav */}
      <div className="flex items-center justify-between">
        <Button variant="ghost" onPress={prev} isDisabled={step === 0}>
          ← Back
        </Button>
        <span className="text-xs text-muted">{footer}</span>
        {step < last ? (
          <Button variant="primary" onPress={next}>
            Next →
          </Button>
        ) : (
          <Button variant="ghost" onPress={() => setStep(0)}>
            ↺ Start over
          </Button>
        )}
      </div>
    </div>
  );
}

/**
 * Two-column step layout: narrative lede on the left, content on the right.
 * `aside` renders in the left column beneath the lede.
 */
export function StepGrid({
  lede,
  aside,
  children,
  wide,
}: {
  lede: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <div className={`grid gap-5 ${wide ? "" : "lg:grid-cols-[1fr_1.1fr]"} lg:items-start`}>
      <div className="flex flex-col gap-4">
        <p className="text-[0.95rem] leading-relaxed text-foreground/90">{lede}</p>
        {aside}
      </div>
      <div>{children}</div>
    </div>
  );
}
