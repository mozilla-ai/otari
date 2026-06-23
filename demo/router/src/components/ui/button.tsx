import type { ComponentProps } from "react";
import { Button as HeroButton } from "@heroui/react";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "outline";

/**
 * App Button wrapper over HeroUI's Button, mirroring the clawbolt convention of
 * routing pages through a small semantic wrapper rather than the library
 * component directly. Filled variants get a subtle elevation; everything else
 * passes straight through (onPress, size, isDisabled, fullWidth, ...).
 */
export function Button({
  variant = "primary",
  className,
  ...props
}: { variant?: Variant } & Omit<ComponentProps<typeof HeroButton>, "variant">) {
  const elevated = variant === "primary" || variant === "secondary" || variant === "danger";
  const merged = [elevated ? "shadow-sm" : "", className].filter(Boolean).join(" ") || undefined;
  return <HeroButton variant={variant} className={merged} {...props} />;
}
