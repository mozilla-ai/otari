import type { ComponentProps } from "react";
import { Card } from "@heroui/react";

/** A padded HeroUI Card, the default surface for app panels. */
export function Panel({ className, ...props }: ComponentProps<typeof Card>) {
  return <Card className={["p-4", className].filter(Boolean).join(" ")} {...props} />;
}
