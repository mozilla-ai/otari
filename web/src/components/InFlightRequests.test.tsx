import { render, screen } from "@testing-library/react";
import { act } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { InFlightRequest } from "@/api/types";
import { InFlightRequests } from "@/components/InFlightRequests";

function request(overrides: Partial<InFlightRequest> = {}): InFlightRequest {
  return {
    id: "live-1",
    endpoint: "/v1/chat/completions",
    model: "ollama:qwen3",
    provider: "ollama",
    user_id: "alice",
    api_key_id: "key-1",
    policy_name: null,
    started_at: "2026-08-10T12:00:00Z",
    elapsed_ms: 5_000,
    ...overrides,
  };
}

describe("InFlightRequests", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders nothing when nothing is in flight", () => {
    const { container } = render(<InFlightRequests requests={[]} total={0} updatedAt={Date.now()} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("advances the elapsed time between polls", () => {
    // The server's elapsed_ms is only as fresh as the last poll (2s apart), so a
    // wait that only moved when a response landed would read as a stalled number
    // on the one screen whose whole job is showing that something is still going.
    vi.useFakeTimers();
    const updatedAt = Date.now();
    render(<InFlightRequests requests={[request({ elapsed_ms: 5_000 })]} total={1} updatedAt={updatedAt} />);

    expect(screen.getByText("5s")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(3_000);
    });

    expect(screen.getByText("8s")).toBeInTheDocument();
  });

  it("never shows a negative wait when a poll lands after a tick", () => {
    // `updatedAt` can be marginally ahead of the tick's `Date.now()`; without the
    // clamp that renders as a request that started in the future.
    vi.useFakeTimers();
    render(<InFlightRequests requests={[request({ elapsed_ms: 0 })]} total={1} updatedAt={Date.now() + 5_000} />);

    expect(screen.getByText("0s")).toBeInTheDocument();
  });

  it("keeps the counted total when the served list is capped", () => {
    render(<InFlightRequests requests={[request()]} total={4} updatedAt={Date.now()} />);

    expect(screen.getByText("4 requests in flight")).toBeInTheDocument();
    expect(screen.getByText("showing the 1 longest-running")).toBeInTheDocument();
  });

  it("drops the cap note when the list is complete", () => {
    render(<InFlightRequests requests={[request()]} total={1} updatedAt={Date.now()} />);

    expect(screen.getByText("1 request in flight")).toBeInTheDocument();
    expect(screen.queryByText(/longest-running/)).not.toBeInTheDocument();
  });
});
