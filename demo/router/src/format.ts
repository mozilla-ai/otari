// Map a 0..1 quality score to a status text color, shared by both walkthroughs.
export const qualClass = (s: number) => (s >= 0.85 ? "text-success" : s >= 0.5 ? "text-warning" : "text-danger");
