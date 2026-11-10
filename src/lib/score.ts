import type { Metric, RetrievalResult } from "./api";

/** Metrics where a smaller number is better. */
export const LOWER_IS_BETTER = new Set(["noise_sensitivity", "latency"]);

/** Quality metrics that count toward the overall score on this dataset. */
export function scoredMetrics(r: RetrievalResult, labeled: boolean): Metric[] {
  return r.metrics.filter((m) => !m.pending && (labeled || !m.referenceBased));
}

/**
 * Overall quality for a run: the mean of its usable metrics, with
 * lower-is-better metrics inverted. Reference-based metrics only count on
 * labeled datasets; pending (not yet computed) metrics are skipped. Latency
 * is reported separately, not scored.
 */
export function overallScore(r: RetrievalResult, labeled: boolean): number {
  const ms = scoredMetrics(r, labeled);
  if (ms.length === 0) return 0;
  return ms.reduce((a, m) => a + (LOWER_IS_BETTER.has(m.key) ? 1 - m.value : m.value), 0) / ms.length;
}
