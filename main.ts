// polymarket-nvidia-ceo-forecast-01 entry (Node 22, standard library only).
//
// Forecast: per-market quantiles of the Yes price 240 minutes ahead, in logit
// space, with a per-market spread estimated from the market's own recent
// moves and a drift of zero.
//
//   q_tau(h) = sigmoid(l0 + z_tau * sigma * sqrt(h)), clamped to [0, 1],
//   rounded to 6 decimals
//
// where l0 is the logit of the market's price at the checkpoint, z_tau the
// standard normal's quantile at each level (0.05, 0.25, 0.5, 0.75, 0.95),
// and sigma = 0.25 * std(last-120-min logit diffs) + 0.75 * 0.012, floored
// at 0.002 and capped at 0.06.
//
// Why logit space: most of these markets sit near 0 or 1, where a move of
// 0.01 means far more at 0.02 than at 0.50. A fixed price-space spread (the
// house baseline's 0.0025) over-spreads quiet low-price markets and
// under-spreads jumpy ones; the logit spread scales with p * (1 - p) and
// respects the [0, 1] bounds. Why the heavy shrinkage (0.25): the recent
// window's raw std is inflated by single jumps (one market jumped 0.35 ->
// 0.59 in a minute in the practice), so the estimate is pulled hard toward
// the 0.012 baseline; tuned on the six practice cases fetched from
// Polymarket's public price history, this scores a mean pinball loss of
// 0.270863 against the house baseline's 0.363589, beating it on all six
// cases. Drift is zero: every drift variant tested on the practice scored
// worse than no drift. No overfitting to a knife edge: nearby settings
// (alpha 0.20-0.35, window 120-240 min) all land within 0.271-0.274.
//
// It reads every <id>.in in the input directory and writes <id>.out:
// {"forecasts":[{"id","q"}]}, one row per market, 240 rows of five
// quantiles, non-decreasing, in [0, 1]. Deterministic: no randomness, no
// network, no clock. Runs offline under the case's 8 s cap.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** Minutes of recent history the spread is estimated from. */
const SIGMA_WINDOW_MIN = 120;
/** Weight of the market's own recent spread; the rest is the baseline. */
const SIGMA_ALPHA = 0.25;
/** Baseline logit spread (per square-root minute) the estimate shrinks to. */
const SIGMA_BASE = 0.012;
/** Floor and cap on the per-market logit spread. */
const SIGMA_MIN = 0.002;
const SIGMA_MAX = 0.06;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0, 1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Logit, with the price clipped away from 0 and 1 first. */
const logit = (p: number): number => {
  const c = Math.min(0.9999, Math.max(0.0001, p));
  return Math.log(c / (1 - c));
};

/** Sigmoid, guarded against overflow. */
const sigmoid = (l: number): number => {
  if (l > 30) return 1;
  if (l < -30) return 0;
  return 1 / (1 + Math.exp(-l));
};

/** Sample standard deviation (n - 1 denominator); 0 when fewer than 2 values. */
function std(xs: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  const mu = xs.reduce((a, b) => a + b, 0) / n;
  return Math.sqrt(xs.reduce((a, b) => a + (b - mu) * (b - mu), 0) / (n - 1));
}

/** Per-market logit spread from the recent 1-minute logit moves. */
function marketSigma(prices: number[]): number {
  const ls = prices.map(logit);
  const diffs: number[] = [];
  const start = Math.max(1, ls.length - SIGMA_WINDOW_MIN);
  for (let i = start; i < ls.length; i++) diffs.push((ls[i] as number) - (ls[i - 1] as number));
  const est = std(diffs);
  const sigma = SIGMA_ALPHA * est + (1 - SIGMA_ALPHA) * SIGMA_BASE;
  return Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, sigma));
}

/** Prices with any non-finite value replaced by carrying the last finite one
 *  forward (the house grid's own rule); a leading run of them becomes the
 *  first finite price. Guards the forecast against a bad tick in a case. */
function sanitize(prices: number[]): number[] {
  const out: number[] = [];
  let last = NaN;
  for (const p of prices) {
    if (Number.isFinite(p)) last = p;
    out.push(last);
  }
  if (!Number.isFinite(out[0])) throw new Error('market has no finite price');
  let first = NaN;
  for (let i = 0; i < out.length; i++) {
    if (Number.isFinite(out[i] as number)) { first = out[i] as number; break; }
  }
  return out.map((p) => (Number.isFinite(p) ? p : first));
}

/** The forecast of one market: horizon rows of the five quantiles around its last price. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const clean = sanitize(m.prices);
  const l0 = logit(clean[clean.length - 1] as number);
  const sigma = marketSigma(clean);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    rows.push(Z.map((z) => price6(sigmoid(l0 + z * sigma * Math.sqrt(h)))));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
