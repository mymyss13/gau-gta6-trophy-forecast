// GAU GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).
//
// Model: a trophy's earn rate is earned-count / players-tracked, so forecast the
// two counts separately and divide. Both counts grow roughly log-linearly in the
// launch week: players flood in (dilution pushes every rate down) while players
// keep earning trophies (pushing rates up). A log-linear least-squares fit of
// log(players) and log(earned_i) against time, extrapolated to the target with
// the move damped toward the last read, gives the median; the spread adds the
// house's random-walk term, the trend fit's own uncertainty, and an allowance
// for Exophase refresh jumps (a single refresh on launch night multiplied the
// tracked players ~6x, so a history that shows no refresh at all is not proof
// none will come: flat histories get the widest spread and a downward prior).
//
// Node 22, standard library only. Reads every <id>.in in the input directory
// and writes one <id>.out per case. Deterministic: no randomness anywhere.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk spread, logit units per square-root day (the house's number). */
const SIGMA = 0.5;
/** How much of the extrapolated log move to keep (1 = full trend, 0 = flat). */
const DAMP = 0.7;
/** Largest logit move allowed away from the last read, either direction. */
const MAX_MOVE = 2.5;
/** Extra logit spread for a refresh jump the history has not shown. */
const JUMP_FLAT = 1.0;
/** Extra logit spread for refreshes between the last read and the target.
    Tuned to 0 on synthetic worlds: the random-walk term plus the trend fit's
    own residual already covers the moves, and extra width only costs pinball
    at every trophy. (Flat histories keep their own wider allowance.) */
const JUMP_SEEN = 0;
/** Downward logit prior when the history is perfectly flat (no refresh seen). */
const FLAT_SHIFT = -0.5;

interface Trophy {
  id: number;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  levels: number[];
  horizon_minutes: number;
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const logistic = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Weighted least squares of y on x (recency weights: the launch flood's drift
    is not the drift that matters at the target); returns {a, b, residSd, lev}. */
function ols(xs: number[], ys: number[], ws: number[], xStar: number): { a: number; b: number; residSd: number; lev: number } {
  const sw = ws.reduce((s, v) => s + v, 0);
  const mx = xs.reduce((s, v, i) => s + ws[i] * v, 0) / sw;
  const my = ys.reduce((s, v, i) => s + ws[i] * v, 0) / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - mx) * (xs[i] - mx);
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < xs.length; i++) ss += ws[i] * (ys[i] - (a + b * xs[i])) ** 2;
  const n = xs.length;
  const residSd = n > 2 ? Math.sqrt(ss / (sw * (n - 2) / n)) : 0;
  const lev = sxx > 1e-12 ? 1 + ((xStar - mx) ** 2) / sxx : 1;
  return { a, b, residSd, lev };
}

/** Recency weight for a read at hour x, half-life 24h from the last read. */
const recency = (x: number, xLast: number): number => Math.exp(-(xLast - x) / 24);

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const dDays = c.horizon_minutes / 1440;
  const t0 = Date.parse(c.history[0].at);
  const tStar = Date.parse(c.history[c.history.length - 1].at) + c.horizon_minutes * 60_000;
  const hours = (t: number): number => (t - t0) / 3_600_000;
  const xStar = hours(tStar);

  const xLast = hours(Date.parse(c.history[c.history.length - 1].at));
  // Players-tracked trend, from reads with a positive count.
  const pPts = c.history
    .filter((r) => r.players_tracked > 0)
    .map((r) => ({ x: hours(Date.parse(r.at)), y: Math.log(r.players_tracked) }));
  const playersFlat = new Set(pPts.map((p) => p.y)).size <= 1;
  const pFit = pPts.length >= 2 ? ols(pPts.map((p) => p.x), pPts.map((p) => p.y), pPts.map((p) => recency(p.x, xLast)), xStar) : null;
  const logPStar = pFit ? pFit.a + pFit.b * xStar : Math.log(c.history[c.history.length - 1].players_tracked || 1);

  const out = c.trophies.map((t, i) => {
    const last = c.history[c.history.length - 1];
    const lastRate = last.rates[i] ?? 0;
    // Earned-count series for this trophy (half a player stands in for zero).
    const ePts: Array<{ x: number; y: number }> = [];
    for (const r of c.history) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined || r.players_tracked <= 0) continue;
      const earned = Math.max((rate / 100) * r.players_tracked, 0.5);
      ePts.push({ x: hours(Date.parse(r.at)), y: Math.log(earned) });
    }
    const ratesFlat = new Set(ePts.map((p) => p.y)).size <= 1;
    let medianLogit = logit(lastRate);
    let trendSd = 0;
    if (ePts.length >= 3 && !ratesFlat) {
      const f = ols(ePts.map((p) => p.x), ePts.map((p) => p.y), ePts.map((p) => recency(p.x, xLast)), xStar);
      const logEStar = f.a + f.b * xStar;
      const rawRate = 100 * Math.exp(logEStar - logPStar);
      const rawLogit = logit(rawRate);
      const move = rawLogit - medianLogit;
      medianLogit += DAMP * Math.max(-MAX_MOVE, Math.min(MAX_MOVE, move));
      trendSd = f.residSd * Math.sqrt(f.lev);
    } else if (playersFlat && ratesFlat) {
      // No refresh has ever shown in this history: expect dilution, widen hard.
      medianLogit += FLAT_SHIFT;
    }
    const rw = SIGMA * Math.sqrt(dDays);
    const jump = playersFlat && ratesFlat ? JUMP_FLAT : JUMP_SEEN;
    const spread = Math.sqrt(rw * rw + trendSd * trendSd + jump * jump);
    const q = Z.map((z) => {
      const v = round4(logistic(medianLogit + z * spread));
      return Math.min(100, Math.max(0, v));
    });
    // Keep the row non-decreasing after rounding/clamping.
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: t.id, q };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: out })}\n`);
}
