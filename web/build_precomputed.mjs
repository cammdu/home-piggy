// Precompute the default saver's results so the page is complete before any computation runs.
// Uses exactly the functions and run sizes the page uses (model.js CONFIG), so the live
// recomputation for the same inputs returns the same numbers.
// Usage: node web/build_precomputed.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { DEFAULT_SAVER } from "./engine.js";
import { CONFIG, subset, replicateSummary, band, detail, stress, bufferPayoff } from "./model.js";

const dir = new URL("./data/", import.meta.url);
const read = (f) => JSON.parse(readFileSync(new URL(f, dir)));
const policy = read("policy_nl_2026.json"), events = read("scenarios.json").events, gp = read("glidepaths.json");
const optTable = gp.optimised.table;
const saver = { ...DEFAULT_SAVER };
const out = { saver, config: CONFIG, glidepaths: gp, methods: {} };
for (const method of ["repo", "hmm"]) {
  const t0 = Date.now();
  const P = read(`params_${method}.json`).replicates;
  const idxs = subset(P.length, CONFIG.nReps);
  const b = band(idxs.map(i => replicateSummary(P[i], policy, saver, CONFIG.bandPaths, optTable)));
  const med = idxs[b.median_pos], now = P[med].init_probs.indexOf(Math.max(...P[med].init_probs));
  const ev = events[method === "hmm" ? "hmm" : "repo_classifier"];
  out.methods[method] = {
    band: b, median_idx: med,
    detail: { ...detail(P[med], policy, saver, CONFIG.detailPaths, optTable), buffer: bufferPayoff(P[med], now), regime_now: P[med].regimes[now] },
    stress: Object.fromEntries(ev.map(e => [e.name, stress(P[med], policy, saver, e, CONFIG.stressPaths, optTable)])),
  };
  const f = (k) => `${k} all3 ${b[k].p_all3.median.toFixed(3)} target ${b[k].p_target.median.toFixed(3)} income ${b[k].p_income.median.toFixed(3)}`;
  console.log(method, `${((Date.now() - t0) / 1000).toFixed(1)} s |`, ["opt", "today", "equity", "bank"].map(f).join(" | "));
}
writeFileSync(new URL("precomputed.json", dir), JSON.stringify(out));
