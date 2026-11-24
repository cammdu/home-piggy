// Precompute the default saver's results so the page is complete before any computation runs.
// Uses exactly the functions and run sizes the page uses (model.js CONFIG), so the live
// recomputation for the same inputs returns the same numbers.
// Usage: node web/build_precomputed.mjs [nl|uk]
import { readFileSync, writeFileSync } from "node:fs";
import { DEFAULT_SAVER, UK_DEFAULT_SAVER } from "./engine.js";
import { CONFIG, subset, replicateSummary, band, detail, stress, bufferPayoff } from "./model.js";

const country = process.argv[2] || "nl";
const dir = new URL(`./data/${country}/`, import.meta.url);
const read = (f) => JSON.parse(readFileSync(new URL(f, dir)));
const policy = read("policy_2026.json"), events = read("scenarios.json").events, gp = read("glidepaths.json");
const optTable = gp.optimised.table;
const saver = { ...(country === "uk" ? UK_DEFAULT_SAVER : DEFAULT_SAVER) };
const out = { country, saver, config: CONFIG, glidepaths: gp, methods: {} };
function build(sv, label) {
  const res = {};
  for (const method of ["repo", "hmm"]) {
    const t0 = Date.now();
    const P = read(`params_${method}.json`).replicates;
    const idxs = subset(P.length, CONFIG.nReps);
    const b = band(idxs.map(i => replicateSummary(P[i], policy, sv, CONFIG.bandPaths, optTable)));
    const med = idxs[b.median_pos], now = P[med].init_probs.indexOf(Math.max(...P[med].init_probs));
    const ev = events[method === "hmm" ? "hmm" : "repo_classifier"];
    res[method] = {
      band: b, median_idx: med,
      detail: { ...detail(P[med], policy, sv, CONFIG.detailPaths, optTable), buffer: bufferPayoff(P[med], now), regime_now: P[med].regimes[now] },
      stress: Object.fromEntries(ev.map(e => [e.name, stress(P[med], policy, sv, e, CONFIG.stressPaths, optTable)])),
    };
    const f = (k) => `${k} all3 ${b[k].p_all3.median.toFixed(3)} target ${b[k].p_target.median.toFixed(3)} income ${b[k].p_income.median.toFixed(3)}`;
    console.log(country, label, method, `${((Date.now() - t0) / 1000).toFixed(1)} s |`, ["opt", "today", "equity", "bank"].map(f).join(" | "));
  }
  return res;
}
out.methods = build(saver, "");
if (country === "uk") out.methods_no_lisa = build({ ...saver, lisa: false }, "no-LISA");
writeFileSync(new URL("precomputed.json", dir), JSON.stringify(out));
