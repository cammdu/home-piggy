// Web Worker: runs re-calibrations off the main thread. Messages (all carry method, saver, optTable):
//   {id, kind: "band", idxs, nPaths}          -> {id, result: [{idx, summary}]}
//   {id, kind: "detail", idx, nPaths}         -> {id, result: detail + buffer}
//   {id, kind: "stress", idx, event, nPaths}  -> {id, result: stress}
import { replicateSummary, detail, stress, bufferPayoff } from "./model.js";

const cache = {};
async function load(name) {
  if (!cache[name]) cache[name] = fetch(`data/${name}.json`).then(r => {
    if (!r.ok) throw new Error(`Could not load data/${name}.json (${r.status})`); return r.json(); });
  return cache[name];
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    const [P, policy] = await Promise.all([load(`params_${msg.method}`), load("policy_nl_2026")]);
    const reps = P.replicates;
    let result;
    if (msg.kind === "band") {
      result = msg.idxs.map(idx => ({ idx, summary: replicateSummary(reps[idx], policy, msg.saver, msg.nPaths, msg.optTable) }));
    } else if (msg.kind === "detail") {
      const p = reps[msg.idx], now = p.init_probs.indexOf(Math.max(...p.init_probs));
      result = { ...detail(p, policy, msg.saver, msg.nPaths, msg.optTable), buffer: bufferPayoff(p, now), regime_now: p.regimes[now] };
    } else if (msg.kind === "stress") {
      result = stress(reps[msg.idx], policy, msg.saver, msg.event, msg.nPaths, msg.optTable);
    } else throw new Error(`Unknown task ${msg.kind}`);
    self.postMessage({ id: msg.id, result });
  } catch (e) {
    self.postMessage({ id: msg.id, error: String((e && e.message) || e) });
  }
};
