// Parity runner: node web/test/parity_runner.mjs <fixture.json> <out.json>
// Replays numpy's random draws through the JS engine and writes its outputs for tests/test_web_parity.py.
import { readFileSync, writeFileSync } from "node:fs";
import { run, historicalScenario } from "../engine.js";

const fx = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = [];
for (const c of fx.cases) {
  const T = c.saver.horizon_years * 12;
  const scenario = c.event ? historicalScenario(c.event, T, c.end_months_before_purchase) : null;
  const draws = { init: Float64Array.from(c.draws.init), U: Float64Array.from(c.draws.U), Z: Float64Array.from(c.draws.Z) };
  const r = run(c.params, fx.policy, c.saver, { nPaths: c.n_paths, table: c.table, scenario, draws, keepPaths: true });
  const T1 = T + 1, pick = (arr) => Array.from(arr);
  out.push({
    name: c.name, V: pick(r.V), B: pick(r.B), G: pick(r.G), D: pick(r.D), Dinc: pick(r.Dinc), Ginc: pick(r.Ginc),
    irr: pick(r.strategies[0].irr), bank_irr: pick(r.bankIrr), house_pa: pick(r.housePa),
    Vpath: pick(r.Vpath), G_mid: pick(r.Gpath[c.mid_month]), D_mid: pick(r.Dpath[c.mid_month]),
    equity_index: pick(r.market.equityIndex), house_index: pick(r.market.houseIndex),
    mortgage_rate: pick(r.market.mortgageRate), cpi: pick(r.market.cpi), regimes: pick(r.market.regimes),
    T1,
  });
}
writeFileSync(process.argv[3], JSON.stringify(out));
