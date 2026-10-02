// A/B parity + latency: browser runtime vs the official PyTorch reference on ref/bench.json.
import fs from "node:fs";
import { test, expect, openReady } from "./fixtures.mjs";
const REF = JSON.parse(fs.readFileSync(new URL("../../ref/reference_bench_cpu.json", import.meta.url))).results;
const FIX = JSON.parse(fs.readFileSync(new URL("../../ref/bench.json", import.meta.url)));
const argmax = (a) => a.indexOf(Math.max(...a));

test("bench parity", async ({ page }) => {
  const device = process.env.DEVICE || "webgpu", backend = process.env.BACKEND || "ort", model = backend === "engine" ? "engine" : process.env.MODEL || "q4f16p";
  const ortv = process.env.ORT || "npm";
  const info = await openReady(page, `?device=${device}&model=${model}&backend=${backend}&ort=${ortv}`);
  expect(info?.runtime).toBe(backend === "engine" ? "webgpu-engine" : device);
  const rows = [];
  for (const [i, f] of FIX.entries()) {
    const r = await page.evaluate(([s, q]) => window.decider.decide(s, q), [f.state, f.question]);
    const a = REF[i].answer, refP = a.probabilities ? Object.values(a.probabilities) : [1 - a.noul, a.noul];
    const p = Object.values(r.answer.probabilities);
    const label = f.label == null ? null : f.question.type === "choice" ? f.question.options.indexOf(f.label) : f.question.type === "noul" ? f.label : f.label;
    rows.push({ name: f.name, tokens: r.tokens, forward_ms: +r.timings.forward_ms.toFixed(1), max_abs_err: +Math.max(...p.map((x, j) => Math.abs(x - refP[j]))).toFixed(4),
      same_top: argmax(p) === argmax(refP), correct_browser: label == null ? null : argmax(p) === label, correct_python: label == null ? null : argmax(refP) === label,
      conf_browser: r.answer.confidence ?? null, conf_python: a.confidence ?? null });
  }
  const s = { model, device, n: rows.length, agree: rows.filter((r) => r.same_top).length, max_err: Math.max(...rows.map((r) => r.max_abs_err)),
    mean_err: +(rows.reduce((t, r) => t + r.max_abs_err, 0) / rows.length).toFixed(4),
    acc_browser: rows.filter((r) => r.correct_browser).length, acc_python: rows.filter((r) => r.correct_python).length, rows };
  fs.mkdirSync("results", { recursive: true });
  fs.writeFileSync(`results/bench-${model}-${device}${ortv === "npm" ? "" : "-ort" + ortv}.json`, JSON.stringify(s, null, 1));
  console.log(JSON.stringify({ ...s, rows: undefined }), rows.map((r) => `${r.name}:${r.tokens}t/${r.forward_ms}ms/err${r.max_abs_err}${r.same_top ? "" : "/FLIP"}`).join(" "));
});
