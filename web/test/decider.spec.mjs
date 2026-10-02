// First-success test: Strands Decider v19 runs in real Chrome on WebGPU with no hosted inference.
import fs from "node:fs";
import { test, expect, openReady } from "./fixtures.mjs";
const REF = JSON.parse(fs.readFileSync(new URL("../../ref/reference_cpu.json", import.meta.url))).results;
const FIX = JSON.parse(fs.readFileSync(new URL("../../ref/fixtures.json", import.meta.url)));
const report = { when: new Date().toISOString() };

test("loads, decides billing twice, reloads from cache, matches Python", async ({ page }) => {
  const device = process.env.DEVICE || "webgpu";
  const model = process.env.MODEL || "q4f16p", backend = process.env.BACKEND || "ort", qs = `?device=${device}&model=${model}&backend=${backend}`;
  const info = await openReady(page, qs);
  expect(info, await page.textContent("#status")).toBeTruthy();
  report.model = model; report.runtime = info.runtime; report.adapter = info.adapterInfo; report.load_first = info.timings; report.cached_first = info.cached;
  expect(info.runtime).toBe(backend === "engine" ? "webgpu-engine" : device);
  await expect(page.getByTestId("status")).toContainText(device === "webgpu" ? "WebGPU in page: available" : "WebGPU");

  // Official example through the UI.
  await page.getByRole("button", { name: "Decide" }).click();
  await expect(page.getByTestId("answer")).toContainText("billing");
  const ui = await page.textContent("#result");
  report.ui_result = ui.replace(/\s+/g, " ").trim();
  expect(ui).toMatch(/confidence 0\.\d{3}/);

  // Second inference, no reload: session stays warm.
  await page.getByRole("button", { name: "Decide" }).click();
  await page.waitForFunction(() => document.body.dataset.inferences === "2");

  // Parity against the official Python implementation on every fixture.
  report.parity = [];
  for (const [i, f] of FIX.entries()) {
    const r = await page.evaluate(([s, q]) => window.decider.decide(s, q), [f.state, f.question]);
    const refA = REF[i].answer, browserP = Object.values(r.answer.probabilities);
    const refP = refA.probabilities ? Object.values(refA.probabilities) : [1 - refA.noul, refA.noul];
    const err = Math.max(...browserP.map((p, j) => Math.abs(p - refP[j])));
    const argmax = (a) => a.indexOf(Math.max(...a));
    report.parity.push({ name: f.name, browser: browserP.map((p) => +p.toFixed(4)), python: refP, max_abs_err: +err.toFixed(4), same_top: argmax(browserP) === argmax(refP), forward_ms: +r.timings.forward_ms.toFixed(1), tokens: r.tokens });
  }
  // Near-ties in the reference (top-2 margin < 0.05) may flip under int4; report them, require agreement elsewhere.
  for (const p of report.parity) { const s = [...p.python].sort((a, b) => b - a); p.near_tie = s[0] - s[1] < 0.05; if (!p.near_tie) expect(p.same_top, p.name).toBe(true); }

  // Warm latency for the official example.
  const lat = [];
  for (let i = 0; i < 5; i++) lat.push((await page.evaluate(() => window.decider.decide("Help! My payouts have been failing for 3 days!", { type: "choice", instructions: "Which team should handle this?", options: ["billing", "sales", "retail"] }))).timings.forward_ms);
  report.warm_forward_ms = lat.map((x) => +x.toFixed(1));

  // Reload: assets must come from the OPFS cache.
  await page.reload();
  const info2 = await openReady(page, qs);
  report.load_reload = info2.timings; report.cached_reload = info2.cached;
  expect(info2.cached).toBe(true);
  await page.getByRole("button", { name: "Decide" }).click();
  await expect(page.getByTestId("answer")).toContainText("billing");

  report.console_errors = page.console.filter((m) => m.type === "error" || m.type === "pageerror").map((m) => m.text.slice(0, 300));
  report.offsite_requests = page.offsite;
  expect(page.offsite, "no requests leave localhost").toEqual([]);
  fs.mkdirSync("results", { recursive: true });
  fs.writeFileSync(`results/latest-${backend === "engine" ? "engine" : model}-${device}.json`, JSON.stringify(report, null, 1));
  console.log(JSON.stringify(report, null, 1));
});
