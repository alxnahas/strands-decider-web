// Run the decide-act loop: decider lives in one Chrome tab (WebGPU), Playwright actuates the sandbox tab.
import { chromium } from "playwright";
import { runLoop } from "./agent.mjs";
const BASE = "http://127.0.0.1:8787";
const tickets = (process.argv[2] || "payouts,refund,bulk,vague").split(",");
const threshold = +(process.env.THRESHOLD || 0.5);
const ctx = await chromium.launchPersistentContext(process.env.PROFILE || "/tmp/decider-chrome-profile", { channel: "chrome", headless: !process.env.HEADED, args: ["--enable-unsafe-webgpu"] });
const model = await ctx.newPage();
await model.goto(`${BASE}/?model=${process.env.MODEL || "q4f16p"}&backend=${process.env.BACKEND || "ort"}`);
await model.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 600000 });
const decide = (state, question) => model.evaluate(([s, q]) => window.decider.decide(s, q), [state, question]);
const goal = "Triage the support ticket: route it to the team that owns the problem, set a priority that matches its urgency, then submit it.";
const results = [];
for (const t of tickets) {
  const page = await ctx.newPage();
  await page.goto(`${BASE}/sandbox.html?ticket=${t}`);
  const r = await runLoop({ page, decide, goal, threshold, log: (e) => console.log(`  [${t}]`, JSON.stringify(e)) });
  console.log(`${t}: stop=${r.stop} ${r.detail || ""}`);
  results.push({ ticket: t, stop: r.stop, detail: r.detail, trace: r.trace });
  await page.close();
}
await ctx.close();
if (process.env.OUT) (await import("node:fs")).writeFileSync(process.env.OUT, JSON.stringify(results, null, 1));
