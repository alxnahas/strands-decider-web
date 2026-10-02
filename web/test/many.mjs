import { chromium } from "playwright";
const ctx = await chromium.launchPersistentContext("/tmp/decider-chrome-profile", { channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu"] });
const page = await ctx.newPage(); page.on("console", (m) => m.type() === "error" && console.log("[err]", m.text().slice(0, 300)));
await page.goto(`http://127.0.0.1:8787/?model=${process.env.MODEL || "q4f16p"}`);
await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 600000 });
const state = "Help! My payouts have been failing for 3 days!";
const qs = { team: { type: "choice", instructions: "Which team should handle this?", options: ["billing", "sales", "retail"] },
  urgent: { type: "noul", instructions: "Does this convey urgency?" }, frustration: { type: "score", instructions: "How frustrated is the writer?", options: ["calm", "frustrated", "depressed"] } };
for (let i = 0; i < 3; i++) {
  const r = await page.evaluate(([s, q]) => window.decider.decideMany(s, q), [state, qs]);
  console.log(JSON.stringify(Object.fromEntries(Object.entries(r.answers).map(([k, a]) => [k, a.choice ?? a.noul?.toFixed(3) ?? a.score.toFixed(2)]))), JSON.stringify(r.timings, (k, v) => typeof v === "number" ? +v.toFixed(0) : v));
}
let seq = 0;
for (const q of Object.values(qs)) seq += (await page.evaluate(([s, q]) => window.decider.decide(s, q), [state, q])).timings.forward_ms;
console.log("sequential full forwards ms", seq.toFixed(0));
await ctx.close();
