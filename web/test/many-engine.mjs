// decideMany on the engine: one state, several questions; compare with independent full decides.
import { chromium } from "playwright";
const ctx = await chromium.launchPersistentContext("/tmp/decider-chrome-profile", { channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu"] });
const page = await ctx.newPage(); page.on("pageerror", (e) => console.log("PAGEERROR", e.message));
await page.goto(`http://127.0.0.1:8787/?backend=${process.env.BACKEND || "engine"}`);
await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 600000 });
const filler = Array.from({ length: 60 }, (_, i) => `[${i}] The customer and agent discussed shipping times, the loyalty programme and an address change.`).join("\n");
const states = { short: "Help! My payouts have been failing for 3 days!", long: filler + "\nLatest message: I was charged twice for my last invoice and need one charge reversed, urgently!" };
const qs = { team: { type: "choice", instructions: "Which team should handle this?", options: ["billing", "sales", "retail", "shipping"] },
  urgent: { type: "noul", instructions: "Does the latest message convey urgency?" }, frustration: { type: "score", instructions: "How frustrated is the writer?", options: ["calm", "frustrated", "furious"] },
  refund: { type: "noul", instructions: "Is the customer asking for money back?" }, lang: { type: "choice", instructions: "What language is the latest message in?", options: ["English", "German", "French"] } };
const fmt = (a) => a.choice ?? (a.noul !== undefined ? a.noul.toFixed(3) : a.score.toFixed(2));
for (const [name, st] of Object.entries(states)) {
  let seq = 0; const indiv = {};
  // fresh state string per run so the LRU cannot help the sequential baseline
  for (const [k, q] of Object.entries(qs)) { const r = await page.evaluate(([s, q]) => window.decider.decide(s, q), [st + " ", q]); seq += r.timings.forward_ms; indiv[k] = fmt(r.answer); }
  for (let rep = 0; rep < 2; rep++) {
    const r = await page.evaluate(([s, q]) => window.decider.decideMany(s, q), [st + (rep ? "  " : "   "), qs]);
    console.log(name, `tokens=${r.tokens}`, "many:", JSON.stringify(Object.fromEntries(Object.entries(r.answers).map(([k, a]) => [k, fmt(a)]))), `prefix ${r.timings.prefix_ms.toFixed(0)}ms total ${r.timings.total_ms.toFixed(0)}ms`, "| 5 separate decides:", JSON.stringify(indiv), `${seq.toFixed(0)}ms`);
  }
}
await ctx.close();
