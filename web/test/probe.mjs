// Ad-hoc probe: open the page in real Chrome, stream console, run one decision.
import { chromium } from "playwright";
const headless = process.env.HEADED ? false : true;
const ctx = await chromium.launchPersistentContext(process.env.PROFILE || "/tmp/decider-chrome-profile", {
  channel: "chrome", headless, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist"],
});
const page = ctx.pages()[0] || (await ctx.newPage());
page.on("console", (m) => console.log(`[${m.type()}]`, m.text().slice(0, 400)));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(`http://127.0.0.1:8787/${process.env.QS || ""}`);
await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 600000 });
console.log(await page.textContent("#status"));
if ((await page.evaluate(() => document.body.dataset.ready)) === "1") {
  for (let i = 0; i < 3; i++) {
    const r = await page.evaluate(() => window.decider.decide("Help! My payouts have been failing for 3 days!", { type: "choice", instructions: "Which team should handle this?", options: ["billing", "sales", "retail"] }));
    console.log(JSON.stringify(r.answer), JSON.stringify(r.timings));
  }
}
await ctx.close();
