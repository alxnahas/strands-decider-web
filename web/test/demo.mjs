// Smoke test for demo.html: load, click every example, type to trigger live re-decision, take screenshots.
import { chromium } from "playwright";
const ctx = await chromium.launchPersistentContext(process.env.PROFILE || "/tmp/decider-chrome-profile", {
  channel: "chrome", headless: !process.env.HEADED, viewport: { width: 1100, height: 820 }, args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
});
const page = ctx.pages()[0] || (await ctx.newPage());
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text().slice(0, 200)); });
await page.goto(`http://127.0.0.1:8787/demo.html${process.env.QS || ""}`);
await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 600000 });
const settle = async (n) => { await page.waitForFunction((n) => +(document.body.dataset.inferences || 0) >= n, n, { timeout: 60000 }); await page.waitForTimeout(350); };
await settle(1);
const chips = await page.$$("#examples .chip");
const out = [];
for (let i = 0; i < chips.length; i++) {
  const before = +(await page.evaluate(() => document.body.dataset.inferences));
  await chips[i].click(); await settle(before + 1);
  out.push({ example: await chips[i].textContent(), answer: (await page.textContent("#answer")).trim(), fwd: await page.textContent("#s-fwd"), tokens: await page.textContent("#s-tok") });
  if (i === 0 || i === 4) await page.screenshot({ path: `results/demo-${i}.png` });
}
// live typing: edit the state character by character
await chips[0].click(); await settle(+(await page.evaluate(() => document.body.dataset.inferences)) + 1);
const n0 = +(await page.evaluate(() => document.body.dataset.inferences));
await page.click("#state"); await page.keyboard.press("End"); await page.keyboard.type(" Also I want to upgrade my plan.", { delay: 40 });
await page.waitForTimeout(800);
const n1 = +(await page.evaluate(() => document.body.dataset.inferences));
out.push({ live: `${n1 - n0} decisions while typing`, answer: (await page.textContent("#answer")).trim() });
await page.screenshot({ path: "results/demo-live.png" });
console.log(JSON.stringify({ pills: await page.$$eval(".pill", (p) => p.map((x) => x.textContent)), out, errors }, null, 1));
await ctx.close();
