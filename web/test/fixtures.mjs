// Real Chrome with a persistent profile, so OPFS-cached model assets survive between tests.
import { test as base, chromium } from "@playwright/test";
export const PROFILE = process.env.PROFILE || "/tmp/decider-chrome-profile";
export const test = base.extend({
  ctx: [async ({}, use) => {
    const ctx = await chromium.launchPersistentContext(PROFILE, {
      channel: "chrome", headless: !process.env.HEADED, args: process.env.CHROME === "stock" ? [] : ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
    });
    await use(ctx); await ctx.close();
  }, { scope: "worker" }],
  page: async ({ ctx }, use) => {
    const page = await ctx.newPage(); page.console = []; page.offsite = [];
    page.on("console", (m) => page.console.push({ type: m.type(), text: m.text() }));
    page.on("pageerror", (e) => page.console.push({ type: "pageerror", text: e.message }));
    page.on("request", (r) => { if (!r.url().startsWith("http://127.0.0.1:8787/") && !r.url().startsWith("data:")) page.offsite.push(r.url()); });
    await use(page); await page.close();
  },
});
export { expect } from "@playwright/test";
export async function openReady(page, qs = "") {
  await page.goto(`http://127.0.0.1:8787/${qs}`);
  await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 10 * 60_000 });
  return page.evaluate(() => (document.body.dataset.ready === "1" ? window.decider.info : null));
}
