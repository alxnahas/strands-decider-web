// The deployed site, end to end: stock Chrome, empty profile (so the weights really download), no WebGPU errors.
// Runs only with LIVE_URL set, e.g. LIVE_URL=https://alxnahas.github.io/strands-decider-web/ npx playwright test live
import { test, expect, chromium } from "@playwright/test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const URL_ = process.env.LIVE_URL;
test.skip(!URL_, "set LIVE_URL to check a deployment");

test("live site: cold load from Hugging Face, decides, reloads from cache", async () => {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "live-"));
  const ctx = await chromium.launchPersistentContext(profile, { channel: "chrome", headless: !process.env.HEADED, args: [] });
  try {
    const page = await ctx.newPage(), logs = [], hosts = new Set();
    page.on("console", (m) => logs.push(m.text()));
    page.on("pageerror", (e) => logs.push(e.message));
    ctx.on("request", (r) => hosts.add(new URL(r.url()).host));
    const open = async () => {
      await page.goto(URL_);
      await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 15 * 60_000 });
      expect(await page.evaluate(() => document.body.dataset.ready), await page.textContent("#p-status")).toBe("1");
      await page.waitForFunction(() => +(document.body.dataset.inferences || 0) >= 1, null, { timeout: 120_000 });
      await expect(page.getByTestId("answer")).toContainText("billing");
      return page.evaluate(() => window.decider.info);
    };
    const cold = await open(), warm = await open();
    test.info().annotations.push({ type: "load", description: `cold ${(cold.timings.total_load_ms / 1000).toFixed(1)} s, cached ${(warm.timings.total_load_ms / 1000).toFixed(2)} s, kernels=${cold.kernels}, hosts=${[...hosts].join(",")}` });
    expect(cold.cached).toBe(false);
    expect(warm.cached).toBe(true);
    expect(logs.filter((l) => /WGSL|Invalid |GPUPipelineError|load failed|404/i.test(l))).toEqual([]);
  } finally { await ctx.close(); fs.rmSync(profile, { recursive: true, force: true }); }
});
