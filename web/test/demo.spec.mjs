// demo.html in stock Chrome (no WebGPU flags, as a user would open it) and with --enable-unsafe-webgpu,
// on both backends: loads, answers the official example, re-decides while typing, logs no WebGPU errors.
import { test, expect, chromium } from "@playwright/test";
import { PROFILE } from "./fixtures.mjs";

const CHROME = { stock: [], flagged: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"] };
const GPU_ERROR = /WGSL|Invalid (ShaderModule|ComputePipeline|BindGroup|CommandBuffer)|GPUPipelineError|GPUValidationError|load failed/i;

// With no ?backend, the demo picks the faster backend for this browser: the engine only when subgroup-matrix is
// available (its portable matmul is ~2x slower than ORT on long prompts), otherwise ONNX Runtime Web.
for (const [chrome, expected] of [["stock", "ort"], ["flagged", "engine"]]) {
  test(`demo: ${chrome} Chrome picks ${expected} by default`, async () => {
    const ctx = await chromium.launchPersistentContext(PROFILE, { channel: "chrome", headless: !process.env.HEADED, args: CHROME[chrome] });
    try {
      const page = await ctx.newPage();
      await page.goto("http://127.0.0.1:8787/demo.html");
      await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 10 * 60_000 });
      expect(new URL(page.url()).searchParams.get("backend")).toBe(expected);
      expect(await page.evaluate(() => window.decider.info.runtime)).toBe(expected === "engine" ? "webgpu-engine" : "webgpu");
    } finally { await ctx.close(); }
  });
}

for (const [chrome, args] of Object.entries(CHROME)) {
  for (const backend of ["engine", "ort"]) {
    test(`demo: ${chrome} Chrome, ${backend} backend`, async () => {
      const ctx = await chromium.launchPersistentContext(PROFILE, { channel: "chrome", headless: !process.env.HEADED, args });
      try {
        const page = await ctx.newPage(), logs = [];
        page.on("console", (m) => logs.push(m.text()));
        page.on("pageerror", (e) => logs.push(e.message));
        await page.goto(`http://127.0.0.1:8787/demo.html?backend=${backend}`);
        await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 10 * 60_000 });
        expect(await page.evaluate(() => document.body.dataset.ready), await page.textContent("#p-status")).toBe("1");
        const info = await page.evaluate(() => window.decider.info);
        test.info().annotations.push({ type: "runtime", description: `${info.runtime} kernels=${info.kernels ?? "-"} load=${info.timings.total_load_ms.toFixed(0)}ms` });
        if (backend === "engine") expect(info.kernels).toBe(chrome === "stock" ? "portable" : "subgroup-matrix");

        await page.waitForFunction(() => +(document.body.dataset.inferences || 0) >= 1, null, { timeout: 60_000 });
        await expect(page.getByTestId("answer")).toContainText("billing");

        const n0 = +(await page.evaluate(() => document.body.dataset.inferences));
        await page.click("#state"); await page.keyboard.press("End");
        await page.keyboard.type(" Please help.", { delay: 60 });
        await page.waitForTimeout(1000);
        expect(+(await page.evaluate(() => document.body.dataset.inferences)) - n0).toBeGreaterThanOrEqual(3);
        await expect(page.getByTestId("answer")).toContainText("billing");

        expect(logs.filter((l) => GPU_ERROR.test(l)).slice(0, 3)).toEqual([]);
      } finally { await ctx.close(); }
    });
  }
}
