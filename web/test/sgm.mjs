import { chromium } from "playwright";
const ctx = await chromium.launchPersistentContext("/tmp/decider-chrome-profile-feat", { channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu"] });
const page = await ctx.newPage(); await page.goto("http://127.0.0.1:8787/sandbox.html");
console.log(await page.evaluate(async () => { const a = await navigator.gpu.requestAdapter(); return JSON.stringify([...a.info.subgroupMatrixConfigs].map((c) => ({ componentType: c.componentType, resultComponentType: c.resultComponentType, M: c.M, N: c.N, K: c.K }))); }));
await ctx.close();
