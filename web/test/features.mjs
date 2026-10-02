import { chromium } from "playwright";
const args = ["--enable-unsafe-webgpu", ...(process.env.ARGS ? process.env.ARGS.split(" ") : [])];
const ctx = await chromium.launchPersistentContext("/tmp/decider-chrome-profile-feat", { channel: "chrome", headless: true, args });
const page = await ctx.newPage(); await page.goto("http://127.0.0.1:8787/sandbox.html");
console.log(await page.evaluate(async () => { const a = await navigator.gpu.requestAdapter(); return [...a.features].sort().join(" "); }));
await ctx.close();
