import { chromium } from "playwright";
const ctx = await chromium.launchPersistentContext("/tmp/decider-chrome-profile", { channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu"] });
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("PAGEERROR", e.message)); page.on("console", (m) => m.type() === "error" && !m.text().includes("VerifyEach") && console.log("ERR", m.text().slice(0, 300)));
await page.goto(`http://127.0.0.1:8787/?model=${process.env.MODEL || "q4f16p"}&${process.env.QS || ""}`);
await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 600000 });
console.log(process.env.MODEL || "q4f16p", JSON.stringify((await page.evaluate(() => window.decider.bench([1, 8, 16, 32, 64, 128, 256, 512]))).bench));
await ctx.close();
