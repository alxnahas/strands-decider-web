import { chromium } from "playwright";
const ctx = await chromium.launchPersistentContext("/tmp/decider-chrome-profile-feat", { channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu"] });
const page = await ctx.newPage();
page.on("console", (m) => !m.text().includes("VerifyEach") && console.log(m.text()));
page.on("pageerror", (e) => console.log("PAGEERROR", e.message));
await page.goto(`http://127.0.0.1:8787/gemmbench/${process.env.PAGE || ""}${process.env.QS ? "?" + process.env.QS : ""}`);
await page.waitForFunction(() => document.body.dataset.done, null, { timeout: 900000 });
await ctx.close();
