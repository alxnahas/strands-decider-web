// The GitHub Pages build: site under a subpath with no COOP/COEP headers, weights from a second origin with CORS
// (as Hugging Face serves them), stock Chrome. Builds with build-pages.mjs into a temp dir, from PAGES_FROM (an export
// directory, build-pages.mjs --from) if set.
import { test, expect, chromium } from "@playwright/test";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { PROFILE } from "./fixtures.mjs";
import { startStatic } from "./static-server.mjs";

const SITE = 8789, ASSETS = 8790, PREFIX = "/strands-decider-web/";
let servers = [], out;

test.beforeAll(async () => {
  out = fs.mkdtempSync(path.join(os.tmpdir(), "pages-"));
  const from = process.env.PAGES_FROM ? ["--from", path.resolve(process.env.PAGES_FROM)] : [];
  execFileSync("node", ["build-pages.mjs", "--out", out, "--assets", `http://127.0.0.1:${ASSETS}/`, ...from], { cwd: new URL("..", import.meta.url).pathname, stdio: "inherit" });
  servers = [await startStatic({ root: path.join(out, "site"), port: SITE, prefix: PREFIX }), await startStatic({ root: path.join(out, "assets"), port: ASSETS, cors: true })];
});
test.afterAll(async () => { servers.forEach((s) => s.close()); if (out) fs.rmSync(out, { recursive: true, force: true }); });

test("pages build: subpath, no isolation headers, cross-origin weights, stock Chrome", async () => {
  const site = path.join(out, "site");
  const big = fs.readdirSync(site, { recursive: true }).filter((f) => fs.statSync(path.join(site, f)).size > 50 * 2 ** 20);
  expect(big, "site must stay small; weights belong in assets/").toEqual([]);

  const ctx = await chromium.launchPersistentContext(PROFILE, { channel: "chrome", headless: !process.env.HEADED, args: [] });
  try {
    const page = await ctx.newPage(), logs = [], hosts = new Set();
    page.on("console", (m) => logs.push(m.text()));
    page.on("pageerror", (e) => logs.push(e.message));
    ctx.on("request", (r) => hosts.add(new URL(r.url()).host));
    let ranged = 0; ctx.on("response", (r) => { if (r.status() === 206) ranged++; });
    await page.goto(`http://127.0.0.1:${SITE}${PREFIX}`);
    await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 10 * 60_000 });
    expect(await page.evaluate(() => document.body.dataset.ready), await page.textContent("#p-status")).toBe("1");
    await page.waitForFunction(() => +(document.body.dataset.inferences || 0) >= 1, null, { timeout: 60_000 });
    await expect(page.getByTestId("answer")).toContainText("billing");
    // Japanese text: a build with on-demand embedding rows fetches them by byte range from the assets origin
    const n = +(await page.evaluate(() => document.body.dataset.inferences));
    await page.getByRole("button", { name: "Ticket in Japanese" }).click();
    await page.waitForFunction((n) => +(document.body.dataset.inferences || 0) > n, n, { timeout: 60_000 });
    await expect(page.getByTestId("answer")).toContainText("shipping");
    const lazy = !!JSON.parse(fs.readFileSync(path.join(out, "assets/engine-weights/manifest.json"), "utf8")).embed;
    expect(ranged > 0, "range requests for embedding rows").toBe(lazy);
    expect([...hosts].sort()).toEqual([`127.0.0.1:${SITE}`, `127.0.0.1:${ASSETS}`].sort());
    expect(logs.filter((l) => /WGSL|Invalid |GPUPipelineError|load failed|embedding rows|404/i.test(l))).toEqual([]);
  } finally { await ctx.close(); }
});
