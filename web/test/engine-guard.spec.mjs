// The engine's WGSL assumes 32-wide subgroups (GDN, norms, attention). Adapters that may run other sizes must be
// refused with a clear reason instead of computing wrong answers. Fake adapters stand in for non-Apple GPUs.
import { test, expect } from "./fixtures.mjs";

const CASES = [
  ["apple / nvidia", ["shader-f16", "subgroups"], 32, 32, []],
  ["intel", ["shader-f16", "subgroups"], 8, 32, ["subgroup size 8-32 (needs exactly 32)"]],
  ["amd", ["shader-f16", "subgroups"], 32, 64, ["subgroup size 32-64 (needs exactly 32)"]],
  ["no subgroups", ["shader-f16"], 4, 128, ["no subgroups", "subgroup size 4-128 (needs exactly 32)"]],
  ["no f16", ["subgroups"], 32, 32, ["no shader-f16"]],
];

test("engine requirements: refuse adapters that may not run 32-wide subgroups", async ({ page }) => {
  await page.goto("http://127.0.0.1:8787/demo.html?backend=ort");
  const got = await page.evaluate(async (cases) => {
    const { engineProblems } = await import("/engine/engine.js");
    return cases.map(([, features, subgroupMinSize, subgroupMaxSize]) =>
      engineProblems({ features: new Set(features), info: { subgroupMinSize, subgroupMaxSize } }));
  }, CASES);
  expect(got).toEqual(CASES.map((c) => c[4]));
});
