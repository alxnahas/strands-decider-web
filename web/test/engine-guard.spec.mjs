// The engine needs shader-f16. Its subgroup kernels assume 32-wide subgroups (GDN, norms, attention), so adapters
// that may run other sizes get the subgroup-free kernels instead. Fake adapters stand in for non-Apple GPUs.
import { test, expect } from "./fixtures.mjs";

const CASES = [  // name, features, subgroup min, max, engineProblems, subgroups32
  ["apple / nvidia", ["shader-f16", "subgroups"], 32, 32, [], true],
  ["intel", ["shader-f16", "subgroups"], 8, 32, [], false],
  ["amd", ["shader-f16", "subgroups"], 32, 64, [], false],
  ["no subgroups", ["shader-f16"], 4, 128, [], false],
  ["no f16", ["subgroups"], 32, 32, ["no shader-f16"], true],
];

test("engine requirements: shader-f16; subgroups only when exactly 32 wide", async ({ page }) => {
  await page.goto("http://127.0.0.1:8787/demo.html?backend=ort");
  const got = await page.evaluate(async (cases) => {
    const { engineProblems, subgroups32 } = await import("/engine/engine.js");
    return cases.map(([, features, subgroupMinSize, subgroupMaxSize]) => {
      const a = { features: new Set(features), info: { subgroupMinSize, subgroupMaxSize } };
      return [engineProblems(a), subgroups32(a)];
    });
  }, CASES);
  expect(got).toEqual(CASES.map((c) => [c[4], c[5]]));
});
