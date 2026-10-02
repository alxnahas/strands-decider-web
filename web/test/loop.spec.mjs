// Loop stages tested independently (sandbox page, scripted decider), then end-to-end with the real model.
import fs from "node:fs";
import { test, expect, openReady } from "./fixtures.mjs";
import { observe, proposeCandidates, choose, policy, execute, checkTerminal, runLoop, ALLOWLIST } from "../loop/agent.mjs";

const fakeDecide = (pick, confidence) => async (_s, q) => ({ answer: { choice: q.options.find((o) => o.includes(pick)) ?? q.options[0], confidence, probabilities: {} } });

test("observe + candidates + terminal", async ({ page }) => {
  await page.goto("/sandbox.html?ticket=payouts");
  const obs = await observe(page);
  expect(obs.ticket).toContain("payouts");
  expect(proposeCandidates(obs).map((c) => c.action)).toEqual(["assign-billing", "assign-sales", "assign-retail"]);
  expect(proposeCandidates({ buttons: [{ action: "rm -rf", label: "x" }] })).toEqual([]);
  expect(checkTerminal(obs)).toBeNull();
  expect(checkTerminal({ outcome: "success" })).toBe("success");
});

test("policy + execute refuse unsafe actions", async ({ page }) => {
  expect(policy({ confidence: 0.4, action: {} }, 0.5)).toBe("escalate");
  expect(policy({ confidence: 0.9, action: {} }, 0.5)).toBe("act");
  expect(policy({ confidence: 0.9, action: undefined }, 0.5)).toBe("escalate");
  await page.goto("/sandbox.html");
  await expect(execute(page, { action: "delete-account" })).rejects.toThrow("not allowlisted");
  expect(ALLOWLIST.some((re) => re.test("submit"))).toBe(true);
});

test("loop logic with a scripted decider: success, error, low confidence, max steps", async ({ page }) => {
  await page.goto("/sandbox.html?ticket=payouts");
  const seq = ["billing", "urgent", "Submit"]; let i = 0;
  const scripted = async (s, q) => fakeDecide(seq[i++], 0.95)(s, q);
  expect((await runLoop({ page, decide: scripted, goal: "g" })).stop).toBe("success");
  await page.goto("/sandbox.html?ticket=payouts");
  expect((await runLoop({ page, decide: fakeDecide("sales", 0.95), goal: "g" })).stop).toBe("error");
  await page.goto("/sandbox.html?ticket=payouts");
  expect((await runLoop({ page, decide: fakeDecide("billing", 0.2), goal: "g" })).stop).toBe("low-confidence");
  await page.goto("/sandbox.html?ticket=payouts");
  expect((await runLoop({ page, decide: fakeDecide("billing", 0.95), goal: "g", maxSteps: 1 })).stop).toBe("max-steps");
  void choose;
});

test("end-to-end with Strands Decider on WebGPU", async ({ ctx }) => {
  const model = await ctx.newPage();
  const info = await openReady(model, `?model=${process.env.MODEL || "q4f16p"}&backend=${process.env.BACKEND || "ort"}`);
  expect(info?.runtime).toMatch(/^webgpu/);
  const decide = (s, q) => model.evaluate(([s, q]) => window.decider.decide(s, q), [s, q]);
  const goal = "Triage the support ticket: route it to the team that owns the problem, set a priority that matches its urgency, then submit it.";
  const out = {};
  for (const t of ["payouts", "vague"]) {
    const page = await ctx.newPage(); await page.goto(`/sandbox.html?ticket=${t}`);
    out[t] = await runLoop({ page, decide, goal, threshold: 0.5 }); await page.close();
  }
  expect(out.payouts.stop).toBe("success");
  expect(await Promise.resolve(out.payouts.detail)).toContain("team=billing");
  expect(out.vague.stop).toBe("low-confidence");
  fs.mkdirSync("results", { recursive: true });
  fs.writeFileSync("results/latest-loop.json", JSON.stringify(out, (k, v) => (k === "escalate" ? undefined : v), 1));
});
