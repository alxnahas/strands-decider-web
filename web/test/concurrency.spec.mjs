// Overlapping requests (as window.decider and the agent loop allow) give the same answers as sequential ones.
// The worker must serialize them: ORT Web hung on two concurrent session.run calls.
import { test, expect, openReady } from "./fixtures.mjs";

const QS = [
  ["Help! My payouts have been failing for 3 days!", { type: "choice", instructions: "Which team should handle this?", options: ["billing", "sales", "retail"] }],
  ["sihamba ngokushesha", { type: "choice", instructions: "What language is this phrase in?", options: ["English", "Zulu", "Dutch"] }],
  ["Absolutely love the new design, checkout is so much faster now!", { type: "noul", instructions: "Is this review positive?" }],
];

test("overlapping decide and decideMany calls match sequential ones", async ({ page }) => {
  const backend = process.env.BACKEND || "ort";
  expect(await openReady(page, `?backend=${backend}`)).toBeTruthy();
  const r = await page.evaluate(async (qs) => {
    const probs = (a) => Object.values(a.probabilities).map((p) => +p.toFixed(4));
    const timed = (p) => Promise.race([p, new Promise((_, no) => setTimeout(() => no(new Error("timed out")), 30_000))]);
    const seq = [];
    for (const [s, q] of qs) seq.push(probs((await window.decider.decide(s, q)).answer));
    const many = window.decider.decideMany(qs[0][0], { team: qs[0][1] });
    const par = await timed(Promise.all([...qs.map(([s, q]) => window.decider.decide(s, q)), many]));
    return { seq, par: par.slice(0, -1).map((x) => probs(x.answer)), many: probs(par.at(-1).answers.team) };
  }, QS);
  expect(r.par).toEqual(r.seq);
  expect(r.many.map((p) => +p.toFixed(2))).toEqual(r.seq[0].map((p) => +p.toFixed(2)));
});
