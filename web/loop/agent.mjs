// Decide-act loop. Each stage is a separate, independently testable function.
//   observe -> proposeCandidates -> choose (Strands Decider) -> policy -> execute -> checkTerminal

/** Allowlisted action ids the executor may ever click (prefixes). Anything else is ignored. */
export const ALLOWLIST = [/^assign-(billing|sales|retail)$/, /^priority-(low|normal|urgent)$/, /^submit$/, /^discard$/];

/** 1. Observe: a small structured snapshot of the page, nothing else. */
export async function observe(page) {
  return page.evaluate(() => ({
    ticket: document.getElementById("ticket")?.textContent ?? "",
    step: document.getElementById("step")?.textContent ?? "",
    banner: document.getElementById("banner")?.textContent ?? "",
    outcome: document.body.dataset.outcome ?? null,
    buttons: [...document.querySelectorAll("button[data-action]")].map((b) => ({ action: b.dataset.action, label: b.textContent.trim() })),
  }));
}

/** 2. Candidates: visible buttons whose action is allowlisted, capped. */
export function proposeCandidates(obs, max = 6) {
  return obs.buttons.filter((b) => ALLOWLIST.some((re) => re.test(b.action))).slice(0, max);
}

/** 3. Choose: one choice question to the decider. State = goal + observation. */
export async function choose(decide, goal, obs, candidates) {
  const state = { goal, ticket: obs.ticket, current_step: obs.step };
  const question = { type: "choice", instructions: `${obs.step} Pick the next action that best achieves the goal.`, options: candidates.map((c) => c.label) };
  const r = await decide(state, question);
  const label = r.answer.choice;
  return { action: candidates.find((c) => c.label === label), confidence: r.answer.confidence, probabilities: r.answer.probabilities, forward_ms: r.timings?.forward_ms };
}

/** 4. Policy: act only above the threshold; otherwise escalate. */
export function policy(choice, threshold) {
  return choice.confidence >= threshold && choice.action ? "act" : "escalate";
}

/** 5. Execute exactly one validated action. */
export async function execute(page, action) {
  if (!ALLOWLIST.some((re) => re.test(action.action))) throw new Error(`action ${action.action} not allowlisted`);
  const btn = page.locator(`button[data-action="${action.action}"]`);
  if ((await btn.count()) !== 1) throw new Error(`action ${action.action} not uniquely present`);
  await btn.click();
}

/** 6. Terminal check, from observation only. */
export function checkTerminal(obs) {
  if (obs.outcome === "success") return "success";
  if (obs.outcome === "error") return "error";
  return null;
}

/** The loop: stops on success, error, low confidence, no candidates, or maxSteps. */
export async function runLoop({ page, decide, goal, threshold = 0.5, maxSteps = 6, log = () => {} }) {
  const trace = [];
  for (let step = 0; step < maxSteps; step++) {
    const obs = await observe(page);
    const term = checkTerminal(obs);
    if (term) return { stop: term, detail: obs.banner, trace };
    const candidates = proposeCandidates(obs);
    if (candidates.length === 0) return { stop: "no-candidates", trace };
    const choice = await choose(decide, goal, obs, candidates);
    const verdict = policy(choice, threshold);
    const entry = { step, observed: obs.step, candidates: candidates.map((c) => c.label), chosen: choice.action?.label, model_confidence: +choice.confidence.toFixed(3), threshold, verdict, forward_ms: choice.forward_ms && +choice.forward_ms.toFixed(0) };
    trace.push(entry); log(entry);
    if (verdict === "escalate") return { stop: "low-confidence", escalate: { obs, candidates, choice }, trace };
    await execute(page, choice.action);
  }
  const obs = await observe(page);
  return { stop: checkTerminal(obs) || "max-steps", detail: obs.banner, trace };
}
