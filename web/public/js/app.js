import { Decider } from "./decider-client.js";
const $ = (id) => document.getElementById(id);
const log = (m) => { $("status").textContent += `\n${m}`; console.log("[decider]", m); };
const params = new URLSearchParams(location.search);
const decider = new Decider({ device: params.get("device") || "webgpu", model: params.get("model") || "q4f16p", onEvent: (e) => {
  if (e.type === "status") log(e.msg);
  if (e.type === "progress" && !e.cached && e.got === e.total) log(`fetch ${e.label}: ${(e.got / 2 ** 20).toFixed(0)}/${(e.total / 2 ** 20).toFixed(0)} MiB`);
}});
window.decider = decider;  // exposed for Playwright and the agent loop
$("status").textContent = `WebGPU in page: ${"gpu" in navigator ? "available" : "NOT available"}`;
decider.ready.then((r) => {
  log(`ready: model=${r.variant} runtime=${r.runtime} load=${r.timings.total_load_ms.toFixed(0)}ms (fetch ${r.timings.fetch_ms.toFixed(0)}ms, session ${r.timings.session_ms.toFixed(0)}ms) cached=${r.cached}`);
  document.body.dataset.ready = "1"; $("ask").disabled = false;
}, (e) => { log(`LOAD FAILED: ${e}`); document.body.dataset.ready = "error"; });

$("form").onsubmit = async (ev) => {
  ev.preventDefault(); $("ask").disabled = true;
  const type = $("qtype").value, options = $("options").value.split(",").map((s) => s.trim()).filter(Boolean);
  const question = { type, instructions: $("question").value, options };
  try {
    const r = await decider.decide($("state").value, question);
    render(r);
  } catch (e) { $("result").textContent = `error: ${e}`; }
  $("ask").disabled = false;
};

function render(r) {
  const a = r.answer, rows = Object.entries(a.probabilities);
  const top = a.type === "choice" ? a.choice : a.type === "noul" ? (a.noul >= 0.5 ? "true" : "false") : null;
  $("result").innerHTML = `<h3 data-testid="answer">${a.type === "choice" ? `→ ${a.choice}` : a.type === "noul" ? `P(true) = ${a.noul.toFixed(3)}` : `score = ${a.score.toFixed(2)}`}
    ${a.confidence !== undefined ? ` <small>(confidence ${a.confidence.toFixed(3)})</small>` : ""}</h3>
    <table>${rows.map(([k, p]) => `<tr class="${k === top ? "chosen" : ""}"><td>${k}</td><td>${p.toFixed(4)}</td><td><div class="bar" style="width:${(p * 200).toFixed(0)}px"></div></td></tr>`).join("")}</table>
    <p><small>${r.runtime} · ${r.tokens} tokens · forward ${r.timings.forward_ms.toFixed(0)} ms · total ${r.timings.total_ms.toFixed(0)} ms</small></p>`;
  document.body.dataset.inferences = String(+(document.body.dataset.inferences || 0) + 1);
}
