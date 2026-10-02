// Port of strands_decider/prompting.py + infer.py option indexing (pointer head v19).
const NOUL_LABELS = ["false", "true"];
const NOUL_CRITERIA = { false: "the statement does not hold for this state", true: "the statement holds for this state" };
const HEADERS = {
  noul: "Decide whether the statement is true of the state.",
  choice: "Select exactly one option.",
  score: "Rate the state against the ordered levels below (lowest first).",
};

export function renderContent(c) {
  return typeof c === "string" ? c.trim() : JSON.stringify(c, null, 2);
}

export function renderState(state) {
  return `<state>\n${renderContent(state)}\n</state>\n`;
}

/** question: {type, instructions, options?: string[] | {name: desc}} */
export function renderQuestion(q) {
  let pairs;
  if (q.type === "noul") pairs = NOUL_LABELS.map((l) => [l, (q.criteria || {})[l] ?? NOUL_CRITERIA[l]]);
  else if (q.type === "choice") pairs = Array.isArray(q.options) ? q.options.map((o) => [o, ""]) : Object.entries(q.options);
  else if (q.type === "score") pairs = q.options.map((d, i) => [String(i), d]);
  else throw new Error(`unknown question type ${q.type}`);
  const prefix = `<question type="${q.type}">\n${HEADERS[q.type]}\n${renderContent(q.instructions)}\n<options>\n`;
  const lines = [], spans = [];
  let cursor = prefix.length;
  pairs.forEach(([name, desc], i) => {
    desc = (desc || "").split(/\s+/).filter(Boolean).join(" ");
    const line = `${i + 1}. ${name}` + (desc ? ` — ${desc}` : "");
    lines.push(line); spans.push([cursor, cursor + line.length]); cursor += line.length + 1;
  });
  const text = prefix + lines.join("\n") + "\n</options>\n</question>\n<answer>";
  return { text, kind: q.type, labels: pairs.map((p) => p[0]), descriptions: pairs.map((p) => p[1]), spans };
}

const utf8 = new TextEncoder();
const byteLen = (s) => utf8.encode(s).length;

/** Byte-level BPE: every char of a token string is exactly one byte of the input. */
function tokenByteOffsets(tokens) {
  const out = []; let pos = 0;
  for (const t of tokens) { out.push([pos, pos + t.length]); pos += t.length; }
  return out;
}

/** Tokenise state and question separately (as infer.py _fit does) and find option positions. */
export function buildInputs(tokenizer, state, question, maxLength = 4096) {
  const rq = renderQuestion(question);
  const qEnc = tokenizer.encode(rq.text, { add_special_tokens: false });
  const sEnc = tokenizer.encode(renderState(state), { add_special_tokens: false });
  let qIds = qEnc.ids, offs = tokenByteOffsets(qEnc.tokens);
  const reserve = Math.min(qIds.length, Math.max(1, Math.floor(maxLength * 0.75)));
  const cut = Math.max(0, qIds.length - reserve);
  qIds = qIds.slice(cut); offs = offs.slice(cut);
  const sIds = sEnc.ids.slice(0, Math.max(1, maxLength - reserve));
  const optIdx = rq.spans.map(([a, b]) => {
    const A = byteLen(rq.text.slice(0, a)), B = byteLen(rq.text.slice(0, b));
    let last = -1;
    offs.forEach(([lo, hi], j) => { if (hi > lo && lo >= A && hi <= B) last = j; });
    if (last < 0) throw new Error("option truncated out of the prompt");
    return sIds.length + last;
  });
  return { ids: [...sIds, ...qIds], optIdx, rq, stateLen: sIds.length };
}

/** derive_confidence / derive_score_confidence from schema.py */
export function readAnswer(rq, probs, ordinalSmoothing = 0.1) {
  const by = Object.fromEntries(rq.labels.map((l, i) => [l, probs[i]]));
  if (rq.kind === "noul") return { type: "noul", noul: by.true, probabilities: by };
  if (rq.kind === "choice") {
    const n = probs.length, pmax = Math.max(...probs);
    const choice = rq.labels[probs.indexOf(pmax)];
    return { type: "choice", choice, probabilities: by, confidence: n <= 1 ? 1 : Math.max(0, Math.min(1, (n * pmax - 1) / (n - 1))) };
  }
  const L = probs.length, ordered = [...Array(L).keys()].map((i) => by[String(i)]);
  const mean = ordered.reduce((s, p, i) => s + i * p, 0);
  return { type: "score", score: mean, probabilities: by, confidence: scoreConfidence(ordered, ordinalSmoothing) };
}

function scoreConfidence(probs, eps) {
  const n = probs.length; if (n <= 1) return 1;
  const total = probs.reduce((a, b) => a + b, 0); if (total <= 0) return 0;
  const p = probs.map((x) => x / total);
  const mean = p.reduce((s, pi, i) => s + i * pi, 0);
  const sigma = Math.sqrt(p.reduce((s, pi, i) => s + pi * (i - mean) ** 2, 0));
  const sMax = (n - 1) / 2, sFloor = eps > 0 ? Math.sqrt(eps) : 0;
  if (sMax <= sFloor) return sigma <= sFloor ? 1 : 0;
  return Math.max(0, Math.min(1, (sMax - sigma) / (sMax - sFloor)));
}
