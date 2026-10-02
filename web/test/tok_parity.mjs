// Node check: JS prompt rendering + tokenisation + opt_idx == Python reference.
import fs from "node:fs";
import { Tokenizer } from "@huggingface/tokenizers";
import { buildInputs } from "../public/js/prompt.js";
const M = "../models/decider";
const tok = new Tokenizer(JSON.parse(fs.readFileSync(`${M}/tokenizer.json`)), JSON.parse(fs.readFileSync(`${M}/tokenizer_config.json`)));
const fx = JSON.parse(fs.readFileSync("../ref/fixtures.json"));
const ref = JSON.parse(fs.readFileSync("../ref/reference_cpu.json")).results;
let ok = true;
fx.forEach((f, i) => {
  const { ids, optIdx } = buildInputs(tok, f.state, f.question);
  const same = JSON.stringify(ids) === JSON.stringify(ref[i].input_ids) && JSON.stringify(optIdx) === JSON.stringify(ref[i].opt_idx);
  ok &&= same; console.log(f.name, same ? "OK" : `MISMATCH js=${ids.length}/${optIdx} py=${ref[i].input_ids.length}/${ref[i].opt_idx}`);
});
process.exit(ok ? 0 : 1);
