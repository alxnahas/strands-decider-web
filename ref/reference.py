"""Official strands-decider reference: dumps prompt, token ids, opt_idx, logits, probs."""
import json, sys, time, torch
from strands_decider import modeling
from strands_decider.infer import load_engine
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
_orig = modeling.StrandsDeciderConfig.from_json
def patched(path):
    c = _orig(path); c.base_model = "models/qwen-base"; return c
modeling.StrandsDeciderConfig.from_json = staticmethod(patched)
device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
FX = sys.argv[2] if len(sys.argv) > 2 else "ref/fixtures.json"; fixtures = json.load(open(FX)); TAG = "" if FX.endswith("fixtures.json") else "_bench"
t = time.time(); eng = load_engine("models/decider", device=device); load_s = time.time() - t
out = {"device": device, "torch": torch.__version__, "load_s": load_s, "results": []}
for fx in fixtures:
    q = fx["question"]
    if q["type"] == "choice": Q = ChoiceQuestion(instructions=q["instructions"], criteria={o: "" for o in q["options"]})
    elif q["type"] == "noul": Q = NoulQuestion(instructions=q["instructions"])
    else: Q = ScoreQuestion(instructions=q["instructions"], criteria=q["options"])
    rq = render_question(Q); st = render_state(fx["state"])
    s, qq = eng._fit(st, [rq.text]); opt = eng._option_idx([rq], len(s))[0].tolist()
    t = time.time(); resp = eng.ask(fx["state"], {"q": Q}); dt = time.time() - t
    ans = resp.answers["q"].model_dump()
    out["results"].append({"name": fx["name"], "prompt": st + rq.text, "input_ids": s + qq[0], "opt_idx": opt,
                           "kind": rq.kind, "slot_labels": list(rq.slot_labels), "answer": ans, "infer_s": dt})
    print(fx["name"], json.dumps(ans), f"{dt:.2f}s", flush=True)
json.dump(out, open(f"ref/reference{TAG}_{device}.json", "w"), indent=1)
