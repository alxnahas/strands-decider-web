# Strands Decider 2B in the browser

[Strands Decider](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) (a Qwen3.5-2B hybrid torso
with a LoRA and a pointer head) running entirely on your GPU in Chrome. No inference server and no API calls: the
weights download once, are cached in the browser's private storage (OPFS), and every decision runs locally on WebGPU.

**Live demo:** see the GitHub Pages link in the repository sidebar. It needs Chrome or Edge with WebGPU, and the
first visit downloads about 1 GB.

Two backends:

- **A hand-written WebGPU engine** (`web/public/engine/`): about 10 WGSL kernels and no runtime dependency. This is
  what the hosted demo uses.
- **ONNX Runtime Web 1.30**, running the ORT GenAI export of the same model. It is available in the local server.

## Results (Apple M4 Pro, Chrome stable)

The official example asks *"Help! My payouts have been failing for 3 days!"* which team should handle it
(billing / sales / retail).

| Runtime | billing | sales | retail | forward |
|---|---|---|---|---|
| PyTorch CPU fp32 (official `strands-decider`) | 0.844 | 0.064 | 0.091 | 4.2 s |
| Browser, hand-written engine, int4 | 0.867 | 0.054 | 0.079 | 47.5 ms* |
| Browser, ONNX Runtime Web, int4 | 0.877 | 0.049 | 0.073 | 81 ms |

On a 27-item benchmark (`ref/bench.json`: choice, true/false and score questions, permuted options, states up to
2,048 tokens), both int4 backends pick the same top answer as Python on 27/27. Mean absolute probability error is
0.019 for both. The error comes from int4 quantization; an int8 ORT build gets 0.006.

Raw forward latency in ms:

| tokens | 8 | 16 | 64 | 128 | 512 | 2048 |
|---|---|---|---|---|---|---|
| ONNX Runtime Web | 40 | 43 | 66 | 109 | 362 | 1455 |
| engine with subgroup-matrix* | 12.8 | 14.8 | 38.7 | 72 | 272 | 1046 |

\* The engine's fast matmul uses `chromium-experimental-subgroup-matrix`. Stock Chrome exposes that feature only
with `--enable-unsafe-webgpu`. Without it, the engine uses a portable matmul that is just as accurate (27/27) but
slower: 91 ms for the example above and 2.9 s at 2,048 tokens. The local demo picks ORT in that case.

### How the engine works

- **Weights:** int4 symmetric round-to-nearest, block 32, in ORT's MatMulNBits layout. Projections that share an
  input are fused (qkv/z/a/b, q/k/v, gate/up). Total size is 1.06 GB in two shards.
- **Matmul:** split-K across workgroups, so small prompts still fill the GPU; coalesced 16-byte weight-block loads;
  8×8×8 f16 subgroup-matrix MMAs; tile heights matched to the prompt length. For M ≤ 8 it uses a multi-row GEMV.
  The tile table was tuned with a correctness check (`web/public/engine/tune.html`).
- **Gated-delta recurrence:** one subgroup per 8 value columns. Each lane holds a 4×8 slice of the 128×128 state in
  registers, and every reduction is a `subgroupAdd`, so the loop needs no barriers.
- **Attention:** flash-style causal GQA with an online softmax and 16 queries per workgroup.
- **Prefix cache:** recurrent state, conv history and K/V can be snapshotted and resumed. A prefix plus suffix gives
  bit-identical results to a full forward. Five questions about one ~1,500-token state take 239 ms once the state is
  cached, against 3.5 s for five separate calls.

It was checked layer by layer against a fake-quantized fp32 PyTorch run (`web/public/engine/test.html`): every
layer is within 1.6% max relative error.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
./convert/build.sh            # download the pinned checkpoints, merge the LoRA, export ONNX + engine weights
cd web && npm install
node server.mjs               # http://127.0.0.1:8787/demo.html
npx playwright test           # parity vs Python, demo in stock and flagged Chrome, Pages build, agent loop
```

The tests run installed Google Chrome. `CHROME=stock` runs them without WebGPU flags, and `BACKEND=engine` runs the
parity tests on the engine. `ref/` holds the Python reference outputs; `ref/reference.py` regenerates them and
needs the official [`strands-decider`](https://github.com/strands-agents) package.

`web/loop/` is a small observe → candidates → choose → policy → execute loop. It drives a sandbox page through
Playwright, with the decider choosing each action (`node loop/run-loop.mjs`).

## Deploy (GitHub Pages + Hugging Face)

GitHub Pages can't hold the weights: sites are capped at 1 GB and git rejects files over 100 MB. So the model files
go to a Hugging Face model repo, which serves them with CORS, and Pages serves only the 0.3 MB site.

```bash
cd web && node build-pages.mjs --out ../dist --assets https://huggingface.co/<user>/<repo>/resolve/main/
hf upload <user>/<repo> ../dist/assets . --repo-type model
```

Set the repository variable `DECIDER_ASSETS` to that `resolve/main/` URL. The `pages` workflow then builds and
deploys the site on every push to `main`. `test/pages.spec.mjs` checks the same setup locally: the site on a
subpath with no custom headers, the weights from a second origin, in stock Chrome.

## Credits and license

- Model: [StrandsAgents/strands-decider-2B-hobson-v19](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)
  (Apache-2.0), on [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (Apache-2.0). The converted
  weights keep that license.
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT) and
  [@huggingface/tokenizers](https://github.com/huggingface/tokenizers) (Apache-2.0).
- This repository: Apache-2.0 (`LICENSE`).
