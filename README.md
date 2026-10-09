# Strands Decider 2B in the browser

[Strands Decider](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) (a Qwen3.5-2B hybrid torso
with a LoRA and a pointer head) running entirely on your GPU in Chrome. No inference server and no API calls: the
weights download once, are cached in the browser's private storage (OPFS), and every decision runs locally on WebGPU.

**Live demo:** see the GitHub Pages link in the repository sidebar. It needs Chrome or Edge with WebGPU, and the
first visit downloads about 470 MB (a compressed build of v21, [below](#the-hosted-build)).

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
| engine, portable matmul (stock Chrome) | 12.5 | 22 | 49 | 92 | 324 | 1292 |

\* The engine's fastest matmul uses `chromium-experimental-subgroup-matrix`. Stock Chrome exposes that feature only
with `--enable-unsafe-webgpu`. Without it, the engine uses a portable tiled matmul (shared-memory tiles, f16 dot
products) that gives the same answers: about 60 ms for the example above, 1.3 s at 2,048 tokens.

### How the engine works

- **Weights:** int4 symmetric round-to-nearest, block 32, in ORT's MatMulNBits layout. Projections that share an
  input are fused (qkv/z/a/b, q/k/v, gate/up). Total size is 1.06 GB in two shards.
- **Matmul:** split-K across workgroups, so small prompts still fill the GPU; coalesced 16-byte weight-block loads;
  8×8×8 f16 subgroup-matrix MMAs; tile heights matched to the prompt length. For M ≤ 8 it uses a multi-row GEMV.
  Without subgroup-matrix: the GEMV up to 20 rows, then a register-tiled GEMM (TM×64 tiles, 16×16 threads,
  `web/public/gemmbench/portable.html`).
  The tile table was tuned with a correctness check (`web/public/engine/tune.html`).
- **Gated-delta recurrence:** one subgroup per 8 value columns. Each lane holds a 4×8 slice of the 128×128 state in
  registers, and every reduction is a `subgroupAdd`, so the loop needs no barriers.
- **Without 32-wide subgroups:** the subgroup kernels assume exactly 32 lanes, which Apple and NVIDIA GPUs report but
  most others (Intel, AMD, mobile) do not guarantee. There, every reduction goes through workgroup memory instead, and
  the recurrence batches its per-token reductions into two rounds. Forced on the M4 Pro (`?subgroups=0`), this gives
  the same JevBench answers and costs about 5% (39, 212 and 844 ms on the hosted build at 68, 512 and 2,048 tokens).
  It has not been run on a GPU that needs it.
- **Attention:** flash-style causal GQA with an online softmax and 16 queries per workgroup.
- **Prefix cache:** recurrent state, conv history and K/V can be snapshotted and resumed. A prefix plus suffix gives
  bit-identical results to a full forward. Five questions about one ~1,500-token state take 239 ms once the state is
  cached, against 3.5 s for five separate calls.

It was checked layer by layer against a fake-quantized fp32 PyTorch run (`web/public/engine/test.html`): every
layer is within 1.6% max relative error.

### The hosted build

The hosted demo runs an unofficial compressed build of
[strands-decider-2B-hobson-v21](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) (weights and
model card: [alxnahas/strands-decider-v21-webgpu](https://huggingface.co/alxnahas/strands-decider-v21-webgpu)).
The engine reads its format from `manifest.json`:

- Decoder layers 14 to 22 are replaced by one trained linear block (layer kind `adapter`).
- The residual stream is Hadamard-rotated and the linears are GPTQ int4; `config.unrotate` takes the final hidden
  state back to the head's basis.
- The embedding keeps the full 248k-token vocabulary at int3, but the download holds only the 32k most common rows
  (`embed_bundle.bin`). The engine fetches any other row from `embed_rows.bin` with an HTTP range request the first
  time a prompt uses it, and maps token ids to slots in a GPU table, so the embedding kernel is unchanged. The demo
  prefetches its examples' rows after loading and shows fetch time apart from the forward pass.
- The download is packed (`engine/wire.js`): every fp16 group scale is an 8-bit code on a per-row log scale, the
  shard (`weights.0.wire.gz`) holds the tensors back to back, and the shard and tokenizer are gzipped. The engine
  decompresses with `DecompressionStream` and rebuilds the fp16 layout before uploading, so the kernels are
  unchanged. The cache keeps the packed bytes; unpacking takes about 0.3 s per load.

The download is about 470 MB. Run through this engine in headless Chrome (stock or flagged, same answers), the
build scores 176 on JevBench's 231 public tasks, the same as v21 in bfloat16 (v21 in the int4 format above: 163).
On 48 of those tasks translated into zh, ja, ko, ar, hi, uk, de and pl, it is within 3 of v21's count in every
language. On held-out OpenBookQA (3,000 tasks) and BoolQ (3,270), never used to choose a build, an MLX simulation
of the same weights is 0.3 and 0.9 points below v21 (BoolQ 95% interval −1.5 to −0.3). GPU time per forward on the
M4 Pro: 29 ms at 68 tokens, 161 ms at 512, 646 ms at 2,048 (stock Chrome, portable matmul: 37, 201 and 800 ms). The scripts that produce the export (layer removal, GPTQ, export) are not in this repository yet.

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

`--from <dir>` takes the assets from an export directory instead (`model/`, `engine-weights/`, and optionally a
`README.md` and `LICENSE.md` for the repo), as for the hosted build.

Set the repository variable `DECIDER_ASSETS` to that `resolve/main/` URL. The `pages` workflow then builds and
deploys the site on every push to `main`. The browser cache is keyed by the assets URL and a fingerprint of the
manifest, so new weights never load stale cached shards; older entries are deleted after a successful load.
`test/pages.spec.mjs` checks the same setup locally: the site on a subpath with no custom headers, the weights from a
second origin with byte ranges, in stock Chrome (`PAGES_FROM=<export dir>` builds it with `--from`).

## Credits and license

- Model:
  [StrandsAgents/strands-decider-2B-hobson-v19](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)
  and [v21](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) (Apache-2.0), on
  [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (Apache-2.0). The converted weights keep that
  license.
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT) and
  [@huggingface/tokenizers](https://github.com/huggingface/tokenizers) (Apache-2.0).
- This repository: Apache-2.0 (`LICENSE`).
