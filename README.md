# InferDrift

InferDrift compares two LLM inference stacks and tells you whether the change
between them hurts answers. For example, you can compare the same model before
and after:

- a quantization;
- an engine upgrade;
- a kernel or attention-backend change;
- a new chat template.

## The problem

A change to an inference stack is usually meant to leave the model's answers
alone. In practice small numeric differences can flip some answers from right to
wrong, and an aggregate benchmark score can hide that: a few answers that get
worse can be cancelled out by a few that get better.

InferDrift runs both stacks on the same items and couples their decoding (for
example, greedy decoding on both). It counts **harmful flips**: items the
reference stack gets right and the candidate stack gets wrong. It then reports
exact one-sided bounds on the harmful-flip rate and gives one verdict at the
tolerance you choose.

Grading uses exact inheritance. When the candidate's output is byte-identical to
the reference output, a deterministic grader would give it the same grade, so
the grade is reused instead of recomputed. A seeded random share of the reused
grades is still re-checked. Every reference output is still graded once. For the
candidate, grading cost therefore grows with the number of outputs that differ,
plus the audit share (by default about 5% of the reused grades), not with the
full size of the item bank.

## Status

**This is an early preview, version 0.1.0.dev0.** Interfaces, report fields and
defaults may change without notice. It is not published on the Python package
index; install it from source (see below).

- **Support: tested on one laptop only (with an RTX 5090 Laptop GPU).** All live testing so
  far ran on a single laptop with that GPU, and the completed live comparisons
  ran on its CPU: the built-in transformers adapter runs models on the CPU, and
  the llama.cpp server was a CPU-only build. No GPU-served stack has yet been
  compared end to end. No other hardware, driver or operating system has been
  tested.
- **Known limitation, being fixed.** Under the current verdict rule, a noisy
  engine can make a harmless change look harmful. With greedy decoding, an engine
  that is not deterministic from run to run produces flips of its own, and the
  verdict counts them as harmful flips caused by the change. The report flags
  engines whose repeated greedy outputs differ, but the verdict does not subtract
  that noise. We are changing the default criterion to the excess harmful-flip
  rate over a measured noise floor, plus the net accuracy change. That change has
  not shipped yet. Until it does, treat a `DEVIATES` verdict on a stack flagged
  as non-deterministic with caution.
- **Experimental: `compare --plan`.** The requalification plan feature
  (`inferdrift/requalify.py`, `inferdrift/plan_audit.py`,
  `inferdrift/plan_stats.py`) did not meet its validation bar. This release
  ships no command that writes a plan, and the plan format may change.

How each adapter has been exercised so far:

| adapter | status |
|---|---|
| Hugging Face transformers (reference, runs on the CPU) | exercised live |
| llama.cpp server | exercised live (CPU-only llama-server build) |
| OpenAI-compatible endpoints | exercised against the local stand-in only |
| vLLM | implemented; exercised against the stand-in's emulation of its protocol; not yet validated against a live vLLM server |
| SGLang | implemented; exercised only against the stand-in's emulation of its protocol |

The statistical method combines standard, well-known pieces: paired
comparison, Clopper-Pearson bounds and a Bonferroni split over planned looks.
No claim of novelty or priority is made.

## Install from source

Python 3.10 or later on Linux. The core package has no third-party
dependencies.

```bash
git clone https://github.com/ChanYCTobias/InferDrift.git
cd InferDrift
python -m pip install .
```

These commands need network access. To compare Hugging Face models locally you
also need the `hf` extra. Install the CPU build of PyTorch first, so that the
extra does not pull a CUDA build (this also needs network access):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[hf]"
```

From a checkout you can also run the tool without installing it, as
`python -m inferdrift ...` from the repository root.

## Offline quickstart

The package includes a local stand-in endpoint. It speaks the OpenAI-compatible
chat API on 127.0.0.1 and needs no model, no network and no API key. Its default
`synthetic` backend returns deterministic placeholder text derived from a hash of
each request. The quickstart therefore checks the plumbing, not a model: most
reference answers are wrong, and two identical stand-ins agree on every item.

```bash
inferdrift standin --port 8099 &
REF_PID=$!
inferdrift standin --port 8100 &
CAND_PID=$!
sleep 1
inferdrift compare --ref http://127.0.0.1:8099/v1 --cand http://127.0.0.1:8100/v1 --out demo
cat demo/summary.md
kill $REF_PID $CAND_PID
```

`compare` prints one JSON line. Here the verdict is `EQUIVALENT_WITHIN_TAU` on
500 items (470 if the code suite is skipped because user namespaces are
unavailable; see below). The run directory `demo/` holds `report.json` (the full
report), `items.jsonl` (one record per item), `events.jsonl` and `summary.md`.

The stand-in can inject faults, so you can see how a report flags them. For
example, start the candidate with `--nondeterministic 0.3` (it perturbs greedy
outputs at random), or start it with `--ignore-seed` and run
`compare --decode sample --seed 3`, then compare the reports. The test suite in
`tests/` builds replay files with known answers and shows `DEVIATES` verdicts as
well.

## Commands

| command | what it does | network |
|---|---|---|
| `inferdrift compare --ref SPEC --cand SPEC` | runs the comparison and writes the report | only if a spec needs it |
| `inferdrift inspect TARGET` | passive profile of a Hub repository, a local model directory, a local GGUF file, or an endpoint: JSON, safetensors headers and GGUF metadata only; code, hooks and pickles are listed, never executed | for Hub repositories and remote endpoints |
| `inferdrift channels SPEC` | detects prompt log-probabilities, top-k log-probabilities, seed honouring, greedy repeat identity, usage reporting and max-token honouring | only if the spec needs it |
| `inferdrift plan-downloads --ref SPEC [--cand SPEC]` | shows, or refuses, the downloads a comparison would need | yes (Hub listing) |
| `inferdrift standin --port PORT` | local metered stand-in endpoint for testing caps and adapters, with fault injection (`--ignore-seed`, `--no-logprobs`, `--truncate-frac`, `--fail-every`, `--bad-usage`, `--nondeterministic`, ...) | no |
| `inferdrift banks` | lists the built-in item banks | no |
| `inferdrift profile-diff A B` | diffs two stack profiles or two reports | no |

Run `inferdrift COMMAND --help` for every option.

A stack spec is one of:

- `hf:org/name[@revision][?attn=sdpa&dtype=float32&...]`: a Hugging Face model
  run locally with transformers (needs the `hf` extra);
- `hf:/path/to/local/model`: the same, from a local directory;
- `http(s)://host[:port][/v1][?dialect=...&model=...]`: an endpoint. The dialect
  is `openai`, `llamacpp`, `vllm`, `sglang` or `auto`.

For the tool-calling items, start vLLM with `--enable-auto-tool-choice` and a
`--tool-call-parser` that fits the model.

API keys are never part of a spec. Put the key in an environment variable and
pass that variable's name, for example `--cand-api-key-env MY_KEY`.

Examples that download a model or call a remote endpoint (these need network
access and, for `hf:` specs, the `hf` extra):

```bash
# an attention-backend change, both stacks local on the CPU
inferdrift compare --ref "hf:Qwen/Qwen3-0.6B?attn=sdpa" --cand "hf:Qwen/Qwen3-0.6B?attn=eager" --out run1

# a local reference against an OpenAI-compatible endpoint, with a hard dollar cap
export CAND_KEY=...   # only the variable's name goes on the command line
inferdrift compare --ref "hf:Qwen/Qwen3-0.6B" \
  --cand "https://my-endpoint.example/v1?model=qwen3-0.6b" --cand-api-key-env CAND_KEY \
  --cand-price 0.10,0.40 --max-usd 0.50 --out run2
```

Plain OpenAI-style chat requests cannot carry chat-template arguments such as
`enable_thinking`. Add `send_template_kwargs=1` to an endpoint spec when the
server accepts `chat_template_kwargs`. Use `mode=raw` with `tokenizer=hf:...`
(needs the `hf` extra) to render prompts on the client side and send token ids
instead.

## What a run does

1. **Profiles both stacks.** Where they can be observed, it records the model id
   and revision, weight dtype, quantization, KV-cache dtype, engine and version,
   attention and GEMM backends, hardware, parallelism, speculative decoding,
   chat-template and tokenizer hashes, decoding parameters and output caps. Each
   field says where it came from, or that it is `unobservable`.
2. **Detects channels and couples decoding.** Greedy decoding is coupled by
   construction. Seeded sampling uses shared Gumbel noise when both stacks run
   in-process. Without verified shared randomness, for example an endpoint that
   ignores the seed, it samples independently, says so, and bounds the excess
   harmful-flip rate over a second, independent reference run, with wider
   bounds.
3. **Evaluates items sequentially.** Items run in a seeded random order. The plan
   has looks at n = 32, 64, 128, ... and at n_max. `DEVIATES` is tested at every
   look at alpha/K, where K is the number of looks, and the run stops at the
   first one. `EQUIVALENT_WITHIN_TAU` is tested only at n_max, at alpha. Both
   use exact Clopper-Pearson bounds, computed with the Python standard library.
4. **Grades with exact inheritance**, as described above. A seeded 5% audit
   (`--audit-fraction`) re-grades inherited outputs; audit calls are counted as
   overhead.
5. **Meters cost.** Hard dollar, token and wall-time caps apply. Every request is
   reserved at its worst case before it is sent, so the run stops at or below the
   cap. A dollar cap needs a price for every endpoint stack.

Built-in item banks, all generated procedurally by `inferdrift/banks/generators.py`
(no dataset download):

- `core`, 500 items: short math (numeric answers), multiple choice, tool calling,
  instruction following (verifiable checks) and unit-tested code;
- `full`: `core` plus 8 needle-in-a-haystack items of about 16k tokens;
- `smoke`: the first 8 items of each `core` suite.

You can also pass your own JSONL bank with `--bank path/to/bank.jsonl`. Code
items run in a sandbox without network access, using user and network
namespaces. When namespaces are unavailable, the code suite is skipped unless you
pass `--sandbox-weak-ok`, which blocks sockets inside Python instead.

## How to read a verdict

You set two numbers. `--tau` is the largest harmful-flip rate you would accept
(default 0.01, one percentage point). `--alpha` is the error rate of the verdict
(default 0.05).

- `EQUIVALENT_WITHIN_TAU`: the run reached n_max, and the one-sided upper bound
  on the harmful-flip rate is at most tau. The change is unlikely to flip more
  than tau of answers from right to wrong on items like those in the bank.
- `DEVIATES`: at some look, the one-sided lower bound on the harmful-flip rate
  was above tau. The change likely flips more than tau of answers from right to
  wrong. See the known limitation above if a stack is flagged as
  non-deterministic.
- `INCONCLUSIVE`: neither holds. Usually there were too few items for the
  chosen tau (with no harmful flips at all you need roughly 300 items for
  tau = 0.01 at alpha = 0.05), or a cap stopped the run early.

The verdict is about the items in the bank and the decoding settings you used,
and the bounds assume the items are independent draws from the distribution the
bank represents.

Where to look in a run directory:

- `summary.md`: verdict, harmful/helpful flip rates and accuracy change with
  their bounds, per-suite churn, grading cost, channels, coupling and cost;
- `report.json`: everything, including `verdict.bound`, `looks` (the bound at
  each look), `coupling.flags`, `stacks.diff` (profile differences between the
  stacks) and `limitations` (anything that weakens the result);
- `items.jsonl`: each item's two outputs, grades and flip (-1 harmful, +1
  helpful, 0 none).

Exit codes: `0` run completed (with `--exit-on-verdict`: `0`
EQUIVALENT_WITHIN_TAU, `10` DEVIATES, `11` INCONCLUSIVE); `3` refused before any
work started, with a JSON line naming the bad input; `4` error. Output on stdout
is strict JSON. A few environment errors, such as an output path that cannot be
written, still end with a Python traceback and exit code 1.

## Safety

- Repository code is never executed: no `trust_remote_code`, no pickle weights.
  Repositories that declare custom code are refused. Chat templates are rendered
  only by transformers' sandboxed Jinja environment, when a model is actually run.
- API keys are read from the environment variables you name, and scrubbed from
  every report, log and error message.
- Downloads are planned and summed first. A job above the transient allowance
  (default 6 GiB, `--transient-cap-gib`) or the free disk space is refused before
  any byte is written, and the work directory is removed at the end.
- Every numeric input has a closed domain and is checked before anything is sent
  or written. Values such as `nan`, `inf` and `1e999` are refused, not coerced.

## Tests

```bash
python -m pip install pytest
python -m pytest
```

Installing pytest needs network access. The tests themselves run offline against
the local stand-in, in about 20 seconds.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md). Please report security problems
privately, as described in [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
