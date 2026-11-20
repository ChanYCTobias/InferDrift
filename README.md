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
example, greedy decoding on both). It counts **harmful flips** (items the
reference stack gets right and the candidate stack gets wrong) and **helpful
flips** (the reverse). An engine can also disagree with itself: the same request
can give a different answer on a second run. InferDrift measures that **noise
floor** for each stack, by running items twice, and judges the change against it.

You choose what the verdict certifies:

- **`net_change`** (the default) bounds the net accuracy change of the
  candidate, per capability (the item's suite). Inside one capability, helpful
  flips offset harmful ones, so this is the "is it worse on balance" question.
- **`harmful_excess`** bounds the rate of harmful flips beyond the reference's
  own noise floor. Helpful flips do not offset harmful ones.
- **`harmful_abs`** is the 0.1.0 rule: it bounds the harmful-flip rate
  itself. Use it only for stacks that are deterministic: on a noisy engine it
  counts the engine's own noise as harm.

Each gives one verdict at the tolerance you choose, from exact bounds (except
`harmful_abs` under sampled decoding without verified shared randomness; see
step 2 of "What a run does"). Under the default and `harmful_excess` criteria,
if the item bank is too small for that tolerance, the verdict says so and states
how many items would be needed (or, past the search's limit of 20,000 items in
the largest capability, the size at which the search stopped), instead of a pass
or a fail.

Grading uses exact inheritance. When the candidate's output is byte-identical to
the reference output, a deterministic grader would give it the same grade, so
the grade is reused instead of recomputed. A seeded random share of the reused
grades is still re-checked. Every reference first-run output is graded once
per run, or, with the reference cache (`--ref-cache`), once across runs, again
with a seeded share re-checked. A reference second run taken from the cache is
graded again on each run (it inherits the first run's grade when the two
outputs are byte-identical and the grader is deterministic). For the candidate,
grading cost therefore grows with the number of outputs that differ, plus the
audit share (by default about 5% of the reused grades), not with the full size
of the item bank.

## Status

**This is an early preview, version 0.2.0.dev0.** Interfaces, report fields and
defaults may change without notice. It is not published on the Python package
index; install it from source (see below).

- **Support: tested on one laptop only (with an RTX 5090 Laptop GPU).** All live testing so
  far ran on a single laptop with that GPU, and the completed live comparisons
  ran on its CPU: the built-in transformers adapter runs models on the CPU, and
  the llama.cpp server was a CPU-only build. No GPU-served stack has yet been
  compared end to end. No other hardware, driver or operating system has been
  tested.
- **New in 0.2 (not yet validated on a live engine).** In 0.1.0 a noisy engine
  could make a harmless change look harmful: an engine that is not deterministic
  from run to run produces flips of its own, and the verdict counted them as
  harm. Version 0.2 measures each stack's noise floor and runs the reference
  twice on every item when a 64-item probe (or the channel probe) sees a
  stack's outputs differ between two runs; a low floor can go unseen on 64
  items, and the report states the floor with the number of items it rests on.
  It changes the default criterion to the net accuracy change per capability
  (see "How to read a verdict"). It also adds a power check
  (`INCONCLUSIVE_UNDERPOWERED` with the number of items needed),
  deterministic-mode reporting, a reference cache and the `inferdrift power`
  command. What is validated so far: all of this is tested offline against the
  local stand-in only, including its emulation of noisy and deterministic
  engines. It has not yet been run against a live vLLM, llama.cpp or SGLang
  server, with the transformers adapter, or on a GPU. The deterministic-mode
  advice is generic: it names no engine-specific setting yet. On a
  deterministic stack, `--criterion harmful_abs` gives the 0.1.0 verdicts.
- **Experimental: `compare --plan`.** The requalification plan feature
  (`inferdrift/requalify.py`, `inferdrift/plan_audit.py`,
  `inferdrift/plan_stats.py`) did not meet its validation bar. This release
  ships no command that writes a plan, and the plan format may change.

How each adapter has been exercised so far (the live runs were of 0.1.0; no
0.2 code path has run against a live engine yet):

| adapter | status |
|---|---|
| Hugging Face transformers (reference, runs on the CPU) | exercised live |
| llama.cpp server | exercised live (CPU-only llama-server build) |
| OpenAI-compatible endpoints | exercised against the local stand-in only |
| vLLM | implemented; exercised against the stand-in's emulation of its protocol; not yet validated against a live vLLM server |
| SGLang | implemented; exercised only against the stand-in's emulation of its protocol |

The statistical method combines standard, well-known pieces: paired
comparison, Clopper-Pearson bounds, an exact paired bound built from them,
McNemar-style conditioning on the discordant items, a Bonferroni split over
capabilities and planned looks, and an intersection-union rule for the pass. No
claim of novelty or priority is made.

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

`compare` prints one JSON line. Here the verdict is `INCONCLUSIVE_UNDERPOWERED`
on 500 items (470 if the code suite is skipped because user namespaces are
unavailable; see below). The stand-ins agree on every item, but the default
tolerance is one point of net accuracy per capability, and the core bank's five
capabilities are too small for that: with no noise at all, the exact bound needs
597 items per capability, and the smallest capability has 30. The JSON line and
`summary.md` give the number of items needed (9,944 in the bank's proportions, in
our run), and `summary.md` also gives the tolerance this bank supports (about 18
points per capability, limited by the 30 code items). Two variations:

```bash
# one capability with a tolerance its 140 items support: EQUIVALENT_WITHIN_TAU
inferdrift compare --ref http://127.0.0.1:8099/v1 --cand http://127.0.0.1:8100/v1 \
  --bank math_short --tau 0.05 --out demo_math
# the 0.1.0 rule (harmful-flip rate, deterministic stacks): EQUIVALENT_WITHIN_TAU
inferdrift compare --ref http://127.0.0.1:8099/v1 --cand http://127.0.0.1:8100/v1 \
  --criterion harmful_abs --out demo_abs
```

`inferdrift power --tau 0.05` prints the number of items needed for one
capability with no noise (117 items); add `--floor 0.03` for an engine whose
answers flip on 3% of repeated requests.

The run directory `demo/` holds `report.json` (the full report), `items.jsonl`
(one record per item), `events.jsonl` and `summary.md`.

The stand-in can inject faults, so you can see how a report flags them. For
example, start the candidate with `--nondeterministic 0.3` (it perturbs greedy
outputs at random, so both stacks are then run twice on every item), or start it
with `--ignore-seed` and run `compare --decode sample --seed 3`, then compare the
reports. A replay file can also emulate a noisy engine: a line with `alt` and
`flip` serves its alternative answer with that probability (see
`inferdrift/standin.py`; `--noise-seed` picks the engine's noise). The test
suite in `tests/` builds replay files with known answers and shows `DEVIATES`
verdicts as well.

## Commands

| command | what it does | network |
|---|---|---|
| `inferdrift compare --ref SPEC --cand SPEC` | runs the comparison and writes the report | only if a spec needs it |
| `inferdrift inspect TARGET` | passive profile of a Hub repository, a local model directory, a local GGUF file, or an endpoint: JSON, safetensors headers and GGUF metadata only; code, hooks and pickles are listed, never executed | for Hub repositories and remote endpoints |
| `inferdrift channels SPEC` | detects prompt log-probabilities, top-k log-probabilities, seed honouring, greedy repeat identity, usage reporting and max-token honouring | only if the spec needs it |
| `inferdrift plan-downloads --ref SPEC [--cand SPEC]` | shows, or refuses, the downloads a comparison would need | yes (Hub listing) |
| `inferdrift power --tau T [--floor F]` | items needed for a tolerance at a noise floor, from the exact bound `compare` uses (also `--criterion`, `--alpha`, `--power`, `--capability-count`, `--items`) | no |
| `inferdrift standin --port PORT` | local metered stand-in endpoint for testing caps and adapters, with fault injection (`--ignore-seed`, `--no-logprobs`, `--truncate-frac`, `--fail-every`, `--bad-usage`, `--nondeterministic`, ...) and a replay noise model (`--noise-seed`) | no |
| `inferdrift banks` | lists the built-in item banks | no |
| `inferdrift profile-diff A B` | diffs two stack profiles or two reports | no |

Run `inferdrift COMMAND --help` for every option. Options of `compare` added in
0.2: `--criterion`, `--power`, `--floor-probe-items`, `--plan-floor`,
`--replicate`, `--ref-cache` and `--stop-if-underpowered`.

A stack spec is one of:

- `hf:org/name[@revision][?attn=sdpa&dtype=float32&...]`: a Hugging Face model
  run locally with transformers (needs the `hf` extra);
- `hf:/path/to/local/model`: the same, from a local directory;
- `http(s)://host[:port][/v1][?dialect=...&model=...]`: an endpoint. The dialect
  is `openai`, `llamacpp`, `vllm`, `sglang` or `auto`.

For the tool-calling items, start vLLM with `--enable-auto-tool-choice` and a
`--tool-call-parser` that fits the model.

If you know whether an endpoint's engine runs in a deterministic or
batch-invariant mode, add `deterministic=on` (or `off`) to its spec; the report
states it as declared and flags a declared `on` whose outputs differ between runs.

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
   ignores the seed, it samples independently and says so; the reference then
   runs twice on every item. (Under `--criterion harmful_abs`, as in 0.1.0, the
   verdict then bounds the excess harmful-flip rate over that second run with
   wider, unpaired bounds. When both endpoints honour the seed and run the same
   sampler, this 0.1.0 construction can be too optimistic, and the report says
   so; use the default criterion for sampled decoding.)
3. **Measures the noise floor.** Under the default and `harmful_excess`
   criteria, both stacks run the first 64 planned items twice (the probe;
   `--floor-probe-items`). A stack whose outputs differ between the two runs is
   noisy. If either stack is noisy, the reference runs twice on every item, and
   a noisy candidate does too. The floor is the larger of the two stacks'
   answer-level repeat-flip rates. Second runs repeat the first run's request
   exactly under greedy decoding; under sampling without verified shared
   randomness every run gets its own seed, so a repeat pair and a cross-stack
   pair are comparable when nothing changed.
4. **Evaluates items sequentially.** Items run in a seeded random order. The plan
   has looks at n = 32, 64, 128, ... and at n_max. Per capability, `DEVIATES` is
   tested at every look at alpha/(m K), where m is the number of capabilities and
   K the number of looks, and the run stops at the first one the power check
   allows. `EQUIVALENT_WITHIN_TAU` is tested only at n_max, at alpha, and needs
   every capability within tau. Both use an exact paired bound (a two-sided
   Clopper-Pearson bound on how often the two runs disagree, combined with a
   conditional Clopper-Pearson bound on the direction of those disagreements),
   computed with the Python standard library. Before either verdict, the power
   check computes, under a simple flip model (independent items, one symmetric
   flip rate per capability at the floor's point estimate), the chance that the
   planned items pass when nothing changed. It uses the measured floor, or
   `--plan-floor` when that is given and the run does not measure a
   significantly higher one; below `--power` (default 0.9) the verdict is
   `INCONCLUSIVE_UNDERPOWERED`. With `--plan-floor` and `--floor-probe-items 0`,
   a run in which no item runs twice does not check the supplied floor at all;
   the gate then rests on that value alone, and the report says so.
   `--criterion harmful_abs` keeps the 0.1.0 rule: Clopper-Pearson bounds on the
   harmful-flip rate at alpha/K and alpha, with no probe and no power check.
5. **Grades with exact inheritance**, as described above. A seeded 5% audit
   (`--audit-fraction`) re-grades inherited outputs; audit calls are counted as
   overhead.
6. **Meters cost.** Hard dollar, token and wall-time caps apply. Every request is
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

You set three numbers. `--tau` is the largest change you would accept (default
0.01, one percentage point) on the criterion's statistic: the net accuracy loss
per capability (`net_change`, the default), the harmful-flip rate beyond the
reference's noise floor (`harmful_excess`), or the harmful-flip rate
(`harmful_abs`). `--alpha` is the error rate of the verdict (default 0.05).
`--power` is the chance of a pass, when nothing changed, that the plan must have
(default 0.9).

- `EQUIVALENT_WITHIN_TAU`: the run evaluated every planned item, and for every
  capability the one-sided upper bound on the statistic is at most tau. Under
  `net_change`: the candidate is unlikely to be more than tau less accurate than
  the reference in any capability, on items like those in the bank; inside a
  capability, helpful flips offset harmful ones.
- `DEVIATES`: at some look, for some capability, the one-sided lower bound on
  the statistic was above tau, and the plan had the requested power.
- `INCONCLUSIVE_UNDERPOWERED`: the planned items are too few for tau at the
  requested power, given the noise floor (measured, or `--plan-floor` as in step
  4). This is never a pass or a fail. The report gives the number of items
  needed, which capabilities are short, and the smallest tolerance the bank
  supports. The number is searched on what the run tested: when `--max-items`
  cut the bank, it is a `--max-items` value for the same bank, seed and item
  order; otherwise, or when even the whole bank is not enough, it is a bank size
  in the bank's proportions. It reaches the requested power while one item fewer
  does not, and every size up to 5% above it (at least 50 and at most 200 sizes,
  unless the end of the bank or the search's cap comes first) was checked to
  reach it as well. Exact power is not monotone in the number of items, and
  sizes beyond that range were not checked. If the search reaches 20,000 items
  in the largest capability without finding enough, the report gives the size
  where it stopped and the power there, instead of a number. A capability is
  short when its own power at its planned count is below `--power`; the report
  gives a size at which it reaches the power on its own (one fewer does not; the
  sizes above it are not checked, so `inferdrift power` at its floor can give a
  slightly larger number); in the joint plan it can need more. With no noise at
  all and alpha = 0.05, power 0.9, the default criterion needs 597 items per
  capability for tau = 0.01, 297 for 0.02 and 117 for 0.05; noise raises these
  numbers (`inferdrift power`). If the bounds already pointed to a deviation,
  the report says so as evidence, not as a verdict.
- `INCONCLUSIVE`: none of these holds, for example a cap stopped the run early,
  the reference failed on some items, or the bounds straddle tau. Under
  `harmful_abs` (no power check) this also covers too few items: with no harmful
  flips at all the 0.1.0 rule needs roughly 300 items for tau = 0.01 at
  alpha = 0.05.

The verdict is about the items in the bank and the decoding settings you used,
and the bounds assume the items are independent draws from the distribution the
bank represents.

Where to look in a run directory:

- `summary.md`: verdict and, when underpowered, the items needed; the noise
  floor; harmful and helpful flips, churn and net change against the floor, with
  their bounds; the verdict per capability; deterministic mode; per-suite
  churn, grading cost, channels, coupling and cost;
- `report.json`: everything, including `verdict.bound` and
  `verdict.per_capability`, `looks` (the bounds at each look), `noise` (the
  floor, who ran twice and why), `power`, `deterministic_mode`,
  `reference_cache`, `coupling.flags`, `stacks.diff` (profile differences between
  the stacks) and `limitations` (anything that weakens the result). The unpaired
  `estimates.accuracy_change` is kept from 0.1.0; under the paired criteria the
  net change is `estimates.net_change`;
- `items.jsonl`: each item's two outputs, grades and flip (-1 harmful, +1
  helpful, 0 none).

Exit codes: `0` run completed (with `--exit-on-verdict`: `0`
EQUIVALENT_WITHIN_TAU, `10` DEVIATES, `11` INCONCLUSIVE, `12`
INCONCLUSIVE_UNDERPOWERED); `3` refused, either before any work started, with a
JSON line naming the bad input, or during a run whose integrity cannot be kept
(under the default and `harmful_excess` criteria, an engine that silently
returns fewer outputs than requested; a corrupt reference cache); `4` error.
Under `harmful_abs` a short return keeps its 0.1.0 handling: the run stops at
that point and is not refused. Output on stdout is strict JSON. A few environment errors, such as an output path that cannot be
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

## Reference cache

`--ref-cache DIR` keeps the reference's outputs and grades, keyed by the
reference's profile digest and spec, the decode settings and the bank, and
reuses them for later candidates: a second candidate against the same reference
sends no reference generation request. A grade is reused only if it was made by
the same grader (the same judge model and template for judge-graded items).
Loading is strict: a damaged entry refuses the run. A missing outputs file is
treated as a miss and rebuilt (`reference_cache.notes` in the report says so).
Two runs writing the same entry at once both leave a valid entry; the last one
wins. The cache works with the default and `harmful_excess` criteria.

## Tests

```bash
python -m pip install pytest
python -m pytest
```

Installing pytest needs network access. The tests themselves run offline against
the local stand-in, in one to two minutes of CPU time.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md). Please report security problems
privately, as described in [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
