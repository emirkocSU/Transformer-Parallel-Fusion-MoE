# EXPERIMENT SPECIFICATION AND PRE-REGISTRATION

Written and committed **before any A/B/C training run**. Analysis code (`src/moefusion/analysis.py`,
`scripts/make_report.py`) implements exactly these rules. Any later change must be recorded as a deviation.

## 1. Hypotheses

* **H1 (primary):** C (parallel + periodic FusionMoE) reaches approximately the same validation LM loss as A
  (serial) in less training wall-clock time.
* **H2 (secondary):** C has a better wall-clock quality trade-off than B (pure parallel) because periodic fusion
  restores some of the representational interaction lost by parallelisation.

## 2. Architectures (all share: 12 layers, d_model 768, 12x64 heads, RoPE theta 1e4, RMSNorm eps 1e-6,
8 SwiGLU experts top-2 with renormalised weights and NO token dropping, tied embeddings, vocab 32,000, context 1024,
no dropout, no biases, causal MHA through ONE attention kernel)

| name | block | MoE applications | expert hidden | total params | active params/token | train FLOPs/token* |
|---|---|---|---|---|---|---|
| A_serial | `a=x+Attn(N(x)); y=a+MoE(N(a))` | 12 | 1920 | 477.65M | 159.15M | 1.068 G |
| B_parallel | `y=x+Attn(N_a(x))+MoE(N_m(x))` | 12 | 1920 | 477.65M | 159.15M | 1.068 G |
| C_parallel_fusion_samewidth | B + `x+FusionMoE(N_f(x))` after blocks 4, 8, 12 | 15 | 1920 | 583.84M | 185.71M | 1.227 G |
| C_parallel_fusion_matched | same as above | 15 | 1536 | 477.67M | 159.17M | 1.068 G |

\* 6N + attention-score convention (PaLM App. B); verified programmatically by `scripts/inspect_models.py`.
Compute matching: 12 x 1920 = 15 x 1536 = 23,040 -> expert parameters and active MoE FLOPs differ by **0.000%**.

Two experiments are read from the same four runs:
* **Experiment 1 (mechanism):** A vs B vs C_same. C_same intentionally has +22% expert parameters and
  +15% training FLOPs; C_same vs A must NOT be used as an efficiency claim.
* **Experiment 2 (compute/parameter matched):** A vs B vs C_matched.

## 3. Fixed conditions (identical for every architecture)

| item | value |
|---|---|
| data | FineWeb-Edu sample-10BT @ revision 87f09149..., first 300k docs, doc-level split (285k / 15k) |
| tokenizer | one ByteLevel BPE artifact, vocab 32,000, sha256 2a6d7dca... |
| sequence | 1024 inputs + 1024 next-token targets per row (rows of 1025 tokens) |
| global batch | 64 sequences = 65,536 tokens / optimizer step |
| micro-batch | largest size that fits ALL four models (memory probe), identical for all; grad accumulation fills the rest |
| steps | smoke 24, **pilot 384 (25,165,824 tokens)**, main 1526 (100,007,936 tokens) |
| data order | epoch permutation seeded by the paired seed; step k always gets the same 64 rows |
| optimizer | AdamW beta (0.9, 0.95), eps 1e-8, wd 0.1 on >=2-D tensors, fused CUDA kernel |
| LR | peak 3e-4, linear warm-up 2% of steps, cosine to 3e-5 |
| clipping | global norm 1.0 |
| precision | BF16 autocast; FP32 master weights, router, norms, softmax/cross-entropy; TF32 disabled |
| aux loss | Switch balance loss, coefficient 0.01 **per MoE application (router), summed** -- see Amendment 1 |
| z-loss | logged, coefficient 0 |
| init | per-parameter generator seeded by (seed, name, shape): A and B bit-identical, C shares every common tensor |
| attention | one backend for all runs, chosen once in Phase 0 (FlashAttention-2 via SDPA when available) |
| MoE backend | reference (sorted dispatch, per-expert GEMMs) for all runs |
| parallel execution in training | reference (sequential dispatch) for all runs; concurrency is a separate systems measurement |
| evaluation | step 0, every 32 steps (pilot) on a fixed 1,024-sequence subset; final on the FULL validation set |
| wall-clock axis | sum of synchronised optimizer-step times; evaluation, checkpointing and warm-up excluded |
| run order | A, B, C_same, C_matched, one fresh process each, same session/GPU |
| seed | 42 (pilot/main); replication 42, 43, 44 |

## 4. Pre-registered analysis rules

* **Primary quality metric:** final validation LM loss (auxiliary terms excluded) on the full validation set;
  perplexity = exp(loss).
* **Paired comparison:** per-sequence loss differences on identical validation sequences; mean, normal 95% CI and
  bootstrap 95% CI (2,000 resamples, seed 0).
* **Equivalence margin:** |dL| < **0.02 nats** with the CI inside (-0.02, 0.02) -> "approximately equal".
  CI excluding 0 -> "better"/"worse" (flagged if below the margin). Otherwise "inconclusive".
* **Speed margin:** step-time differences below **2%** are "no speed difference".
* **Time-to-target thresholds:** worst final periodic-subset loss among the compared models + {0.00, 0.05, 0.10, 0.25}
  nats (every model reaches every threshold; no threshold is chosen after seeing which model it favours).
* **Loss at common wall-clock:** periodic-curve loss interpolated at the training time at which the fastest model
  finished.
* **Recovery fraction:** (L_B - L_C)/(L_B - L_A), reported only if B is significantly worse than A.
* **Q10 rule:** the fusion-interval ablation is justified if dL(C_same - B) has a 95% CI entirely below 0.

## 5. Interpretation patterns

| case | condition | reading |
|---|---|---|
| 1 | B worse than A; C ~ A; C faster than A | strong support for the fusion hypothesis |
| 2 | B ~ C | fusion probably unnecessary at this scale |
| 3 | A materially better than B and C | same-layer Attention->MoE dependency matters |
| 4 | C better than A per token, not per wall-clock | quality gain eaten by systems cost |
| 5 | C better than A per wall-clock, compute-matched | very interesting; needs 3 seeds |
| 6 | no measured kernel overlap | mathematically parallel, no hardware speed-up in this implementation |

## 6. Failure handling

NaN/Inf loss or gradient -> run stopped, `failure.json`, pipeline continues with the other architectures and the
failure is reported. Router collapse (<= top_k experts with >= 1% of assignments in a layer for 3 consecutive
logging intervals after warm-up) -> run stopped and reported; any router fix must be applied to all architectures and
all runs restarted. OOM -> only micro-batch/accumulation may change (for all models), never the architecture.

## 6a. Amendment 1 (before any pilot comparison run)

*Original rule:* balance loss averaged over MoE applications (0.01 x mean over 12 or 15 routers).
*Problem found in the A100 smoke stage* (24 steps, no comparison data yet): per-router balance loss stayed at
1.17-1.52 (1.0 = balanced), utilisation CV rose to 1.28 and A_serial raised a router-collapse warning. Averaging gives
each router only 0.01/12 (A, B) or 0.01/15 (C) of balancing pressure: 12-15x weaker than the cited reference and
**not equal across architectures**. Switch Transformer adds the auxiliary loss for EACH switch layer.
*New rule:* 0.01 x L_balance for every router, summed (`aux_loss_reduction: sum`). Every router of every
architecture now receives identical pressure. Evidence before adoption (tiny CPU model, same data/seed, stressed
LR 3e-3, 40 steps): mean -> 6 collapse warnings and one persistent collapse (C_same stopped), per-router balance
1.05-1.15; sum -> 0 warnings, 1.01-1.02, unchanged LM loss. Applied identically to A, B and C; no pilot comparison had
been run. The smoke router criterion was aligned with the training guard (a transient early warning is recorded;
a persistent collapse or a collapsed layer in the final interval fails the smoke stage).

## 6b. Amendment 2 -- MAIN protocol (fixed after the pilot, before any MAIN run)

*Pilot findings that shaped it (seed 42, 25M tokens):* B beat A by 0.179 nats; C_same - B = +0.005 (inside the margin);
C_matched - B = +0.059; no attention/MoE kernel overlap on the A100; A had a starved-expert episode (steps 16-44) that the
collapse guard did not flag.

* **Runs:** A_serial, B_parallel, C_parallel_fusion_samewidth, C_parallel_fusion_matched; 1526 steps x 65,536 tokens
  = 100,007,936 tokens each (97,664 of 287,264 training rows: no row is seen twice); **seed 42 only** (owner's
  decision; seed variance is therefore NOT measured). All other conditions of section 3 unchanged.
* **Order:** B, C_same, C_matched, A (fusion-relevant runs first if the Colab session is lost); fresh process per run.
* **Systems benchmark/profiling:** not repeated (token-independent; pilot values stand).
* **Checkpoints:** local, every 400 steps, one file at a time; completed runs survive a lost session via Drive.
* **Monitoring added (never stops a run):** dead/starved-expert alert when an expert receives < 0.0125 (= 0.1 x uniform)
  of its layer's assignments in a logging interval.
* **Decision rule** (`analysis.multi_seed_summary`): for X - Y with paired per-sequence dL and 95% CI,
  threshold = max(0.02 nats, observed seed spread; with one seed: 0.02). *better/worse*: CI excludes 0 and |dL| >=
  threshold; *approximately equal*: |dL| < 0.02; otherwise *inconclusive*.
  **Fusion verdict:** USEFUL if C_matched beats B (same budget); ADDS QUALITY AT EXTRA COST if only C_same beats B
  (then judged per wall-clock: C_same vs B at the common training time); NOT USEFUL if C_same is approximately equal
  to or worse than B; otherwise INCONCLUSIVE. Single-seed results are reported as not replicated.

## 6c. Amendment 3 -- DIAGNOSTIC phase (written before any diagnostic run; to be run BEFORE MAIN)

*Question:* is the pilot's B > A gap (0.179 nats) a real effect (e.g. the parallel block keeps the MoE router more stable
early in training) or an artefact of seed, recipe or regime? All groups use the pilot budget (384 steps x 65,536 tokens),
micro-batch 16 (= pilot), the same data and code; each group is analysed on its own (paired B - A on the full
validation set).

| group | models | change vs pilot |
|---|---|---|
| diag_seed42 | A_serial, B_parallel | none (re-baseline with the current code) |
| diag_dense | A_dense, B_dense | dense SwiGLU FFN, hidden 3840 = 2 x 1920 (identical active FFN FLOPs; = 1-expert top-1 MoE, constant router) |
| diag_warmup10 | A_serial, B_parallel | warm-up 10% (38 steps) instead of 2% (8 steps) |
| diag_seed43 | A_serial, B_parallel | seed 43 (init + data order) |

D4: every MoE run records the top-1 expert of a fixed 8,192-token validation probe at each evaluation; churn = fraction
of probe tokens whose top-1 expert changed since the previous evaluation (StableMoE routing fluctuation).

*Rules* (d = paired B - A; margin 0.02 nats):
* **R1 seed:** ROBUST if d(seed 43) has the sign of d(seed 42) and both |d| >= 0.02.
* **R2 dense:** MoE-SPECIFIC if the dense gap has the opposite sign or |d_dense| < 0.5 |d_moe|; otherwise not MoE-specific.
* **R3 warm-up:** LARGELY EXPLAINED by early training if |d_warmup10| <= 0.5 |d_seed42|.
* **R4 routing:** SUPPORTS routing fluctuation if A's mean churn over the first quarter >= 1.5 x B's; does not if <= 1.1x.
* **Overall:** not robust -> gap not established (seed noise or < 0.02 nats); robust and R3 explained -> warm-up artefact; robust, MoE-specific and R4
  supports -> real MoE routing-stability effect; robust and MoE-specific without R4 -> real but mechanism unconfirmed;
  robust and not MoE-specific -> property of the regime/recipe, not of MoE routing.
These rules decide what to test next; single runs per condition are not replication.

## 7. Phases

0 environment, 1 data verification, 2 unit tests, 3 budget, 4 memory probe, 5 smoke, 6 systems benchmark,
7 profiling, 8 training (pilot), 9 report. MAIN, 3-seed replication and the fusion-interval ablation (2 / 8 /
final-only, widths from `flop_counter.matched_hidden`) are NOT run automatically.
