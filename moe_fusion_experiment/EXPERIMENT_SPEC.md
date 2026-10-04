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

## 7. Phases

0 environment, 1 data verification, 2 unit tests, 3 budget, 4 memory probe, 5 smoke, 6 systems benchmark,
7 profiling, 8 training (pilot), 9 report. MAIN, 3-seed replication and the fusion-interval ablation (2 / 8 /
final-only, widths from `flop_counter.matched_hidden`) are NOT run automatically.
