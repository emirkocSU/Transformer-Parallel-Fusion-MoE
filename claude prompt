You are acting as a senior ML systems researcher, LLM architecture researcher, PyTorch/CUDA engineer, and reproducibility engineer.

Your task is to DESIGN, IMPLEMENT, VERIFY, PACKAGE, AND PREPARE a rigorous A/B/C architecture experiment that can be run on a SINGLE NVIDIA A100 GPU in Google Colab.

This is not a toy notebook and not a conceptual demo.

The resulting project must be scientifically interpretable, reproducible, runnable, tested, and resistant to accidental confounders.

Do not fabricate benchmark results.

Do not claim GPU parallelism unless it is actually measured.

Do not silently change experimental conditions between architectures.

Do not optimize one architecture differently from the others unless the optimization is explicitly part of a separate systems experiment.

The central hypothesis is:

Can we remove the strict Attention → MoE dependency from most Transformer layers, run Attention and MoE as parallel branches, and recover the lost interaction using periodic Fusion MoE layers, while obtaining a better quality-versus-wall-clock tradeoff?

==================================================
0. THE THREE CORE ARCHITECTURES
==================================================

We must implement three core architectures.

A — SERIAL BASELINE

Traditional pre-norm Transformer ordering:

x
→ RMSNorm
→ causal self-attention
→ residual addition
→ RMSNorm
→ MoE FFN
→ residual addition

Mathematically:

a_l = x_l + Attention(RMSNorm_attn(x_l))

x_(l+1) = a_l + MoE(RMSNorm_moe(a_l))

The critical property is:

The MoE sees the attention-updated representation from the SAME layer.

--------------------------------------------------

B — PURE PARALLEL CONTROL

Attention and MoE receive the same original residual stream representation.

Attention branch:

A_l = Attention(RMSNorm_attn(x_l))

MoE branch:

M_l = MoE(RMSNorm_moe(x_l))

Merge:

x_(l+1) = x_l + A_l + M_l

The MoE does NOT see the same-layer attention result.

This architecture is essential as the control.

Without B, we cannot determine whether architecture C succeeds because of periodic fusion or merely because parallel layers themselves work.

--------------------------------------------------

C — PARALLEL + PERIODIC FUSION

Most layers use the same parallel block as B:

x_(l+1) = x_l
          + Attention(RMSNorm_attn(x_l))
          + MoE(RMSNorm_moe(x_l))

After every N parallel blocks, insert a Fusion MoE:

x_fused = x + FusionMoE(RMSNorm_fusion(x))

Initial default:

fusion_interval = 4

For a 12-layer backbone:

Embedding
↓
P1
↓
P2
↓
P3
↓
P4
↓
FusionMoE
↓
P5
↓
P6
↓
P7
↓
P8
↓
FusionMoE
↓
P9
↓
P10
↓
P11
↓
P12
↓
FusionMoE
↓
Final RMSNorm
↓
LM Head

The FusionMoE must operate on the representation AFTER attention and MoE information have been merged through the preceding parallel blocks.

The initial C model therefore contains:

12 parallel attention/MoE blocks
+
3 FusionMoE modules.

Do NOT accidentally interpret a FusionMoE as another attention layer.

It is an extra token-wise MoE transformation.

==================================================
1. TWO DISTINCT QUESTIONS MUST BE TESTED
==================================================

Do NOT collapse everything into a single comparison.

We need two related but distinct experiments.

-----------------------------------
EXPERIMENT 1 — MECHANISM TEST
-----------------------------------

Question:

Does periodic fusion recover quality lost by pure parallelization?

Use:

A_serial_samewidth
B_parallel_samewidth
C_parallel_fusion_samewidth

All ordinary MoE modules use the same expert width.

The C model therefore has additional parameters and additional MoE FLOPs because it contains FusionMoE modules.

That is INTENTIONAL for this experiment.

Interpretation:

B versus A:
effect of removing same-layer Attention → MoE dependency.

C versus B:
effect of adding periodic fusion.

Do NOT use C versus A from this test alone to claim superior compute efficiency because C has additional compute.

-----------------------------------
EXPERIMENT 2 — COMPUTE/PARAMETER MATCHED TEST
-----------------------------------

Question:

Does periodic fusion still produce a useful quality/speed tradeoff when the total MoE parameter budget and active MoE compute are approximately matched?

Use:

A_serial_matched
B_parallel_matched
C_parallel_fusion_matched

A and B contain:

12 MoE applications.

C contains:

12 regular MoE applications
+
3 FusionMoE applications
=
15 MoE applications.

Therefore choose the C expert hidden dimension so that:

15 × C_expert_width
≈
12 × baseline_expert_width

Set the baseline expert hidden dimension to:

1920

Therefore:

C matched expert hidden dimension:

1536

because:

12 × 1920
=
15 × 1536
=
23040

This gives an unusually clean compute/parameter match.

Use:

A expert_hidden = 1920
B expert_hidden = 1920
C matched expert_hidden = 1536

C same-width mechanism experiment uses:

C_samewidth expert_hidden = 1920

Write a script that independently calculates and verifies:

total parameters
trainable parameters
attention parameters
router parameters
expert parameters
embedding parameters
LM head parameters
active parameters per token
estimated forward linear-layer FLOPs per token
estimated attention FLOPs at sequence length 1024
estimated total training FLOPs

The script must print a comparison table.

For the matched comparison, fail with a clear warning if the intended expert parameter/FLOP mismatch is unexpectedly above 2%.

Do not rely only on my algebra.

Verify it programmatically.

==================================================
2. INITIAL MODEL CONFIGURATION
==================================================

Use a decoder-only causal language model.

Initial shared configuration:

number of backbone layers: 12
d_model: 768
attention heads: 12
head dimension: 64
context length: 1024
number of experts: 8
experts selected per token: top-2
normalization: RMSNorm
expert activation: SwiGLU
position encoding: RoPE
precision: BF16 on A100
dropout: 0 unless a strong technical reason requires otherwise
biases: avoid unless required
causal LM objective: next-token prediction

Use tied embedding/output weights if doing so does not create implementation complications.

Whatever choice is made must be IDENTICAL between A, B and C.

Do not use different attention types.

Do not use MLA in one model and MHA in another.

Do not use GQA only in one architecture.

Use ordinary causal multi-head self-attention for this experiment because we are trying to isolate layer ordering rather than attention architecture.

==================================================
3. ATTENTION IMPLEMENTATION
==================================================

Prefer PyTorch's native scaled_dot_product_attention / optimized SDPA backend when it actually selects an efficient A100-compatible implementation.

The code must detect and record which backend is being used.

Do not blindly install FlashAttention if native PyTorch SDPA provides the required fused/flash implementation.

If native SDPA is unavailable or inappropriate, support FlashAttention-2 as an optional backend.

Whichever attention backend is selected:

A
B
C

must use the SAME implementation.

Provide:

attention_backend = "sdpa"
attention_backend = "flash_attn"

through configuration if practical.

At runtime print:

GPU name
GPU compute capability
CUDA version
PyTorch version
BF16 support
attention backend
SDPA/FlashAttention availability

If the environment silently falls back to an inefficient attention implementation, make that visible in the logs.

==================================================
4. MoE IMPLEMENTATION
==================================================

Implement a correct top-2 token-routing Mixture-of-Experts FFN.

Each expert is a SwiGLU FFN.

Conceptually:

router_logits = W_router x

router_probs = softmax(router_logits)

top2_experts, top2_weights = TopK(router_probs, k=2)

Normalize selected routing weights appropriately.

Each token must be processed by exactly two selected experts.

Weighted expert outputs are combined into the token output.

DO NOT DROP TOKENS in the reference implementation.

Token dropping would add another variable to the experiment.

Implement load balancing diagnostics.

Track at minimum:

tokens assigned to each expert
fraction of assignments per expert
mean router probability per expert
router entropy
max/min utilization ratio
coefficient of variation of expert utilization
load-balancing auxiliary loss

The training objective must be:

L_total
=
L_language_model
+
lambda_balance × L_balance

But validation perplexity MUST be calculated from the LANGUAGE MODEL LOSS ONLY.

Do not include router auxiliary loss when calculating perplexity.

Choose a reasonable load-balancing coefficient based on established MoE practice, document it, and use exactly the same coefficient for A/B/C.

Do not invent a coefficient without documenting the reasoning.

Optional router z-loss may be included only if:

1. it is clearly documented,
2. it is identical across all models,
3. LM loss and auxiliary losses are reported separately.

-----------------------------------
MoE backends
-----------------------------------

Provide at least:

1. reference PyTorch implementation
2. optimized implementation when safely available

The reference implementation prioritizes correctness.

The optimized implementation may use a suitable grouped-GEMM/block-sparse MoE approach if its installation is reliable on Colab A100.

MegaBlocks may be investigated.

Do NOT make the whole project dependent on a fragile optional package.

If an optimized package fails to install:

fall back to the correct PyTorch implementation,
record the fallback,
continue the experiment,
and do not fabricate performance claims.

All A/B/C architectures must use the same MoE backend during any single comparison.

==================================================
5. TRUE PARALLEL EXECUTION
==================================================

THIS SECTION IS CRITICAL.

Architecture B and C are mathematically parallelizable.

That does NOT mean the code is actually running both branches concurrently.

This:

attn = attention(x)
moe = moe(x)

is sequential dispatch and must NOT be called "real GPU parallel execution."

We need TWO implementations of the parallel block:

-----------------------------------
parallel_reference
-----------------------------------

Calculate the two branches using ordinary execution.

Purpose:

mathematical correctness.

-----------------------------------
parallel_concurrent
-----------------------------------

Attempt true CUDA concurrency using separate CUDA streams or another technically justified mechanism.

Conceptually:

stream_attention
stream_moe

launch attention branch on stream_attention

launch MoE branch on stream_moe

synchronize correctly

merge:

x + attention_output + moe_output

Be extremely careful about:

autograd
stream ownership
record_stream where required
CUDA events
synchronization
race conditions
tensor lifetime
backward correctness

DO NOT ASSUME that two CUDA streams automatically give a speedup.

The attention and MoE kernels may both saturate the A100.

Actual kernel overlap must be MEASURED.

-----------------------------------
Correctness requirement
-----------------------------------

With identical weights and inputs:

parallel_reference

and

parallel_concurrent

must produce numerically equivalent forward outputs within an appropriate BF16/FP32 tolerance.

Also compare gradients.

For a small deterministic FP32 test where possible:

compare parameter gradients.

The concurrent implementation is not accepted unless:

outputs are correct
gradients are correct
no synchronization errors occur.

-----------------------------------
Performance verification
-----------------------------------

Measure:

attention branch time
MoE branch time
combined elapsed time
training step time

Use CUDA events.

Also produce a PyTorch profiler trace with CPU and CUDA activities enabled.

Add meaningful ranges such as:

parallel_attention_branch
parallel_moe_branch
parallel_merge
fusion_moe
serial_attention
serial_moe

Export traces.

Where possible determine:

whether attention/MoE CUDA kernels overlap,
how much overlap occurs,
whether concurrency survives backward,
whether only forward overlaps,
whether no useful overlap occurs.

If:

T_parallel ≈ T_attention + T_moe

then report:

"No meaningful GPU overlap was achieved."

Do NOT call it parallel speedup.

If:

T_parallel < T_attention + T_moe

quantify the measured overlap/speedup.

Architecture-level parallelism and hardware-level speedup must be reported as separate concepts.

==================================================
6. DATASET
==================================================

Use:

HuggingFaceFW/fineweb-edu

Prefer the official:

sample-10BT

subset as the source corpus.

DO NOT train directly from a live streaming internet dataset during timed training runs.

Network variability would corrupt throughput measurements.

Instead build a deterministic local tokenized dataset BEFORE benchmarked training starts.

The final timed experiment must read from local storage.

-----------------------------------
Dataset preparation
-----------------------------------

Create a reproducible script such as:

scripts/prepare_fineweb.py

It should:

1. download/read a deterministic portion of FineWeb-Edu sample-10BT
2. preserve source ordering deterministically
3. create a fixed train/validation split
4. tokenize exactly once
5. pack tokens into contiguous sequences
6. store them efficiently locally
7. generate metadata and checksums

Use a fixed document-level split before token packing.

Avoid validation leakage caused by slicing a single continuous token stream after concatenating train and validation documents.

Store a manifest containing:

dataset name
configuration
dataset revision if available
source file names
source hashes when practical
number of source documents
number of train documents
number of validation documents
tokenizer identifier/hash
number of train tokens
number of validation tokens
sequence length
creation timestamp
script version/git commit

-----------------------------------
Token budgets
-----------------------------------

Support several experiment sizes:

SMOKE:
~1–2 million training tokens

PILOT:
~20–30 million training tokens

MAIN:
100 million training tokens

OPTIONAL_EXTENDED:
300 million training tokens

Do not download billions of tokens unnecessarily for the initial experiment.

The main scientific comparison should default to 100M tokens per model.

Validation should be large enough to provide a stable loss estimate.

Target approximately 5M validation tokens if practical.

The validation corpus must be identical for A/B/C.

==================================================
7. TOKENIZER
==================================================

Use ONE tokenizer for every model.

Preferred target vocabulary:

approximately 32,000 tokens.

For maximum reproducibility, choose one of two strategies:

OPTION A:
train one ByteLevel BPE tokenizer with vocab_size=32000 from a fixed deterministic subset of the training documents.

OPTION B:
use a well-established open tokenizer if training our own tokenizer creates unnecessary complexity.

If choosing B instead of A, explain the decision.

My preference for this project is OPTION A if it can be made deterministic and straightforward.

The tokenizer must be trained ONCE.

Save:

tokenizer files
tokenizer config
special tokens
vocabulary hash

All architectures must load the exact same tokenizer artifact.

Do not retrain a tokenizer separately for each model.

==================================================
8. TRAINING CONFIGURATION
==================================================

Use the same optimizer/training recipe unless changing it is absolutely necessary.

Suggested starting point:

optimizer: AdamW
betas: (0.9, 0.95)
weight decay: 0.1
gradient clipping: 1.0
precision: BF16
learning rate: approximately 3e-4
scheduler: cosine decay
warmup: 2% of training steps

If you believe one of these values should be changed, document the reason BEFORE observing A/B/C results.

Do not tune hyperparameters separately for A/B/C.

That would contaminate the architecture comparison.

-----------------------------------
Batching
-----------------------------------

Use:

sequence length = 1024

Define a fixed GLOBAL token batch size shared by all architectures.

Because A100 VRAM may be 40GB or 80GB, automatically detect memory.

Perform a short memory probe.

Determine the largest microbatch that ALL compared models can run safely.

Then use the COMMON microbatch size across A/B/C.

Use gradient accumulation to reach the same global token batch size.

Do not let model A use batch size 16 while C uses 8 during the primary training comparison unless absolutely unavoidable.

If model memory requirements differ:

use the largest microbatch safe for ALL architectures.

For a separate throughput-only benchmark, optionally report:

common-batch throughput

and

maximum-safe-batch throughput

as two different metrics.

Do not mix them.

==================================================
9. RANDOMNESS AND INITIALIZATION
==================================================

Use paired experimental seeds.

Suggested seeds:

42
43
44

For seed 42:

A/B/C receive matched random conditions as much as structurally possible.

Same for 43 and 44.

Use:

same data order
same dropout state if dropout exists
same initialization seed
same batch order

Where modules have identical shapes and semantics, initialize/copy identical weights between architectures before training when scientifically appropriate.

For example:

embeddings
attention projections
matching experts
LM head

This reduces initialization noise.

Do not force nonsensical weight copying between differently shaped C_matched expert layers.

Record all seed values.

For the first main run, one seed is acceptable to identify gross differences.

Before making a strong conclusion, support a 3-seed confirmation run.

==================================================
10. CHECKPOINTING AND COLAB RESILIENCE
==================================================

Google Colab runtimes can disconnect.

Implement robust resumable checkpoints.

Checkpoint must contain:

model state
optimizer state
scheduler state
GradScaler state if applicable
current step
tokens processed
epoch/data position or deterministic sampler state
Python RNG state
NumPy RNG state
PyTorch CPU RNG
CUDA RNG state
experiment config
git commit/hash if available

Resuming should reproduce the data stream rather than restarting the dataset.

Support optional Google Drive checkpoint destination.

Suggested:

/content/drive/MyDrive/moe_fusion_experiment/

However:

DO NOT stream training batches directly from Google Drive.

For performance:

copy prepared dataset shards to /content local disk before timed training.

Save checkpoints/results back to Drive.

==================================================
11. REQUIRED EXPERIMENT PHASES
==================================================

Implement the workflow in explicit phases.

-----------------------------------
PHASE 0 — ENVIRONMENT VALIDATION
-----------------------------------

Print:

GPU model
VRAM
CUDA
driver
PyTorch
Python
BF16 availability
attention backend
MoE backend
free disk space
RAM

Run nvidia-smi.

Fail early on unsupported configuration.

-----------------------------------
PHASE 1 — UNIT TESTS
-----------------------------------

All tests must pass before training.

-----------------------------------
PHASE 2 — SMOKE TRAIN
-----------------------------------

Run each architecture for a tiny dataset.

Verify:

loss decreases
no NaNs
no exploding router imbalance
checkpoints save/load
evaluation works
profiling works

-----------------------------------
PHASE 3 — PILOT
-----------------------------------

Approximately 20–30M tokens.

Run:

A serial
B parallel
C parallel+fusion

single paired seed.

Goal:

identify catastrophic architecture bugs before expensive training.

-----------------------------------
PHASE 4 — MAIN
-----------------------------------

100M tokens per architecture.

Run primary comparison.

-----------------------------------
PHASE 5 — REPLICATION
-----------------------------------

If budget permits:

3 seeds.

Do not claim statistical robustness from one seed.

-----------------------------------
PHASE 6 — FUSION FREQUENCY ABLATION
-----------------------------------

Only after the A/B/C core experiment works.

Test:

C2: fusion every 2 blocks
C4: fusion every 4 blocks
C8: fusion every 8 blocks
C_final: only final fusion

Keep comparison fair.

Do not run this expensive ablation automatically unless explicitly enabled.

Default:

fusion_interval = 4.

==================================================
12. REQUIRED UNIT TESTS
==================================================

Create a proper tests/ directory.

Use pytest.

At minimum implement:

-----------------------------------
test_model_shapes.py
-----------------------------------

Check logits shape.

Check layer output dimensions.

-----------------------------------
test_causality.py
-----------------------------------

This test is mandatory.

Take the same prefix.

Change future tokens.

Verify logits at earlier positions do NOT change.

This detects accidental future-token leakage.

-----------------------------------
test_lm_shift.py
-----------------------------------

Verify that next-token labels/logits are shifted correctly.

-----------------------------------
test_moe_top2.py
-----------------------------------

Verify:

each token selects exactly two experts
routing weights are finite
weights combine correctly
no token disappears
all outputs have correct shape

-----------------------------------
test_moe_balance_metrics.py
-----------------------------------

Check utilization statistics.

-----------------------------------
test_no_token_drop.py
-----------------------------------

Ensure the reference MoE does not silently discard overflow tokens.

-----------------------------------
test_parallel_equivalence.py
-----------------------------------

Set identical weights.

Compare:

parallel_reference

versus

parallel_concurrent.

Forward outputs must match within tolerance.

-----------------------------------
test_parallel_gradients.py
-----------------------------------

Compare gradients of reference and concurrent parallel implementation.

-----------------------------------
test_checkpoint_resume.py
-----------------------------------

Run several steps.

Checkpoint.

Resume.

Verify continued training state matches an uninterrupted reference within appropriate tolerance.

-----------------------------------
test_dataset_determinism.py
-----------------------------------

Same seed + same config must return same training token batches.

-----------------------------------
test_train_val_separation.py
-----------------------------------

Verify source-document IDs do not overlap between training and validation.

-----------------------------------
test_parameter_budget.py
-----------------------------------

Verify parameter/FLOP accounting.

-----------------------------------
test_loss_finite.py
-----------------------------------

One forward/backward training step for every architecture.

No NaN or Inf.

==================================================
13. METRICS TO LOG
==================================================

Every training run must log machine-readable metrics.

Use CSV and/or JSONL.

Do not require Weights & Biases.

Optional W&B integration may be offered, disabled by default.

Log at least:

step
tokens_processed
wall_clock_seconds
LM train loss
total loss
load balance loss
router z-loss if used
learning rate
gradient norm
tokens/sec
step time
GPU peak allocated memory
GPU peak reserved memory

Router metrics:

expert_0_fraction
...
expert_7_fraction
router entropy
expert utilization CV
max/min utilization

Validation events:

validation LM loss
perplexity
validation tokens
validation wall time

Important:

perplexity = exp(validation LM loss)

Do not use total loss containing auxiliary router terms.

==================================================
14. PERFORMANCE BENCHMARKING
==================================================

Performance benchmarks must be separate from normal training measurements.

Warm up GPU first.

Synchronize CUDA correctly.

Use:

torch.cuda.Event

for GPU timing.

Do not use naive:

time.time()

without CUDA synchronization.

For benchmark reporting use:

median
p50
p95
mean
standard deviation

after warmup iterations.

Benchmark:

forward only
forward + backward
complete optimizer step

Measure:

tokens/sec
sequences/sec
ms/step
peak VRAM

For parallel B/C additionally measure:

attention-only branch
MoE-only branch
combined branch
fusion layer

Run enough repeated iterations to reduce noise.

==================================================
15. PROFILING
==================================================

Create:

scripts/profile_model.py

Use torch.profiler with CPU and CUDA activities.

Export Chrome/Perfetto compatible JSON traces.

Profile a small number of representative steps after warmup.

Use named ranges.

Produce traces such as:

profiles/A_serial.json
profiles/B_parallel_reference.json
profiles/B_parallel_concurrent.json
profiles/C_parallel_fusion.json

If possible, write a small parser that reports:

CUDA stream IDs
kernel intervals
whether kernels from attention and MoE overlap
estimated overlap duration
estimated overlap fraction

If automated overlap parsing is unreliable:

say so explicitly,
preserve the trace,
and provide exact instructions for manually inspecting the trace.

Never invent an overlap percentage.

==================================================
16. PRIMARY EVALUATION METRICS
==================================================

The most important comparison is NOT simply final perplexity.

Produce at minimum:

1.
Validation loss vs training tokens

This measures statistical/sample efficiency.

2.
Validation loss vs wall-clock time

THIS IS THE PRIMARY QUALITY/SPEED METRIC.

3.
Perplexity vs training tokens

4.
Perplexity vs wall-clock

5.
Training tokens/sec

6.
Median training step time

7.
Peak VRAM

8.
Total parameters

9.
Active parameters/token

10.
Estimated FLOPs/token

11.
Expert utilization metrics

12.
Measured attention/MoE concurrency evidence

==================================================
17. PRE-REGISTER SUCCESS CRITERIA
==================================================

Define criteria BEFORE looking at results.

Primary hypothesis:

C can reach approximately the same validation LM loss as A in less wall-clock training time.

Secondary hypothesis:

C has better wall-clock quality tradeoff than B because periodic fusion restores some representational capacity.

Important interpretation patterns:

CASE 1

B worse than A
C approximately recovers A
C faster than A

Strong evidence supporting the fusion hypothesis.

-----------------------------------

CASE 2

B approximately equals C

Fusion is probably unnecessary at this scale.

-----------------------------------

CASE 3

A materially outperforms both B and C

Same-layer Attention → MoE dependency appears important.

-----------------------------------

CASE 4

C outperforms A per token but not per wall-clock

Fusion may improve model quality while adding too much systems cost.

-----------------------------------

CASE 5

C outperforms A per wall-clock under compute-matched conditions

Very interesting result.

Requires replication across seeds before strong claims.

-----------------------------------

CASE 6

Parallel CUDA streams show no meaningful kernel overlap

Architecture may be mathematically parallelizable, but this implementation/hardware does not obtain systems-level parallel speedup.

Report exactly that.

Do not reinterpret it as a successful performance result.

==================================================
18. STATISTICAL REPORTING
==================================================

For three-seed runs calculate:

mean
standard deviation

for final:

validation loss
perplexity
tokens/sec
wall-clock to selected loss thresholds

If the learning curves cross, avoid reducing everything to final loss.

Also calculate time-to-target.

For example:

time to validation loss = X

Choose target values only after ensuring all models actually reach the threshold.

Do not choose a target specifically because it favors one model.

==================================================
19. PLOTS
==================================================

Automatically generate publication-quality plots.

Required:

loss_vs_tokens.png
loss_vs_wallclock.png
ppl_vs_tokens.png
ppl_vs_wallclock.png
throughput_comparison.png
step_time_comparison.png
vram_comparison.png
expert_utilization.png

For fusion ablations:

fusion_interval_vs_loss.png
fusion_interval_vs_throughput.png

Do not manually hardcode results.

Read data from experiment logs.

==================================================
20. REPOSITORY STRUCTURE
==================================================

Create a clean project similar to:

moe_fusion_experiment/
│
├── README.md
├── EXPERIMENT_SPEC.md
├── METHODOLOGY.md
├── REFERENCES.md
├── requirements.txt
├── requirements-lock.txt or equivalent
├── pyproject.toml
│
├── configs/
│   ├── A_serial.yaml
│   ├── B_parallel.yaml
│   ├── C_parallel_fusion_samewidth.yaml
│   ├── C_parallel_fusion_matched.yaml
│   ├── smoke.yaml
│   ├── pilot.yaml
│   └── main.yaml
│
├── src/
│   ├── config.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── attention.py
│   ├── moe.py
│   ├── blocks.py
│   ├── model.py
│   ├── data.py
│   ├── tokenizer.py
│   ├── losses.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── checkpoint.py
│   ├── metrics.py
│   ├── flop_counter.py
│   ├── benchmarking.py
│   └── utils.py
│
├── scripts/
│   ├── prepare_tokenizer.py
│   ├── prepare_fineweb.py
│   ├── verify_dataset.py
│   ├── inspect_models.py
│   ├── run_smoke.py
│   ├── run_experiment.py
│   ├── benchmark_models.py
│   ├── profile_model.py
│   ├── aggregate_results.py
│   └── make_plots.py
│
├── tests/
│   ├── test_model_shapes.py
│   ├── test_causality.py
│   ├── test_lm_shift.py
│   ├── test_moe_top2.py
│   ├── test_no_token_drop.py
│   ├── test_parallel_equivalence.py
│   ├── test_parallel_gradients.py
│   ├── test_checkpoint_resume.py
│   ├── test_dataset_determinism.py
│   ├── test_train_val_separation.py
│   ├── test_parameter_budget.py
│   └── test_loss_finite.py
│
├── colab/
│   └── A100_MoE_Fusion_Experiment.ipynb
│
├── results/
├── plots/
├── profiles/
└── checkpoints/

If a simpler structure is clearly better, explain why before changing it.

==================================================
21. GOOGLE COLAB NOTEBOOK
==================================================

The notebook is an important deliverable.

Create one notebook:

colab/A100_MoE_Fusion_Experiment.ipynb

It should allow me to execute the entire experiment from a fresh Google Colab A100 runtime.

I should not have to manually reconstruct commands from README text.

Organize it into clear cells:

CELL 1
Environment detection

CELL 2
Install only required dependencies

CELL 3
Optional Google Drive mount

CELL 4
Clone/load project

CELL 5
Run unit tests

CELL 6
Prepare/download tokenizer

CELL 7
Prepare FineWeb-Edu fixed dataset

CELL 8
Verify dataset

CELL 9
Inspect parameter/FLOP budget

CELL 10
Memory probe

CELL 11
Run smoke test

CELL 12
Benchmark A/B/C

CELL 13
Profile parallel execution

CELL 14
Run pilot experiment

CELL 15
Run main A

CELL 16
Run main B

CELL 17
Run main C

CELL 18
Evaluate checkpoints

CELL 19
Aggregate results

CELL 20
Generate plots

CELL 21
Produce final report

Add a top-level configuration:

RUN_MODE = "smoke"

with allowed values:

smoke
pilot
main
full

So I cannot accidentally burn substantial GPU credits.

Do not automatically launch a 300M-token experiment merely by opening/running the notebook.

==================================================
22. DEPENDENCY MANAGEMENT
==================================================

Google Colab already contains many CUDA/PyTorch packages.

Do NOT blindly upgrade PyTorch/CUDA.

First inspect environment compatibility.

Pin package versions that are actually used.

Generate an environment manifest.

Save:

pip freeze
PyTorch version
CUDA version
driver
GPU
Python version

If installing FlashAttention:

validate compatibility BEFORE compilation.

If installation fails:

do not break the notebook.

Use PyTorch SDPA fallback.

If installing MegaBlocks or another optimized MoE package:

make it optional.

The correctness reference implementation must always remain available.

==================================================
23. DATA I/O MUST NOT CORRUPT TIMINGS
==================================================

Timed training must not depend on downloading data.

Prepare local binary/memmap/token shards first.

Use DataLoader configuration appropriate for Colab.

Measure dataloader wait time.

If CPU/data loading becomes a bottleneck, identify it.

Do not attribute data pipeline stalls to the model architecture.

During architecture throughput benchmark, optionally benchmark with pre-generated synthetic token tensors already resident/available locally so pure model compute can be measured separately from end-to-end training throughput.

Therefore report TWO performance categories:

MODEL COMPUTE THROUGHPUT

and

END-TO-END TRAINING THROUGHPUT.

==================================================
24. MEMORY AND OOM HANDLING
==================================================

Never silently modify model architecture after an OOM.

Architecture must remain fixed.

Allowed adaptations:

microbatch size
gradient accumulation

Not allowed:

reducing d_model only for C
reducing layers only for C
reducing context only for C

unless intentionally running a separately labelled experiment.

After OOM:

clear gradients
empty cache if necessary
reinitialize cleanly
record what happened

Then choose a common microbatch safe for all models.

==================================================
25. MODEL INSPECTION OUTPUT
==================================================

Before training, print a human-readable table containing:

Model
Backbone layers
Attention layers
Ordinary MoE layers
Fusion MoE layers
d_model
heads
experts
top-k
expert hidden dimension
total params
expert params
active params/token
estimated FLOPs/token
expected BF16 parameter memory

This table must make accidental mismatches obvious.

==================================================
26. CORRECTNESS BEFORE SPEED
==================================================

Never optimize code before a clear reference version passes tests.

Development sequence:

1.
reference serial

2.
reference parallel

3.
reference fusion

4.
unit tests

5.
tiny training sanity check

6.
optimized attention

7.
optimized MoE

8.
concurrent CUDA implementation

9.
equivalence tests again

10.
profiling

11.
real experiment

If an optimization changes outputs incorrectly, revert it.

==================================================
27. COMPILATION
==================================================

torch.compile may be tested.

But:

Do not enable it for one architecture and disable it for another in the primary benchmark.

First benchmark all architectures without compilation.

Then optionally benchmark:

A compiled
B compiled
C compiled

as a separate systems comparison.

Record graph breaks.

If CUDA streams or MoE routing interact badly with torch.compile, prioritize correctness and clear reporting.

==================================================
28. OPTIONAL FUSED PARALLEL PROJECTION EXPERIMENT
==================================================

Do NOT make this part of the mandatory first experiment.

After the core experiment is correct, investigate whether linear projections feeding attention and MoE can be fused or grouped to reduce launch/memory overhead.

This is inspired by parallel-layer implementation ideas, but it is a separate systems optimization.

Never let this obscure the architecture experiment.

==================================================
29. FUSION ABLATION
==================================================

Only after the main A/B/C comparison succeeds.

Support configuration:

fusion_interval = 2
fusion_interval = 4
fusion_interval = 8
fusion_interval = "final_only"

For fair compute matching, automatically calculate the appropriate C expert width given the number of MoE applications.

General formula:

baseline_moe_count × baseline_width
≈
C_total_moe_count × C_width

Round only to hardware-friendly dimensions.

Report the exact resulting mismatch.

Do not pretend rounded configurations are exactly matched.

==================================================
30. FINAL REPORT
==================================================

Automatically generate:

FINAL_REPORT.md

It must contain:

A. Experiment question

B. Architecture definitions

C. Dataset

D. Tokenizer

E. Training setup

F. Hardware/software environment

G. Parameter budget

H. FLOP budget

I. Validation results

J. Throughput results

K. Wall-clock comparison

L. VRAM comparison

M. Router/expert behavior

N. CUDA concurrency/profiling evidence

O. Failures or anomalies

P. Statistical uncertainty

Q. Interpretation

R. Limitations

S. Recommended next experiment

The report must clearly separate:

MEASURED RESULT

from

INTERPRETATION

from

SPECULATION.

Never write:

"Fusion works"

unless the measured experiment supports that conclusion.

Instead use precise language such as:

"Under the tested 12-layer, d_model=768 configuration and 100M-token training budget..."

==================================================
31. REFERENCES
==================================================

Create REFERENCES.md.

Verify every citation.

Do not hallucinate references.

Include relevant primary sources for:

Transformer
parallel Transformer/parallel attention-MLP precedent
PaLM
GPT-J if used as architectural precedent
Sandwich Transformer
Switch Transformer
Mixtral
MoE load balancing
MegaBlocks
FlashAttention / FlashAttention-2
FineWeb / FineWeb-Edu
Chinchilla/scaling considerations if referenced

Prefer:

official papers
official repositories
official documentation.

If unsure whether a claim exists in a paper:

verify it before including it.

==================================================
32. EXPERIMENT MANIFEST
==================================================

For every completed run create:

experiment_manifest.json

with:

architecture
config
seed
dataset hash
tokenizer hash
code git commit
GPU
VRAM
PyTorch
CUDA
attention backend
MoE backend
parallel execution backend
parameter counts
FLOP estimates
batch sizes
optimizer
scheduler
learning rate
number of tokens
start/end time
checkpoint paths

This should make a run independently traceable.

==================================================
33. FAILURE CONDITIONS
==================================================

The system should detect and clearly report:

NaN loss
Inf loss
NaN gradients
router collapse
extreme expert imbalance
all tokens routed to same experts
OOM
dataset corruption
validation leakage
checkpoint mismatch
missing CUDA concurrency
unexpected attention backend fallback
invalid parameter budget
incorrect causal masking

Do not continue a long expensive training run after a clear catastrophic failure.

Save diagnostic information first.

==================================================
34. ROUTER COLLAPSE GUARD
==================================================

Create an alert if expert usage becomes severely imbalanced.

Do not automatically change the router architecture during the comparison.

Record the event.

If training becomes unusable:

stop the run and report why.

Any later router fix must be applied identically to A/B/C and the experiments restarted.

==================================================
35. SCIENTIFIC FAIRNESS RULES
==================================================

These rules are non-negotiable.

Do not:

train C longer
give C more dataset
use different tokenizer
use different optimizer
use different LR schedule
use different context length
use different attention implementation
use different precision
give only one architecture torch.compile
give only one architecture FlashAttention
give only one architecture optimized MoE
select different validation sets
exclude bad seeds selectively

unless the comparison is explicitly labelled as a separate ablation.

==================================================
36. PRIMARY QUESTIONS THE FINAL RESULTS MUST ANSWER
==================================================

At the end I want explicit answers to:

Q1.
How much quality is lost when going from:

A Serial

to

B Pure Parallel?

Q2.
How much of that loss is recovered by:

C Parallel + Fusion?

Q3.
Does C still help when expert parameter/compute budget is matched?

Q4.
Does the mathematically parallel structure produce actual A100 wall-clock speedup?

Q5.
Do attention and MoE kernels actually overlap?

Q6.
What fraction of theoretical parallelism becomes real hardware acceleration?

Q7.
What is the time-to-target validation loss for A/B/C?

Q8.
Which model has the best:

validation loss per token?

Q9.
Which model has the best:

validation loss per wall-clock second?

Q10.
Is fusion_interval=4 promising enough to justify 2/8/final-only ablations?

==================================================
37. DO NOT OVERCLAIM
==================================================

A 100M-token, ~100–200M-active-parameter experiment cannot establish that an architecture will necessarily outperform frontier billion/trillion-parameter models.

The final report must explicitly distinguish:

small-scale architectural evidence

from

large-scale LLM claims.

If C succeeds:

say it justifies scaling the experiment.

Do not claim a new state-of-the-art Transformer architecture.

==================================================
38. DELIVERABLES
==================================================

Before declaring the task complete, I expect:

1.
Complete repository.

2.
Working Colab A100 notebook.

3.
requirements/environment files.

4.
FineWeb-Edu preparation pipeline.

5.
Tokenizer preparation pipeline.

6.
A implementation.

7.
B implementation.

8.
C implementation.

9.
same-width C config.

10.
compute-matched C config.

11.
unit tests.

12.
smoke-test command.

13.
pilot command.

14.
main experiment command.

15.
benchmark code.

16.
CUDA profiling code.

17.
checkpoint/resume implementation.

18.
metrics logger.

19.
plot generator.

20.
parameter/FLOP comparison script.

21.
experiment manifest generator.

22.
FINAL_REPORT template/generator.

23.
verified references.

24.
README with exact execution instructions.

==================================================
39. ACCEPTANCE GATE
==================================================

DO NOT say the project is ready until all of these are true:

[ ] repository imports successfully

[ ] pytest passes

[ ] causal leakage test passes

[ ] top-2 routing test passes

[ ] no-token-drop test passes

[ ] serial smoke training reduces loss

[ ] parallel smoke training reduces loss

[ ] fusion smoke training reduces loss

[ ] checkpoints restore

[ ] dataset is deterministic

[ ] validation split does not overlap training documents

[ ] parameter accounting is printed

[ ] compute-matching is verified

[ ] profiler trace is produced

[ ] reference vs concurrent parallel outputs match

[ ] reference vs concurrent gradients match

[ ] A/B/C can all execute on the detected GPU

[ ] main notebook cells are ordered and runnable

If you cannot verify an item in your current environment:

mark it:

UNVERIFIED — REQUIRES A100 COLAB

rather than pretending it passed.

==================================================
40. HOW TO WORK
==================================================

Do not merely explain how I could build this.

BUILD IT.

Create the files.

Write the code.

Run all tests available in your current environment.

Fix failures.

Run the smallest smoke training available.

Inspect the logs.

Do not stop at pseudocode.

Do not give me fragments requiring me to assemble the repository myself.

If your environment lacks an A100:

complete everything possible locally,
mark only genuine A100-dependent validation as pending,
and make the Colab notebook execute those checks automatically when I open it on my A100 runtime.

Do not ask me to choose trivial implementation details.

Use engineering judgment.

Only ask me a question if proceeding is genuinely impossible without my answer.

Otherwise make the safest scientifically defensible choice and document it.

==================================================
41. FINAL RESPONSE TO ME
==================================================

When finished, give me a concise completion report containing:

1.
Repository tree.

2.
What was implemented.

3.
Tests run and exact pass/fail status.

4.
What was actually executed in your environment.

5.
What remains A100-only.

6.
Exact first command/notebook cell I should run in Colab.

7.
Expected outputs from the smoke stage.

8.
Where checkpoints/results/plots will appear.

9.
Any known limitation.

10.
Do NOT provide experimental conclusions unless the corresponding training run actually completed.

The goal is not to make the idea look good.

The goal is to determine whether it is actually good.
