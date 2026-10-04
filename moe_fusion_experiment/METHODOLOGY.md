# METHODOLOGY - how confounders are controlled

## Same foundation for A, B and C
* **One code path.** `SerialBlock` and `ParallelBlock` own the same sub-modules with the same names; only `forward`
  differs. `FusionBlock` reuses the same `RMSNorm` and `MoE` classes. A and B therefore have *identical* parameter
  sets; this is asserted by `tests/test_model_shapes.py::test_identical_init_across_architectures`.
* **Same initial weights.** Every tensor is drawn from its own generator keyed by (seed, parameter name, shape).
  The embedding, all 12 attention layers, all 12 routers and all 12 backbone expert banks are bit-identical across
  A, B and C_same. C_matched shares everything except the (differently shaped) expert banks. No weights are copied
  between differently shaped tensors.
* **Same number of loops.** Every model runs exactly 12 Transformer layers (12 attention, 12 backbone MoE). C adds
  3 FusionMoE applications *between* layers; the backbone loop count is unchanged. Every model performs the same
  number of optimizer steps on the same token batches in the same order (`assert_fair` + automated fairness audit).
* **Same compute where it matters.** C_matched shrinks the expert width so that 15 x 1536 = 12 x 1920: total expert
  parameters and active MoE FLOPs are identical to A/B (0.000% difference, checked programmatically; the run stops
  if the mismatch exceeds 2%). Time differences in Experiment 2 can therefore not come from "doing more MoE work".
* **Same auxiliary-loss weight.** The balance loss is averaged (not summed) over MoE applications, so the 15-MoE
  models do not receive 25% more router regularisation.

## Same environment
* All runs execute sequentially in ONE Colab session, on the same GPU, each in a fresh Python process (clean CUDA
  context and allocator), with the same software, the same attention kernel (pinned via `sdpa_kernel`), the same
  numeric settings (TF32 off, BF16 autocast) and the same micro-batch.
* Data is read from local disk into RAM before training; batch fetch time is logged (`data_wait`) so data stalls
  cannot be attributed to an architecture.
* The pipeline records GPU, driver, CUDA, PyTorch, pip freeze, code fingerprint and dataset hashes in every run
  manifest; the fairness audit fails if any of them differs between runs.

## Time measurement
* Training wall-clock = sum of per-step times, each step bracketed by `torch.cuda.synchronize()`. Every model
  performs the same per-step bookkeeping (one host transfer of the scalars), so logging overhead is identical.
* Evaluation, checkpoint writes and an untimed warm-up (forward/backward without optimizer step, gradients
  discarded) are excluded from the wall-clock axis and reported separately.
* The systems benchmark is separate: CUDA events, warm-up, repeated iterations, **interleaved and rotated rounds**
  across models to cancel GPU clock/thermal drift, median/p95/mean/std reported.

## Parallel execution is measured, not assumed
* Primary training uses sequential dispatch for every architecture (identical kernels; no architecture receives a
  systems optimisation the others lack).
* `parallel_backend: concurrent` runs the attention branch and the MoE branch on two CUDA streams with explicit
  stream waits and `record_stream` lifetime protection. It is accepted only if outputs and gradients match the
  reference (unit tests on the GPU + a full-size check in the benchmark).
* Overlap is quantified twice: branch timings (T_attn, T_moe, T_reference, T_concurrent) and kernel intervals parsed
  from `torch.profiler` traces. If T_concurrent ~ T_reference the report states "No meaningful GPU overlap was
  achieved".

## Validation
* Document-level split before packing (no stream slicing across splits); leakage is checked by source id and by
  exact text hash; a re-tokenisation spot check proves the `.npy` arrays were produced from the `.jsonl` documents
  with the shipped tokenizer.
* Validation perplexity uses the LM loss only. The periodic curve uses a fixed evenly spaced subset; the final
  number uses all 15,088 validation sequences, and per-sequence losses are stored for paired statistics.

## What a single pilot can and cannot say
One seed => the CIs reflect evaluation-data noise only. The pilot is a bug detector and a first look; confirmatory
statements require MAIN plus the 3-seed replication. Hyperparameters are shared, not tuned per architecture.
