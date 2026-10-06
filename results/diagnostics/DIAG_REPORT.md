# DIAGNOSTIC REPORT - why did B (parallel) beat A (serial)?

_Pre-registered rules: EXPERIMENT_SPEC.md Amendment 3. Pilot budget (25M tokens) per run._

## Measured B - A (paired, full validation set; negative = B better)

| group | B - A (nats) | 95% CI |
|---|---|---|
| seed42 | -0.1719 | [-0.1725, -0.1714] |
| dense | -0.1357 | [-0.1363, -0.1351] |
| warmup10 | -0.1971 | [-0.1977, -0.1964] |
| seed43 | -0.1189 | [-0.1196, -0.1183] |

## Routing stability (top-1 expert churn of a fixed validation probe between evaluations)

* A: first quarter 0.6482, whole run 0.2828
* B: first quarter 0.5245, whole run 0.2336
* A_s43: first quarter 0.6218, whole run 0.2716
* B_s43: first quarter 0.5204, whole run 0.2327
* dead-expert alerts (seed 42): A at steps [8], B at steps [16]

## Rules

* **R1_seed**: ROBUST across seeds  `{'seed42': -0.1719, 'seed43': -0.1189}`
* **R2_dense**: NOT MoE-specific (dense shows at least half of the MoE gap)  `{'moe': -0.1719, 'dense': -0.1357}`
* **R3_warmup**: NOT explained by warm-up  `{'warmup2pct': -0.1719, 'warmup10pct': -0.1971}`
* **R4_routing**: weak / inconclusive  `{'A_early_churn': 0.6482, 'B_early_churn': 0.5245, 'ratio': 1.2359}`

## Overall reading

**Robust but NOT MoE-specific: the parallel advantage also appears with a dense FFN, i.e. it is a property of this training regime (small, heavily under-trained, one shared recipe) rather than of MoE routing.**

Single diagnostic runs per condition: these rules decide what to test next; they are not a replacement for replication.
