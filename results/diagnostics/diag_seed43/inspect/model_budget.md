| Model | Layers | Attn | MoE | Fusion | d | heads | E | k | hidden | total | experts | active/tok | FLOPs/tok(train) | bf16 GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_serial | 12 | 12 | 12 | 0 | 768 | 12 | 8 | 2 | 1920 | 477.654M | 424.673M | 159.149M | 1.068B | 0.96 |
| B_parallel | 12 | 12 | 12 | 0 | 768 | 12 | 8 | 2 | 1920 | 477.654M | 424.673M | 159.149M | 1.068B | 0.96 |
