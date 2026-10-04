# REFERENCES

Identifiers were checked against arXiv / ACL Anthology / official pages. Specific numerical claims used in this
project are listed with the source they come from.

1. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762.
2. Wang, B. & Komatsuzaki, A. *GPT-J-6B* (parallel attention + feed-forward block). 2021.
   https://github.com/kingoflolz/mesh-transformer-jax
3. Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways.* arXiv:2204.02311 (JMLR 2023).
   Parallel layers: roughly 15% faster training at large scale; small degradation at 8B, none at 62B.
   FLOP convention (6N + 12·L·H·Q·T) from its Appendix B.
4. Press, Smith & Levy. *Improving Transformer Models by Reordering their Sublayers* (Sandwich Transformer).
   ACL 2020. arXiv:1911.03864.
5. Lu et al. *Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View*
   (Macaron Net). arXiv:1906.02762.  Gulati et al. *Conformer.* Interspeech 2020. arXiv:2005.08100.
6. Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017.
   arXiv:1701.06538.
7. Lepikhin et al. *GShard.* arXiv:2006.16668.
8. Fedus, Zoph & Shazeer. *Switch Transformers.* JMLR 2022. arXiv:2101.03961.
   Load-balancing loss alpha·N·sum(f_i·P_i) with alpha = 1e-2 ("sufficiently large to ensure load balancing while
   small enough to not overwhelm the primary cross-entropy objective"); selective FP32 router precision.
9. Zoph et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* arXiv:2202.08906. Router z-loss
   (coefficient 1e-3 in the paper; logged but not used in the objective here).
10. Jiang et al. *Mixtral of Experts.* arXiv:2401.04088. 8 experts, top-2 routing, SwiGLU experts.
11. Dai et al. *DeepSeekMoE.* arXiv:2401.06066.  Muennighoff et al. *OLMoE.* arXiv:2409.02060.
12. Gale et al. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys 2023. arXiv:2211.15841.
    (Dropless MoE motivation; not installed here.)
13. Dao et al. *FlashAttention.* NeurIPS 2022. arXiv:2205.14135.  Dao. *FlashAttention-2.* arXiv:2307.08691.
14. Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024
    Datasets & Benchmarks. arXiv:2406.17557. Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
15. Hoffmann et al. *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556.
    Kaplan et al. *Scaling Laws for Neural Language Models.* arXiv:2001.08361 (C ~ 6N per token).
16. Shazeer. *GLU Variants Improve Transformer* (SwiGLU). arXiv:2002.05202.
17. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
18. Zhang & Sennrich. *Root Mean Square Layer Normalization.* NeurIPS 2019. arXiv:1910.07467.
19. Loshchilov & Hutter. *Decoupled Weight Decay Regularization* (AdamW). ICLR 2019. arXiv:1711.05101.
20. Radford et al. *Language Models are Unsupervised Multitask Learners* (GPT-2; residual-projection init scaled by
    1/sqrt(N_residual)). 2019.
21. PyTorch documentation: CUDA semantics - streams and backward passes; `torch.profiler`;
    `torch.nn.attention.sdpa_kernel`. https://pytorch.org/docs/stable/notes/cuda.html
