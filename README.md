# Mechanistic Interpretability Replications

Reimplementations of three mechanistic-interpretability papers using [NNsight](https://nnsight.net) and the [National Deep Inference Fabric (NDIF)](https://ndif.us) for remote execution on Llama-2-7B. Each replication reproduces the core results of its source paper and runs end-to-end in Google Colab against remote GPUs — no local hardware required.

| Replication | Source | Key result reproduced |
|---|---|---|
| [Induction Circuits (ARENA 1.2)](replications/arena_1_2) | [ARENA 1.2: Intro to Mech Interp](https://arena3-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp) | Previous-token head **L0H7** → induction head **L1H4**; baseline induction score **0.560** |
| [Do Llamas Work in English?](replications/do_llamas_work_in_english) | [Wendler et al. 2024 (arXiv:2402.10588)](https://arxiv.org/abs/2402.10588) | English acts as a latent "pivot" language — English token probability peaks in middle layers before the target language |
| [Safety Head Attribution](replications/safety-head-attribution) | [Zhou et al. 2024 (arXiv:2410.13708)](https://arxiv.org/abs/2410.13708) | Ablating a single safety-critical attention head (~0.006% of parameters) substantially raises the harmful-response rate |

## What's implemented

**Induction Circuits (ARENA 1.2)** — Full reimplementation of the ARENA mech-interp curriculum in NNsight: caching activations, attention-pattern visualisation, induction-head detection on repeated-token sequences, direct logit attribution, zero/mean ablation, and reverse-engineering the OV copying and QK previous-token circuits (composition scores, targeted K-composition ablation).

**Do Llamas Work in English?** — Logit-lens analysis across all 32 layers of Llama-2-7B on French→Chinese translation and French cloze tasks. Tracks English vs. target-language token probability, distribution entropy, and an energy metric (alignment of hidden states with the unembedding subspace), with 95% confidence intervals.

**Safety Head Attribution** — Three attribution/ablation methods:
- **SHIPS** — query-level safety-head scoring via KL divergence of the output distribution when each head's query projection is masked.
- **Sahara** — dataset-level attribution using the principal angle between hidden-state subspaces (SVD) before and after masking.
- **Surgery** — single-head ablation followed by a harmful-response evaluation over a jailbreak dataset (baseline vs. ablated).

## Tech stack

- **[NNsight](https://nnsight.net)** — activation caching and intervention via `model.trace()` / `model.generate()` contexts.
- **[NDIF](https://ndif.us)** — remote execution of Llama-2-7B on shared GPUs.
- PyTorch, Transformers, NumPy/Pandas, Matplotlib/Seaborn, circuitsvis.

## Running

Each replication is a self-contained notebook. To run one:

1. Get an NDIF API key at https://login.ndif.us and a HuggingFace token (for Llama-2 access) at https://huggingface.co/settings/tokens.
2. Open the notebook in Google Colab or Jupyter.
3. Paste your keys into the setup cell.
4. Run all cells top to bottom — model execution happens remotely on NDIF.

See each replication's own README for per-experiment details, runtimes, and result tables.

## Repository layout

```
replications/
├── arena_1_2/                    # Induction circuits (ARENA 1.2)
├── do_llamas_work_in_english/    # Latent-language logit-lens study
└── safety-head-attribution/      # SHIPS / Sahara / ablation
```
