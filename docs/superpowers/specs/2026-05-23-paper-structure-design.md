# Paper Structure Design

**Target venue:** Journal (TMLR or similar)
**Core contribution:** Comparative analysis of text-augmented fusion mechanisms under data scarcity + practitioner use-case guide
**Novelty:** Evaluation protocol + failure mode taxonomy + text source ablation

---

## Structure

| Section | Role |
|---------|------|
| Introduction | Problem, gap, aim, contributions |
| Background | Time series forecasting, text-augmented forecasting, data-scarcity problem |
| Related Work | Full survey — text+TS fusion, data-efficient forecasting, multimodal learning |
| Experimental Protocol | Datasets, fractions, seeds, metrics, text encoding pipeline |
| Fusion Models | Full subsection per model (7 architectures) |
| Text Source Ablation | Template vs LLM vs random |
| Data-Scarcity Sweep & Failure Mode Taxonomy | Degradation curves, α-collapse, gate suppression, FiLM fragility |
| Practitioner Guide | Decision table: data budget × horizon → fusion choice |
| Conclusion | Summary, limitations, future work |
| Appendix | Dataset details, equations, extra ablations, reproducibility checklist |

---

## Section 1 — Introduction

Text-augmented time series forecasting is a growing research area: language descriptions encode domain knowledge — regime labels, seasonal context, metadata — that raw numerical data cannot. Recent fusion architectures pair time series encoders with text representations to improve forecast accuracy. Many real domains such as energy, finance, and healthcare have rich textual metadata alongside limited labelled history, making this a practically important problem.

The gap in existing work is that fusion models are evaluated almost exclusively at full training data. No systematic study has examined how these models behave as training data shrinks — the realistic scenario for new deployments, rare events, or domain shifts. It is unknown whether fusion with text confers robustness under data scarcity, or whether the choice of fusion mechanism determines that robustness.

This paper aims to investigate whether text-augmented fusion models degrade more gracefully than strong TS-only baselines as training data shrinks, and whether the fusion mechanism determines the degree of that robustness.

Our contributions are three-fold. First, a controlled evaluation protocol spanning 5 training fractions, 2 horizons, 5 datasets, 7 architectures, and 3 seeds. Second, a failure mode taxonomy — α-collapse, gate suppression, and FiLM fragility — characterising why apparent robustness does not imply text utilisation. Third, a text source ablation showing that template descriptions match LLM-generated text at zero inference cost, refuting the assumption that richer text is always better.

---

## Section 2 — Background

**Time series forecasting.** Given a sequence of observed values $\mathbf{x}_{1:T} \in \mathbb{R}^{T \times C}$, the forecasting task is to predict future values $\mathbf{x}_{T+1:T+H}$ over a horizon $H$. Early deep learning approaches relied on recurrent architectures; Transformer-based models [1] subsequently became dominant by capturing long-range dependencies through self-attention. PatchTST [2] improved upon this by segmenting the input into non-overlapping patches, preserving local temporal structure and reducing attention complexity. Despite this, simple linear models such as DLinear [3] remain competitive baselines, particularly in low-complexity settings, making them a useful stress test for more expressive architectures.

**Text-augmented forecasting.** A growing line of work augments numerical time series with natural language inputs to inject domain knowledge that raw values cannot encode. Time-LLM [4] reprograms a frozen LLM to act as a forecasting backbone. LLM4TS [5] aligns pre-trained language models with time series structure through a two-stage fine-tuning pipeline. Time-VLM [6] extends this to vision-language models, treating time series as a third modality alongside text and image. In all these approaches, text is assumed to add signal; the mechanism by which it is fused into the forecasting backbone is treated as a secondary concern.

**Data scarcity in forecasting.** Real-world deployments rarely have access to full historical training sets. New sensor deployments, rare regime shifts, and domain transfers all present forecasting problems with limited data. LLM4TS [5] motivates the use of pre-trained models partly on data-efficiency grounds, and recent zero-shot and few-shot approaches [7, 8] confirm that data scarcity is an active open problem. What remains unexamined is how the choice of text fusion mechanism interacts with training data volume — the question this paper addresses.

*(Figures to be added.)*

---

## Section 3 — Related Work

**Text-augmented and LLM-based forecasting.** The dominant paradigm treats a pre-trained language model as the forecasting backbone and maps time series into the token space of that model. GPT4TS [12] freezes a GPT-2 backbone and fine-tunes only the layer norms, demonstrating that language model representations transfer to time series. Time-LLM [4] goes further by reprogramming the input layer so that patch embeddings are projected into word-token space before being processed by a frozen LLM. LLM4TS [5] introduces a two-stage alignment pipeline and reports competitive results under limited data. More recent work targets the fusion side directly: Zhou et al. [13] propose a dual-tower architecture for high-dimensional datasets; Zhao et al. [14] use multi-level text alignment to anchor LLM representations to temporal structure; Su et al. [15] treat text as a reinforcement signal rather than a conditioning input; and MGTS-Net [16] incorporates graph structure alongside text for multivariate settings. Time-VLM [6], the grounding paper for this work, extends the paradigm to vision-language models by encoding time series as images alongside text. Across all of these, text quality and fusion mechanism are treated as engineering choices rather than research variables. Jin et al. [18] survey this landscape comprehensively. Our work departs by holding the text source and architecture constant across a controlled data-scarcity protocol, isolating the fusion mechanism as the variable of interest.

**Data-efficient and low-resource forecasting.** A parallel line of work asks whether forecasting models can generalise under limited supervision. LLM4TS [5] motivates language model transfer partly on data-efficiency grounds, showing improved few-shot performance over train-from-scratch baselines. Nochumsohn et al. [7] propose a frequency-domain framework for zero-shot forecasting, bypassing the training data requirement entirely. Gopali et al. [8] demonstrate in-context and few-shot forecasting using LLMs without gradient updates. These approaches address data scarcity by reducing dependence on training data altogether. Our framing is different: we ask how models that do require training data degrade as that data shrinks, and whether text fusion can slow that degradation.

**Fusion mechanisms in multimodal learning.** The mechanisms used to combine modalities in text-augmented forecasting are largely inherited from vision-language research. Feature-wise Linear Modulation (FiLM) [11] conditions one stream on another via learned affine transformations, originally for visual question answering. Cross-attention [1] allows one modality to attend to representations of another, and has become a standard building block in multimodal architectures. Gated fusion and ensemble weighting offer softer alternatives that can suppress a modality when it carries no signal. Despite the prevalence of these mechanisms, their behaviour under training data scarcity has not been studied. We show that the same mechanism can appear robust while actually suppressing the text signal entirely — a distinction invisible without diagnostic instrumentation.

*(Figures to be added.)*

---

## References

| # | Reference |
|---|-----------|
| [1] | Vaswani et al. (2017) — Attention Is All You Need — arXiv:1706.03762 |
| [2] | Nie et al. (2023) — PatchTST — arXiv:2211.14730 |
| [3] | Zeng et al. (2022) — DLinear — arXiv:2205.13504 |
| [4] | Jin et al. (2023) — Time-LLM — arXiv:2310.01728 |
| [5] | Chang et al. (2023) — LLM4TS — arXiv:2308.08469 |
| [6] | Zhong et al. (2025) — Time-VLM — arXiv:2502.04395 |
| [7] | Nochumsohn et al. (2024) — Beyond Data Scarcity — arXiv:2411.15743 |
| [8] | Gopali et al. (2025) — In-Context and Few-Shots Learning for TS — arXiv:2512.07705 |
| [9] | Zhou et al. (2021) — Informer *(ETT datasets)* — arXiv:2012.07436 |
| [10] | Wu et al. (2021) — Autoformer *(Weather dataset)* — arXiv:2106.13008 |
| [11] | Perez et al. (2018) — FiLM — arXiv:1709.07871 |
| [12] | Zhou et al. (2023) — GPT4TS / One Fits All — arXiv:2302.11939 |
| [13] | Zhou, Wang et al. (2025) — Unveiling the Potential of Text in High-Dimensional TS Forecasting — arXiv:2501.07048 |
| [14] | Zhao, Chen, Sun (2025) — Enhancing TS via Multi-Level Text Alignment with LLMs — arXiv:2504.07360 |
| [15] | Su et al. (2025) — Text Reinforcement for Multimodal TS Forecasting — arXiv:2509.00687 |
| [16] | Hao, Bao, Li (2025) — MGTS-Net — arXiv:2510.16350 |
| [17] | Wang et al. (2020) — MiniLM — arXiv:2002.10957 |
| [18] | Jin et al. (2024) — Large Language Models for Time Series: A Survey — arXiv:2402.01801 |
