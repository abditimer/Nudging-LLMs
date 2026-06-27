# Literature Review: Novelty and Gap Assessment

## Project Framing

Based on [README.md](/Users/abditimer/Documents/PhD/experiments/nudging/README.md) and the current code in [nudging/experiment.py](/Users/abditimer/Documents/PhD/experiments/nudging/nudging/experiment.py), this project is best described as:

> A prefix-completion probe for testing whether LLMs can reproduce creative texts that were released after their documented training cutoff dates.

The strongest version of the claim is not "first study of LLM memorization" or "first post-cutoff study". Those are too broad. The more defensible novelty claim is:

> To our knowledge, this is the first empirical study of verbatim and near-verbatim continuation on temporally verified post-cutoff creative texts across multiple creative-text domains and model families using graded prefix reveal.

Important wording changes:

- Use "cross-domain creative-text" instead of "cross-modal". Songs, books, and podcast transcripts are all text in the current setup.
- Use "documented post-cutoff content" instead of "guaranteed post-training content" unless model-specific training/update histories are fully auditable.
- Treat release date as strong evidence against contamination, not absolute proof.

## Literature Matrix

| Paper | Venue / Year | Core task | Data type | Uses post-cutoff timing? | Uses prefix/suffix continuation? | Closest overlap with this project | Why it matters |
|---|---|---|---|---|---|---|---|
| Carlini et al., *Extracting Training Data from Large Language Models* | USENIX Security 2021 | Training-data extraction attack | Web text, PII, code | No | Yes | Foundational memorization/extraction paper | Establishes that LMs can emit verbatim training examples under prompting. |
| Kandpal et al., *Deduplicating Training Data Mitigates Privacy Risks in Language Models* | ICML 2022 | Measure extractable memorization vs duplication | Web text | No | Yes | Mechanism paper | Shows duplication strongly drives memorization risk, which matters for songs/books likely repeated online. |
| Mireshghallah et al., *An Empirical Analysis of Memorization in Fine-tuned Autoregressive Language Models* | EMNLP 2022 | Membership and extraction after fine-tuning | NLP task datasets | No | Partly | Tangential — de-prioritise | Fine-tuning memorization is tangential; targets here are pretrained models. Cite briefly in related work but don't build on it. |
| Chang et al., *Speak, Memory: An Archaeology of Books Known to ChatGPT/GPT-4* | EMNLP 2023 | Membership-style probing for memorized books | Copyrighted books | No | No | Book-side adjacent prior art | Strong evidence that copyrighted books are memorized, but not a post-cutoff continuation design. |
| Karamolegkou et al., *Copyright Violations and Large Language Models* | EMNLP 2023 | Redistribution of copyrighted text | Popular books, coding problems | No | Yes | Closest prior art on copyrighted text continuation | Very relevant comparator; your work differs if you center post-cutoff creative texts and temporal verification. |
| Ippolito et al., *Preventing Generation of Verbatim Memorization in Language Models Gives a False Sense of Privacy* | INLG 2023 | Test limits of verbatim-only defenses | Memorized training text | No | Yes | Methodological warning | Important because your exact-match metric alone is too narrow; near-match behavior also matters. |
| Yin et al., *ALCUNA: Large Language Models Meet New Knowledge* | EMNLP 2023 | Benchmark new knowledge handling | Synthetic novel entities | Yes, indirectly | No | Adjacent "new knowledge" benchmark — de-prioritise | QA over synthetic facts rather than real creative text. Useful contrast ("new knowledge" thread exists) but thin overlap. Drop if page space is tight. |
| Roberts et al., *To the Cutoff... and Beyond? A Longitudinal Perspective on LLM Data Contamination* | ICLR 2024 | Measure contamination using release time | Code and math benchmarks | Yes | No | Main threat to any broad temporal novelty claim | They already use release date/cutoff timing as a contamination probe, but not for creative texts or continuation. |
| Vu et al., *FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation* | Findings of ACL 2024 | Evaluate current-knowledge QA | Dynamic factual QA | Yes | No | Adjacent freshness benchmark — de-prioritise | Useful contrast point ("we study verbatim recall, not factual currency") but low overlap. Drop if page space is tight. |
| Duarte et al., *DE-COP: Detecting Copyrighted Content in Language Models Training Data* | ICML 2024 | Detect copyrighted training inclusion | Book excerpts before and after cutoff | Yes | No | Strongly adjacent on post-cutoff books | Important because it already uses pre/post-cutoff books, but as a detection benchmark rather than free continuation. |
| Huang et al., *Demystifying Verbatim Memorization in Large Language Models* | EMNLP 2024 | Controlled memorization analysis | Injected sequences in Pythia | No | Yes | Mechanism paper — cite but don't lean on | Uses synthetic injected sequences in Pythia, not real-world creative content. Good for establishing mechanism but keep at arm's length in the narrative. |
| Nasr et al., *Scalable Extraction of Training Data from Aligned, Production Language Models* | ICLR 2025 | Extract memorized data from aligned models | Open and closed LLMs | No | Yes | Production-model memorization prior art | Shows alignment does not remove memorization risk in closed models. |
| Shi et al., *Detecting Pretraining Data from Large Language Models* | ICLR 2024 | Likelihood-based membership inference (MIN-K%) | Pre/post-cutoff web text and benchmarks | Yes | No | Strong adjacent prior art — must address | Uses release timing explicitly to test pre vs post-cutoff data. Temporal framing is very close to this project. Novelty claim must distinguish: detection via likelihood vs. continuation under graded prefix reveal. |
| Meeus et al., *Did the Neurons Read this Book Too? Risks of Reconstruction and Exfiltration of Pre-Training Data* | USENIX Security 2024 | Membership inference on books, pre vs post-cutoff | Copyrighted books, pre/post-cutoff releases | Yes | No | Strong adjacent prior art — must address | Explicitly tests pre vs post-cutoff books for memorization. Very close to your temporal novelty claim. Distinguish: detection task vs. continuation task; books vs. songs/podcasts. |
| Hayes et al., *Measuring memorization in language models via probabilistic extraction* | NAACL 2025 | Probabilistic extraction metric | Language-model training examples | No | Yes | Closest methodological challenge to current setup | Very important: single-run greedy extraction is an unstable measure. Your design should consider repeated sampling or extraction probability. |

## What Is Already Known

The literature already supports all of the following:

- LLMs can emit memorized training text under the right prompt conditions.
- Copyrighted books and similar long-form materials can be memorized.
- Duplication and repetition materially affect extraction risk.
- Alignment and simple verbatim filters do not fully solve leakage.
- Release-time information can be used to study contamination and benchmark validity.
- "New knowledge" and "post-cutoff" evaluation are active topics.
- Likelihood-based membership inference can distinguish pre vs post-cutoff texts (Shi et al., Meeus et al.).

That means the project should not claim novelty for:

- studying memorization in LLMs
- using prefix prompts to test extraction
- using cutoff dates in evaluation
- studying copyrighted or creative text in general
- using temporal release date as a signal for pre/post-cutoff membership

## The Specific Gap This Project Can Target

The gap that still appears open is the intersection of:

1. Real creative texts rather than synthetic facts or code/math problems.
2. Temporally documented post-cutoff release dates.
3. Prefix-completion probing rather than multiple-choice detection or QA.
4. Graded prefix reveal percentages to test how context length changes continuation behavior.
5. Cross-provider comparison using a single evaluation protocol.

In other words:

> Prior work has studied memorization extraction, copyrighted-book recall, and time-based contamination separately. It does not appear to have systematically studied continuation of documented post-cutoff creative texts across graded prefix lengths and model families.

This is an inference from the papers reviewed above and from targeted primary-source search as of 2026-03-27. I did not find a primary-source paper centered on songs or podcast transcripts as the main memorization benchmark.

## Recommended Contribution Statement

Use something close to this:

> Prior memorization work has largely focused on extracting likely training examples, measuring copyrighted-text recall in likely pretraining-era corpora, or studying post-cutoff contamination in code and QA benchmarks. We study a different regime: whether LLMs can continue creative texts released after their documented cutoff dates, under controlled prefix reveal, and whether any apparent continuation is verbatim, near-verbatim, lexical, or merely semantic.

## Novelty Risks and How To Defuse Them

### 1. "Guaranteed post-training content"

Risk:
Closed models may have undocumented continued training, retrieval, or product updates.

Fix:
Say "documented post-cutoff content" or "content released after the model's stated knowledge cutoff."

### 2. "Cross-modal"

Risk:
The current data are all text.

Fix:
Say "cross-domain creative-text" or "across creative content domains."

### 3. "First systematic study"

Risk:
Too easy to falsify because several neighboring studies exist.

Fix:
Narrow the claim to post-cutoff creative-text continuation with graded prefix reveal.

### 4. Exact-match overclaim

Risk:
Exact match alone can miss leakage and overstate "no memorization."

Fix:
Keep fuzzy, overlap, and semantic metrics, and consider repeated generations per item/condition.

## Methodological Implications for This Repo

The literature suggests several upgrades to the current design:

- Add repeated generations per prompt condition to estimate extraction probability, not just one-shot success.
- Record decoding settings explicitly; memorization behavior depends on sampling.
- Separate "verbatim memorization" from "semantic plausibility" in the analysis and claims.
- Track potential duplication/exposure proxies for each text, especially for lyrics or highly reposted transcripts.
- Add a stronger temporal verification protocol for every item in the dataset.

These are not required for the project to be useful, but they would make the novelty and validity arguments much stronger.

## Suggested Related Work Structure for the Paper

1. Memorization and extraction in language models.
2. Copyrighted text redistribution and book memorization.
3. Data contamination and training-cutoff evaluation.
4. Benchmarks for new knowledge and temporal freshness.
5. Gap: post-cutoff creative-text continuation.

## Primary Sources

### Core reading list (13 original papers)

1. Carlini et al. 2021. *Extracting Training Data from Large Language Models*. USENIX Security. https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting
2. Kandpal et al. 2022. *Deduplicating Training Data Mitigates Privacy Risks in Language Models*. ICML / PMLR. https://proceedings.mlr.press/v162/kandpal22a.html
3. Mireshghallah et al. 2022. *An Empirical Analysis of Memorization in Fine-tuned Autoregressive Language Models*. EMNLP. https://aclanthology.org/2022.emnlp-main.119/ *(de-prioritise — tangential)*
4. Chang et al. 2023. *Speak, Memory: An Archaeology of Books Known to ChatGPT/GPT-4*. EMNLP. https://aclanthology.org/2023.emnlp-main.453/
5. Karamolegkou et al. 2023. *Copyright Violations and Large Language Models*. EMNLP. https://aclanthology.org/2023.emnlp-main.458/
6. Ippolito et al. 2023. *Preventing Generation of Verbatim Memorization in Language Models Gives a False Sense of Privacy*. INLG. https://aclanthology.org/2023.inlg-main.3/
7. Yin et al. 2023. *ALCUNA: Large Language Models Meet New Knowledge*. EMNLP. https://aclanthology.org/2023.emnlp-main.87/ *(de-prioritise — thin overlap, drop if space is tight)*
8. Roberts et al. 2024. *To the Cutoff... and Beyond? A Longitudinal Perspective on LLM Data Contamination*. ICLR. https://openreview.net/forum?id=m2NVG4Htxs
9. Vu et al. 2024. *FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation*. Findings of ACL. https://aclanthology.org/2024.findings-acl.813/ *(de-prioritise — low overlap, drop if space is tight)*
10. Duarte et al. 2024. *DE-COP: Detecting Copyrighted Content in Language Models Training Data*. ICML / PMLR. https://proceedings.mlr.press/v235/duarte24a.html
11. Huang et al. 2024. *Demystifying Verbatim Memorization in Large Language Models*. EMNLP. https://aclanthology.org/2024.emnlp-main.598/ *(cite but don't lean on — synthetic sequences)*
12. Nasr et al. 2025. *Scalable Extraction of Training Data from Aligned, Production Language Models*. ICLR. https://openreview.net/forum?id=vjel3nWP2a
13. Hayes et al. 2025. *Measuring memorization in language models via probabilistic extraction*. NAACL. https://aclanthology.org/2025.naacl-long.469/

### Recommended additions (not yet in reading list)

14. Shi et al. 2024. *Detecting Pretraining Data from Large Language Models*. ICLR. https://openreview.net/forum?id=zWqr3MQuNs *(must read — temporal pre/post-cutoff framing is very close to this project; novelty claim must distinguish from it)*
15. Meeus et al. 2024. *Did the Neurons Read this Book Too? Risks of Reconstruction and Exfiltration of Pre-Training Data*. USENIX Security. https://www.usenix.org/conference/usenixsecurity24/presentation/meeus *(must read — pre/post-cutoff book membership inference; must distinguish from this)*

## Bottom Line

The project looks plausibly novel if it is framed narrowly and carefully:

- not "a first study of memorization"
- not "a first post-cutoff benchmark"
- but plausibly "a first study of post-cutoff creative-text continuation as a memorization probe"

**Two papers must be added to the reading list before writing the related work section:** Shi et al. 2024 (MIN-K%) and Meeus et al. 2024 both use temporal pre/post-cutoff framing on real texts. The novelty claim holds if you carefully distinguish your contribution (free-form continuation with graded prefix reveal on songs/podcasts) from their detection-via-likelihood approach on books/web text.

The gap argument — *continuation of documented post-cutoff creative texts with graded prefix reveal* — holds up against everything reviewed here.
