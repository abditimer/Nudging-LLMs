# AI Agent Guide: Nudging Experiment

This file helps AI agents (Claude Code and others) quickly understand this project and contribute effectively without wasting context on re-discovery.

For the current handoff state, read `PROJECT_STATUS.md` first. This file is the stable repo guide; `PROJECT_STATUS.md` tracks where the work was left and what should happen next.

---

## What This Project Is

A PhD research project studying **LLM memorization**. The core question:

> Can LLMs reproduce creative content (songs, podcasts) released **after their training cutoff date**?

The "nudging" technique gives a model increasing percentages of a text (0%, 25%, 60%, 90%) as context and measures how well it can complete the rest. High exact match scores on post-training content would indicate memorization.

**4 Primary Hypotheses (from README.md):**
1. Completion accuracy will be near-zero for exact matches (<5%) for guaranteed post-training content
2. Songs will show higher completion than prose
3. Commercial models will outperform open-source models
4. Longer prefix provides context but does not enable memorization

---

## Project Structure

```
nudging/
├── nudging/           # Core source package
│   ├── metrics.py     # All 4 evaluation metrics
│   ├── experiment.py  # run_single_experiment(), run_experiments()
│   ├── models.py      # OllamaClient (local Ollama HTTP wrapper)
│   └── data_loader.py # load_data(), text preprocessing
├── experiments/       # Runnable experiment scripts
├── configs/           # ExperimentConfig, ModelConfig, DataConfig dataclasses
├── notebooks/         # Exploratory Jupyter notebooks
├── results/           # CSVs, figures, logs (git-ignored outputs)
├── data/              # NOT in repo — songs and podcasts as .txt files
└── tests/             # Unit tests
```

---

## The Metrics (nudging/metrics.py)

These are the 4 metrics used to compare **model-generated text** against the **target (withheld) text**:

### 1. `exact_match_score(generated, target) → float`
- **Method**: Character-by-character comparison after lowercasing
- **Formula**: `matching_chars / len(target)`
- **Range**: 0.0–1.0
- **What it means**: Strictest measure of verbatim reproduction. A score >0.05 on post-training content would be a significant finding.
- **Weakness**: Penalises heavily for minor shifts (punctuation, extra word at the start)

### 2. `fuzzy_match_score(generated, target) → float`
- **Method**: RapidFuzz Levenshtein (edit distance), normalised
- **Formula**: `fuzz.ratio(generated, target) / 100`
- **Range**: 0.0–1.0
- **What it means**: Tolerates small typos/insertions. Catches "near-verbatim" reproduction that exact match would miss.
- **Weakness**: Still position-sensitive; large reorderings score poorly

### 3. `token_overlap_score(generated, target) → float`
- **Method**: Jaccard similarity on word sets
- **Formula**: `|intersection| / |union|` of unique word sets
- **Range**: 0.0–1.0
- **What it means**: Did the model use the same vocabulary? Word-order agnostic — detects lexical overlap even when sentences are restructured.
- **Weakness**: Ignores word frequency and position; "the" appearing once counts the same as any other word

### 4. `semantic_similarity_score(generated, target) → float`
- **Method**: Cosine similarity of `all-MiniLM-L6-v2` sentence embeddings
- **Formula**: `dot(emb1, emb2) / (norm1 × norm2)`
- **Range**: typically 0.0–1.0
- **What it means**: Captures meaning/theme similarity regardless of wording. High here but low exact match means the model understands the topic but hasn't memorised the text.
- **Weakness**: Can be high for thematically similar but factually different text

---

## How the Experiment Works (nudging/experiment.py)

```
text → _get_split_text(text, percentage%)
         ├── context  = first N% of words  (shown to model)
         └── target   = remaining words     (withheld, used for scoring)

prompt = "Continue this text:<StartText>{context}</StartText>Continue:"

model_client.generate(prompt) → generated_response

score all 4 metrics(generated_response, target)
```

Two entry points:
- `run_single_experiment()` — returns exact match only (quick check)
- `run_experiments()` — returns all 4 metrics + metadata (use this for analysis)

---

## What Good Results Look Like

| Pattern | Interpretation |
|---|---|
| `exact_match ≈ 0`, `semantic_similarity` moderate | No memorization; model generating plausible continuations |
| `exact_match` rises with context % | Context guiding generation, not memorization |
| `exact_match > 0.05` at low context % | Potential memorization signal — investigate |
| `fuzzy > exact` by large margin | Model "almost" reproduces with small errors |
| `token_overlap` high, `exact` low | Correct vocabulary, wrong order/structure |

---

## Current Experimental Setup

- **Model**: `qwen2.5:0.5b-instruct` via local Ollama (`http://localhost:11434`)
- **Context percentages**: `[0, 25, 60, 90]`
- **Content**: Taylor Swift songs (post-training cutoff)
- **Delay between calls**: 5 seconds (rate limiting)
- **Max tokens**: Dynamic — set to `target_word_count` to control generation length

Config lives in `configs/experiment_config.py`. The active constant is `EXPERIMENT_BASELINE_ONLY_SONGS_QWEN`.

---

## Active Investigation (as of Feb 2026)

The open question being explored in `notebooks/metrics.ipynb`:

> Does **limiting max_tokens to the target word count** produce better or worse metric scores than unlimited generation?

Early observation: limited generation (~141 words) vs unlimited (~185 words). The hypothesis is that limiting generation reduces padding/hallucination and improves metric scores.

---

## Current Handoff (as of 2026-06-27)

The current working tree has active changes in:

- `notebooks/metrics.ipynb`
- `nudging/metrics.py`
- `LITERATURE_REVIEW.md`

The immediate task is to finish the metrics notebook investigation: compare unrestricted generation, model-level length limiting, and post-generation trimming against the same withheld target. The notebook has the scaffolding, but final metric outputs were not saved.

Important mismatch: the notebook is using a newer explicit continuation prompt, while `nudging/experiment.py` still uses the older prompt shown above. Before scaling experiments, pick the canonical prompt and record it in `notebooks/prompt_log.md`.

See `PROJECT_STATUS.md` for the full handoff checklist.

---

## Skill Decision

Keep this file as a repo-local agent guide. It is project-specific and should travel with the repository.

A future Codex skill could be created for the reusable workflow of designing and validating LLM memorization experiments, but it should not be a direct copy of this file. The skill should be short, procedural, and point agents back to repo-local files for project state.

---

## What NOT to Change Without Asking

- The 4 core metrics in `nudging/metrics.py` — these are the measurement foundation
- The prompt format in `experiment.py:_generate_response()` — any change invalidates prior results
- The train/test split logic in `_get_split_text()` — same reason

## Safe to Iterate On

- Config values (model, percentages, max_tokens, delay)
- Adding new metrics alongside existing ones
- Visualisation and analysis notebooks
- The experiment runner scripts in `experiments/`
