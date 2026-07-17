# Project Status: Nudging Experiment

Last updated: 2026-07-17

## Summary

This project studies whether LLMs can continue creative texts released after their documented training cutoff dates. The experiment reveals a prefix of a text, asks a model to continue it, and compares the generated continuation with the withheld target text.

The current working phase is measurement design: making sure generation length, prompting, and metrics produce defensible results before scaling to more models and content types.

## The Four Project Goals

From `README.md`, the project is organized around four research questions:

1. Measure how accurately LLMs complete creative content released after their training cutoff.
   - Working hypothesis: exact-match completion should be near zero, below roughly 5%, for genuinely post-cutoff content.
2. Test whether completion accuracy varies by content type.
   - Working hypothesis: songs may show higher completion than prose because lyrics are more repetitive and widely duplicated.
3. Test whether completion accuracy varies by model provider or model family.
   - Working hypothesis: commercial frontier models may behave differently from smaller open-source/local models.
4. Test how prefix length affects completion accuracy for unseen content.
   - Working hypothesis: longer prefixes provide local context but should not enable true memorization of unseen text.

## Current Experimental Shape

- Current model: `qwen2.5:0.5b-instruct` through local Ollama.
- Current content focus: Taylor Swift songs in the private `data/` directory.
- Main context percentages in config: `0`, `25`, `60`, `90`.
- Current notebook focus: one song at `70%` context to debug metrics and generation length.
- Main scoring metrics:
  - `exact_match_score`
  - `fuzzy_match_score`
  - `token_overlap_score`
  - `semantic_similarity_score`

## Where Things Were Left

The active work was in `notebooks/metrics.ipynb`.

The notebook takes one known song, reveals 70% of it, asks the model to generate the remaining 30%, and compares three candidate outputs:

1. Unrestricted model generation.
2. Generation constrained by passing `max_tokens` into `OllamaClient`.
3. Unrestricted generation trimmed after the fact to the target word count.

The open question was:

> Does limiting generation length to the withheld target length improve metric validity by reducing padding and hallucinated continuation?

The notebook has the comparison scaffolding, but the final metric outputs are not saved in the notebook state.

Update: `nudging/experiment.py` now has an initial implementation of the recommended main-pipeline approach:

1. compute the withheld target word count,
2. pass an approximate Ollama token budget via `num_predict = ceil(target_word_count * 1.5)`,
3. trim the raw model output to exactly the target word count before metric scoring,
4. return metadata including raw generated words, `num_predict`, and whether trimming/length control were applied.

This still needs validation with real notebook runs before treating it as final.

## Modified Files In The Current Working Tree

- `notebooks/metrics.ipynb`
  - Expanded into a step-by-step metrics exploration notebook.
  - Adds the limited-vs-unlimited-vs-trimmed generation comparison.
  - Uses a more explicit continuation prompt than the main experiment pipeline.
- `nudging/metrics.py`
  - Changed semantic similarity to lazily load `SentenceTransformer`.
  - This avoids loading the embedding model unless semantic similarity is actually requested.
- `LITERATURE_REVIEW.md`
  - Adds Shi et al. 2024 and Meeus et al. 2024 as important adjacent work.
  - Tightens the novelty claim to documented post-cutoff creative-text continuation with graded prefix reveal.

## Important Mismatch To Resolve

`notebooks/metrics.ipynb` is using a newer explicit continuation prompt:

```text
The following is an incomplete text.
Continue it from exactly where it ends. Do not repeat any of the text shown.
Write exactly {requested_words_count} words.

<StartText>
{context_text}
</StartText>

Continue from here:
```

But `nudging/experiment.py` still uses the older prompt:

```text
Continue this text:<StartText>
{context}
</StartText>
Continue:
```

Before running larger experiments, decide which prompt is canonical. Changing the prompt changes the experiment, so record the chosen prompt version in `notebooks/prompt_log.md`.

## What Has To Be Done Next

1. Run `notebooks/metrics.ipynb` end to end and save the metric outputs for the three candidate generation strategies.
2. Validate the new main-pipeline length-control implementation in `nudging/experiment.py` against the notebook comparison.
3. Decide whether to keep the current combined strategy as canonical:
   - model-level token cap with `num_predict`
   - post-generation trimming to target word count
4. Make the experiment output record decoding settings, especially:
   - model name
   - temperature
   - `num_predict` / max token setting
   - prompt version
   - whether post-generation trimming was applied
5. Add repeated generations per item and context percentage, so results estimate extraction probability rather than one-shot behavior.
6. Add or update tests around:
   - split logic
   - target word count handling
   - prompt construction
   - output trimming
   - metric behavior on empty strings and short examples
7. Install test dependencies in the active Python environment. `pytest` was not available when last checked.
8. Update the literature review before writing related work:
   - Shi et al. 2024 and Meeus et al. 2024 must be addressed directly.
   - Avoid claiming that release date is absolute proof of non-training exposure.
   - Prefer "documented post-cutoff content" over "guaranteed post-training content."

## Next Research Phase — 2026-07-17

Move repeatable runs from `notebooks/metrics.ipynb` into the experiment runner, keeping the notebook for inspection, plots, and narrative reporting.

1. Establish a controlled baseline across 20–50 texts at context percentages `[0, 25, 50, 75, 90]`, with a fixed prompt and `temperature=0.0`.
2. Add controls: correct context, no context, and context from a different song. These distinguish continuation from generic lyric vocabulary or prior model knowledge.
3. Make word-position accuracy and n-gram overlap the primary recovery metrics. Keep fuzzy similarity and token overlap as descriptive secondary metrics, and compare them with the controls.
4. Record one row per actual generation condition, with raw output length stored as metadata. Do not include separate trimmed rows when every candidate is trimmed before scoring.
5. Ensure configured temperature is passed from `ExperimentConfig` to `OllamaClient.generate`; the runner currently falls back to the client's default temperature.
6. After the baseline, vary model size/family and then temperature, with repeated runs for stochastic settings.

Borrow the useful discipline from Karpathy's Autoresearch—fixed configuration, automated runs, append-only results, and stable evaluation—but do not let an agent autonomously modify prompts, metrics, and hypotheses to maximise overlap.

## Skill / Agent Guide Decision

`CLAUDE.md` is currently a repo-local agent guide. It should stay in the repo because it describes the exact project layout, current experiment setup, and what not to change without asking.

A Codex skill would be useful later if the workflow becomes reusable across projects, for example:

- designing memorization experiments
- validating prompt logs
- adding extraction-probability runs
- checking novelty claims against the literature

Do not simply copy all of `CLAUDE.md` into a skill. A good skill should be shorter and procedural, with project-specific details kept in repo files like this one.

## Recommended Immediate Next Session

Start by resolving the metric notebook:

1. Open `notebooks/metrics.ipynb`.
2. Run the limited/unlimited/trimmed comparison.
3. Record the metric outputs and observations in the notebook.
4. Decide whether `max_tokens`, trimming, or both should become part of the main experiment.
5. Only then edit `nudging/experiment.py`.
