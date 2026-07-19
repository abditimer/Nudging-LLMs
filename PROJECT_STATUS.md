# Project Status: Nudging Experiment

Last updated: 2026-07-17

## Summary

This project studies whether LLMs can continue creative texts released after their documented training cutoff dates. The experiment reveals a prefix of a text, asks a model to continue it, and compares the generated continuation with the withheld target text.

The current working phase is measurement design: making sure generation length, prompting, and metrics produce defensible results before scaling to more models and content types.

### The Four Project Goals

From `README.md`, the project is organized around four research questions:

1. Measure how accurately LLMs complete creative content released after their training cutoff.
   - Working hypothesis: exact-match completion should be near zero, below roughly 5%, for genuinely post-cutoff content.
2. Test whether completion accuracy varies by content type.
   - Working hypothesis: songs may show higher completion than prose because lyrics are more repetitive and widely duplicated.
3. Test whether completion accuracy varies by model provider or model family.
   - Working hypothesis: commercial frontier models may behave differently from smaller open-source/local models.
4. Test how prefix length affects completion accuracy for unseen content.
   - Working hypothesis: longer prefixes provide local context but should not enable true memorization of unseen text.

## Pilot Configuration (Partially Frozen)

- pilot name and output filename: `pilot_600_v4` / `pilot_600_v4.csv`
- Ollama models: `qwen2.5:0.5b-instruct` and `llama3.2:1b-instruct-q4_K_M`
- contexts: `[0, 25, 50, 75, 90]`
- temperatures: `[0.0, 0.7]`
- prompt version: `v4` (the latest prompt recorded in `notebooks/prompt_log.md`)
- token multiplier: `1.5`
- fixed seed: `42`
- primary condition: token-capped generation, then trim to target words.

The text list is not frozen yet: the local dataset currently has three source
texts, rather than the 30 required for the pilot. Add the 30 exact text IDs to
`PILOT_600.selected_text_ids` before starting the full batch.

## Current Run: Two-Song Pilot

- configuration and output: `pilot_songs_40_v4` / `pilot_songs_40_v4.csv`
- source texts: `songs::taylor_swift::the_fate_of_ophelia` and
  `songs::taylor_swift::shake_it_off`
- grid: 2 texts × 5 contexts × 2 models × 2 temperatures = 40 runs
- all other settings match the partially frozen 600-run pilot above.

## Configuration Tracks

- `PILOT_600` in `configs/experiment_config.py` is the frozen batch-pilot
  configuration. It is the source of truth for saved, repeatable runs.
- `experimental()` in the same file creates a one-model configuration for
  `notebooks/metrics.ipynb` and `notebooks/metrics_clean.ipynb`. It is for
  prompt, length, and metric debugging only; its results are not batch data.

## Skill / Agent Guide Decision

`CLAUDE.md` is currently a repo-local agent guide. It should stay in the repo because it describes the exact project layout, current experiment setup, and what not to change without asking.

A Codex skill would be useful later if the workflow becomes reusable across projects, for example:

- designing memorization experiments
- validating prompt logs
- adding extraction-probability runs
- checking novelty claims against the literature

Do not simply copy all of `CLAUDE.md` into a skill. A good skill should be shorter and procedural, with project-specific details kept in repo files like this one.
