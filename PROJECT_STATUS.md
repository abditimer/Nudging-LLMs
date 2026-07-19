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

## Pilot Configuration (Frozen)

- pilot name and output filename, e.g. `pilot_600_v2`; pilot_600_v1
- the two exact Ollama model names;
- a deterministic list of the 30 source-text IDs;
- contexts: [0, 25, 50, 75, 90]
- temperatures [0.0, 0.7]
- prompt version `v2_2026_07_17`
- token multiplier: `1.5`
- fixed seed
- primary condition: token-capped generation, then trim to target words.

## Skill / Agent Guide Decision

`CLAUDE.md` is currently a repo-local agent guide. It should stay in the repo because it describes the exact project layout, current experiment setup, and what not to change without asking.

A Codex skill would be useful later if the workflow becomes reusable across projects, for example:

- designing memorization experiments
- validating prompt logs
- adding extraction-probability runs
- checking novelty claims against the literature

Do not simply copy all of `CLAUDE.md` into a skill. A good skill should be shorter and procedural, with project-specific details kept in repo files like this one.
