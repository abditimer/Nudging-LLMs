# Results Directory

This directory contains saved outputs from the memorisation experiments.

## Structure

- `metrics/` — one CSV row per attempted generation condition.
- `logs/` — terminal-style execution logs for each named configuration.
- `figures/` — later, publication-ready figures and tables.

## Running experiments

List the named configurations:

```bash
python experiments/run_memorisation_experiment.py --list-configs
```

Run the current one-text, one-model smoke grid:

```bash
python experiments/run_memorisation_experiment.py --config smoke
```

It runs 10 conditions:

```text
1 text × 1 model × 2 temperatures × 5 context percentages
```

For a single-condition check:

```bash
python experiments/run_memorisation_experiment.py --config smoke --max-runs 1
```

The smoke configuration writes:

```text
metrics/pilot_smoke_v3.csv
logs/pilot_smoke_v3.log
```

Completed `run_id` values are skipped when the same command is run again.

## CSV schema

The results CSV stores run metadata, length diagnostics, and numeric metrics.
It does **not** store prompt text or generated-response text.

| Group | Columns |
| --- | --- |
| Run status | `run_id`, `status`, `error` |
| Condition metadata | `text_title`, `category`, `model`, `temperature`, `seed`, `context_percentage` |
| Length diagnostics | `context_words`, `target_words`, `num_predict`, `raw_generated_words`, `generated_words`, `raw_length_ratio`, `scored_length_ratio` |
| Scores | `exact_match`, `fuzzy_match`, `token_overlap`, `semantic_similarity` |

Scores are stored as decimal values, such as `0.12`; format them as percentages
only in notebooks, tables, and figures.

## Interpretation note

`raw_generated_words` is the model's returned word count before target-length
trimming. The model can repeat instructions from the input prompt. Because the
CSV deliberately does not retain generated text, such repetition cannot be
audited from the CSV and may affect the numeric scores. Use the exploratory
notebooks to inspect individual generations before interpreting batch metrics.

`raw_length_ratio` is `raw_generated_words / target_words`. A value near `1.0`
means the model generated approximately the desired span; a value above `1.0`
means it over-generated before trimming; a value below `1.0` means it stopped
early. `scored_length_ratio` uses the post-trim word count, so it cannot exceed
`1.0`.

The current score names should be interpreted carefully:

- `exact_match` is character-position overlap, not all-or-nothing exact string equality.
- `fuzzy_match` is edit-distance similarity.
- `token_overlap` is shared unique vocabulary and can be high for generic lyric language.
- `semantic_similarity` is blank unless enabled in the experiment configuration; it is not evidence of memorisation.
