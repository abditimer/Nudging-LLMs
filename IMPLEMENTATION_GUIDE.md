# 600-Run Memorisation Pilot: Ordered Implementation Guide

## Aim and boundary

Turn the one-text exploratory workflow in `notebooks/metrics.ipynb` and
`notebooks/metrics_clean.ipynb` into one reproducible 600-run pilot with the
smallest useful change to the existing project.

The notebooks remain the **test bench**: inspect a prompt, a single output, and
metric behaviour there. The package and terminal runner become the only path
for saved batch results.

The fixed pilot grid is:

```text
30 texts × 5 context percentages × 2 models × 2 temperatures = 600 runs
```

Use contexts `[0, 25, 50, 75, 90]` and temperatures `[0.0, 0.7]`.

## Ordered checklist

### 1. Freeze and record the pilot design

Write a new **Pilot Configuration (Frozen)** section in `PROJECT_STATUS.md`.
Record:

- pilot name and output filename, e.g. `pilot_600_v2`;
- the two exact Ollama model names;
- a deterministic list of the 30 source-text IDs;
- contexts `[0, 25, 50, 75, 90]`;
- temperatures `[0.0, 0.7]`;
- prompt version `v2_2026_07_17` (or the final name you choose);
- token multiplier `1.5`;
- fixed seed, if supported by your Ollama/model combination;
- primary condition: token-capped generation, then trim to target words.

Use `notebooks/prompt_log.md` only to record the prompt wording, change date,
reason, and observations. Do not change the prompt while a batch is underway.

### 2. Create one canonical, versioned prompt in code

Do **not** parse or import `prompt_log.md` at runtime. Markdown is the research
record, not executable configuration.

Add one small module, `nudging/prompts.py`, containing:

- the final v2 template from the notebook;
- a `build_continuation_prompt(version, context_text, target_word_count)`
  function;
- an error for an unknown version.

Add `prompt_version` to `ExperimentConfig`, and make
`nudging/experiment.py` call this builder instead of embedding its older
prompt. Save `prompt_version` in every result row.

This fixes the current mismatch: the notebook uses v2, while
`nudging/experiment.py` still uses the old `Continue this text` prompt.

### 3. Make package generation exactly match the chosen notebook condition

Refactor the one-run function in `nudging/experiment.py` so it alone performs:

1. whitespace-word split into context and held-out target;
2. v2 prompt construction;
3. `num_predict = ceil(target_word_count * 1.5)`;
4. one Ollama generation;
5. trimming that output to `target_word_count` words;
6. scoring the trimmed output;
7. returning metadata, scores, and the raw/trimmed output.

Forward `temperature` and `seed` explicitly to `OllamaClient.generate()`.
They are configured today but not passed by the runner, so generation quietly
uses the client default of `0.7`.

Keep unrestricted and token-capped comparisons in the notebooks only. They are
useful diagnostics but are not separate batch conditions; otherwise they double
the run count without answering the primary pilot question.

Remove the notebook-only `max_tokens=None` argument from direct Ollama calls.
Use `num_predict` for the intended cap.

### 4. Align configuration with the actual pilot

In `configs/experiment_config.py`, define one clearly named pilot config rather
than relying on the generic `EXPERIMENT_BASELINE` (which currently defaults to
one model and 40% context).

It should contain the frozen:

- model list;
- temperatures;
- contexts;
- prompt version;
- seed;
- token multiplier;
- selected text IDs or deterministic selection rule;
- output filename.

Do not use the current `max_samples` field to select 30 texts. In the data
loader it truncates **each text by a character percentage**; it does not limit
the number of source files. Add a clearly named selection mechanism instead,
such as `selected_text_ids` or `max_texts`.

### 5. Make metrics suitable for the research question

Keep existing `fuzzy_match` and `token_overlap` as descriptive secondary
metrics. They can be non-zero for generic lyric vocabulary and do not establish
recovery of the withheld continuation.

Add the notebook’s word-position accuracy as the primary recovery metric:

```text
matching lower-cased words at the same position / target word count
```

Keep the existing character-position score only if you label it precisely as
`character_position_accuracy`; its present name, `exact_match`, is misleading
because it is not all-or-nothing exact-string equality.

Make semantic similarity an explicit optional setting. The notebook correctly
allows it to be off, but the package currently computes it for every run,
which would load the embedding model during the pilot.

### 6. Define one complete, auditable result row

One actual generation condition produces one row. Save decimal scores in the
file; show percentages only in terminal output, notebooks, and plots.

```text
run_id
status
error
text_title
category
model
temperature
seed
prompt_version
context_percentage
context_word_count
target_word_count
num_predict
raw_generated_response
trimmed_generated_response
raw_generated_words
generated_words
word_position_accuracy
character_position_accuracy
fuzzy_match
token_overlap
semantic_similarity
```

Use a deterministic `run_id` based on text ID, model, temperature, context,
prompt version, and seed. This is what makes reliable resume behaviour possible.

### 7. Turn the existing runner into a resumable batch runner

Update `experiments/run_memorisation_experiment.py` to:

1. load the selected 30 texts;
2. loop over texts, contexts, models, and temperatures;
3. read `results/metrics/pilot_600_v2.csv` if present;
4. skip any completed `run_id`;
5. call the canonical one-run package function;
6. append each result immediately;
7. record failures as `status="error"` with the error message;
8. show concise progress, such as `142/600 complete`.

Appending after every attempt protects the experiment against Ollama failures
or a laptop restart. A rerun must resume rather than duplicate completed work.

No database, YAML layer, orchestration agent, or notebook batch loop is needed
for this pilot.

### 8. Add focused tests before live runs

Extend `tests/test_experiment.py` with fake-client tests for:

- the requested prompt version and prompt contents;
- `temperature` and `seed` forwarding;
- `num_predict` using the frozen `1.5` multiplier;
- trimming to the target word count;
- required result fields;
- stable `run_id` generation;
- skipping an already completed run on resume.

Do not unit-test Ollama itself.

### 9. Validate in increasing sizes

Run these in order:

1. one text × one context × one model × one temperature;
2. inspect the saved row and raw/trimmed text manually;
3. a 10-run smoke test;
4. the full 600-run pilot.

Before step 4, confirm that result rows contain the correct prompt version,
model, temperature, seed, lengths, and scores, and that restarting the runner
skips completed IDs.

### 10. Analyse only after the batch is complete

Use `experiments/evaluate_results.py` and a reporting notebook to aggregate
the saved CSV, compare model/context/temperature groups, and create plots.
Keep `metrics.ipynb` for explanatory examples and metric diagnostics, not as
the source of aggregate numbers.

## Research caveat: stochastic decoding

The `temperature=0.7` half has only one output per
text/context/model combination. It is a screening signal, not an estimate of a
model’s probability of reproducing a continuation.

Record a fixed seed if Ollama supports it, interpret any `0.7` difference
cautiously, and do not claim it is stable from one output. For a follow-up,
repeat the most informative conditions with several seeds and report a mean and
range (or confidence intervals).

Also report fuzzy/token overlap as weak descriptive evidence only. Strong
overlap on well-known songs can reflect training exposure, supplied context, or
generic vocabulary. The pilot supports claims about this selected dataset and
these local models—not all LLMs.

## Definition of done

The pilot is ready for analysis when `results/metrics/pilot_600_v2.csv` has
600 completed or documented-error rows, rerunning does not repeat completed
runs, all rows carry the frozen metadata and scores, and the final design is
recorded in both `PROJECT_STATUS.md` and `notebooks/prompt_log.md`.
