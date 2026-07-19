# 600-Run Memorisation Pilot: Ordered Implementation Guide

## Aim and boundary

Turn the one-text exploratory workflow in `notebooks/metrics.ipynb` and
`notebooks/metrics_clean.ipynb` into one reproducible 600-run pilot with the
smallest useful change to the existing project.

The notebooks remain the **test bench** for prompt and metric behaviour. The
package and terminal runner become the only path for saved batch results.

The fixed pilot grid is:

```text
30 texts × 5 context percentages × 2 models × 2 temperatures = 600 runs
```

Use contexts `[0, 25, 50, 75, 90]` and temperatures `[0.0, 0.7]`.

## Ordered checklist

### 1. Freeze and record the pilot design

Write a **Pilot Configuration (Frozen)** section in `PROJECT_STATUS.md`.
Record:

- pilot name and output filename, currently `pilot_600_v4`;
- the two exact Ollama model names;
- a deterministic list of the 30 source-text IDs;
- contexts `[0, 25, 50, 75, 90]`;
- temperatures `[0.0, 0.7]`;
- prompt version `v4`;
- token multiplier `1.5`;
- fixed seed, if supported by your Ollama/model combination;
- primary condition: token-capped generation, then trim to target words.

Use `notebooks/prompt_log.md` only to record the prompt wording, change date,
reason, and observations. Do not change the prompt while a batch is underway.

### 2. Create one canonical, versioned prompt in code

Do **not** parse or import `prompt_log.md` at runtime. Markdown is the research
record, not executable configuration.

Maintain the canonical prompt module, `nudging/prompt.py`, containing:

- the frozen v4 template from the notebook;
- a `build_continuation_prompt(version, context_text, target_word_count)`
  function;
- an error for an unknown version.

Add `prompt_version` to `ExperimentConfig`, and make
`nudging/experiment.py` call this builder instead of embedding a prompt.

The prompt log remains the research record; it is not read at runtime.

### 3. Make package generation exactly match the chosen notebook condition

Refactor the one-run function in `nudging/experiment.py` so it alone performs:

1. whitespace-word split into context and held-out target;
2. v4 prompt construction;
3. `num_predict = ceil(target_word_count * 1.5)`;
4. one Ollama generation;
5. trimming that output to `target_word_count` words;
6. scoring the trimmed output;
7. returning metadata and scores.

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

Keep the existing `exact_match` character-position score as the primary score
for this pilot. Do not add word-position accuracy or rename the existing CSV
column during this implementation pass.

Make semantic similarity an explicit optional setting. The notebook correctly
allows it to be off, but the package currently computes it for every run,
which would load the embedding model during the pilot.

### 6. Define one complete, auditable result row

One actual generation condition produces one row. Save decimal scores in the
file; show percentages only in terminal output, notebooks, and plots. The CSV
stores metadata, lengths, and scores only; it deliberately does not store the
prompt version or generated text.

```text
run_id
status
error
text_title
category
model
temperature
seed
context_percentage
context_words
target_words
num_predict
raw_generated_words
generated_words
raw_length_ratio
scored_length_ratio
exact_match
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
3. read `results/metrics/pilot_600_v4.csv` if present;
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

- v4 prompt contents and rejection of an unknown prompt version;
- `temperature` and `seed` forwarding;
- `num_predict` using the frozen `1.5` multiplier;
- trimming to the target word count;
- the configured result fields;
- stable `run_id` generation;
- skipping an already completed run on resume.

Do not unit-test Ollama itself.

### 9. Validate in increasing sizes

Run these in order:

1. one text × one context × one model × one temperature;
2. confirm the saved row's metadata, lengths, and scores;
3. a 10-run smoke test;
4. the full 600-run pilot.

Before step 4, confirm the model, temperature, seed, lengths, and scores, and
that restarting the runner skips completed IDs. Reviewing generated text is a
future validation goal, not a prerequisite for this pilot.

### Future validation goal: review generated text

After the batch workflow is established, add a deliberate text-review process
for a small, documented sample of conditions. Decide separately whether that
review should use temporary local output, a protected research artifact, or a
new optional export. Do not add generated text to the pilot CSV as part of the
current implementation.

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

The pilot is ready for analysis when `results/metrics/pilot_600_v4.csv` has
600 completed or documented-error rows, rerunning does not repeat completed
runs, all rows carry the frozen metadata and scores, and the final design is
recorded in both `PROJECT_STATUS.md` and `notebooks/prompt_log.md`.
