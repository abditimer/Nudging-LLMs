# Prompt Iteration Log

A record of prompts tried, why they changed, and what was observed.

---

## v1 — Original

**Date:** 2026-03-27
**Location:** `nudging/experiment.py` `_generate_response()`

```
Generate characters when you see <Generate>
You must generate exactly {test_words_count} words!
Continue the text that comes after <StartText>.
<StartText>
{context}
</StartText>
<Generate>
```

**Reasoning:** Initial prompt. Uses XML-style tags to structure input/output.
**Observed issue:** Model restarts the song from the beginning instead of continuing from the cutoff point. Generated text matches the song's opening, not the withheld target portion. Near-zero exact match scores as a result.

---

## v2 — Explicit continuation

**Date:** 2026-03-27

```
The following is an incomplete text. Continue it from exactly where it ends — do not repeat any of the text shown. Write exactly {test_words_count} words.

<StartText>
{context}
</StartText>

Continue from here:
```

**Reasoning:** Explicitly tells the model not to repeat shown text. Removes `<Generate>` tag (noise for small models). Simpler structure better suited to `qwen2.5:0.5b`.
**Observed issue:** TBD

---

## v2 — Explicit continuation

**Date:** 2026-07-17

````Complete the text below.

Rules:
- Output only the continuation—no title, explanation, quotation marks, or labels.
- Do not repeat any text from <StartText>.
- Write close to but at most {target_word_count} whitespace-separated words.
- Stop immediately after the final word.
- do not leave it up to me, it is up to you to continue the text in a way that is consistent with the style, tone, and content of the text in <StartText>.

<StartText>
{context_text}
</StartText>

Continuation:```

**Reasoning:** The model was not sticking to the agreed amount, so testing a more formal approach that has enforcement mechanism. had to add something on getting close to the word count as it would only spit out a few words.
**Observed issue:** TBD
````

---

## Next research steps — 2026-07-17

Move repeatable runs from `notebooks/metrics.ipynb` into the experiment runner, keeping the notebook for inspection, plots, and narrative reporting.

1. Establish a controlled baseline across 20–50 texts at context percentages `[0, 25, 50, 75, 90]`, with a fixed prompt and `temperature=0.0`.
2. Add controls: correct context, no context, and context from a different song. These distinguish continuation from generic lyric vocabulary or prior model knowledge.
3. Make word-position accuracy and n-gram overlap the primary recovery metrics. Keep fuzzy similarity and token overlap as descriptive secondary metrics, and compare them with the controls.
4. Record one row per actual generation condition, with raw output length stored as metadata. Do not include separate trimmed rows when every candidate is trimmed before scoring.
5. Ensure configured temperature is passed from `ExperimentConfig` to `OllamaClient.generate`; the runner currently falls back to the client's default temperature.
6. After the baseline, vary model size/family and then temperature, with repeated runs for stochastic settings.

Use the useful discipline from Karpathy's Autoresearch—fixed configuration, automated runs, append-only results, and stable evaluation—but do not let an agent autonomously modify prompts, metrics, and hypotheses to maximise overlap.
