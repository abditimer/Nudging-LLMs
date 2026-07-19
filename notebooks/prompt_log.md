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

## v3 — Explicit continuation

**Date:** 2026-07-17

```
Complete the text below. Rules:
1. Output only the continuation—no title, explanation, quotation marks, or labels.
2. Do not repeat any text from <StartText>.
3. Write close to but at most {target_word_count} whitespace-separated words.
4. Stop immediately after the final word.
5. do not leave it up to me, it is up to you to continue the text in a way that is consistent with the style, tone, and content of the text in <StartText>.

<StartText>
{context_text}
</StartText>

Continuation:
```

**Reasoning:** The model was not sticking to the agreed amount, so testing a more formal approach that has enforcement mechanism. had to add something on getting close to the word count as it would only spit out a few words.
**Observed issue:** The model can still stop far short of the target length,
because both the prompt and `num_predict` give it an upper bound rather than a
minimum length requirement.

---

## v4 — Length-encouraging continuation

**Date:** 2026-07-19

```
Complete the text below. Rules:
1. Output only the continuation—no title, explanation, quotation marks, or labels.
2. Do not repeat any text from <StartText>.
3. Continue with approximately {target_word_count} whitespace-separated words.
4. Do not finish early; keep writing until the continuation is close to that length.
5. Never exceed {target_word_count} whitespace-separated words.
6. Continue consistently with the style, tone, and content of <StartText>.

<StartText>
{context_text}
</StartText>

Continuation:
```

**Reasoning:** v3's “close to but at most” wording allowed very short answers.
v4 keeps the hard maximum but explicitly asks the model to continue until it is
near the target length.

**Observed issue:** To be tested in `metrics.ipynb` and `metrics_clean.ipynb`.
