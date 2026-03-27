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
