PROMPT_TEMPLATES = {
    "v3": (
        "Complete the text. Rules:\n"
        "1. Output only the continuation—no title, explanation, quotation marks, or labels.\n"
        "2. Do not repeat any text from <StartText>.\n"
        "3. Write close to but at most {target_word_count} whitespace-separated words.\n"
        "4. Stop immediately after the final word.\n"
        "5. do not leave it up to me, it is up to you to continue the text in a way "
        "that is consistent with the style, tone, and content of the text in <StartText>.\n\n"
        "<StartText>\n{context_text}\n</StartText>\n\n"
        "Continuation:"
    ),
}

def build_continuation_prompt(version, context_text, target_word_count):
    try:
        template = PROMPT_TEMPLATES[version]
    except KeyError:
        raise ValueError(f"Unknown prompt version: {version!r}. Known: {sorted(PROMPT_TEMPLATES)}")
    return template.format(context_text=context_text, target_word_count=target_word_count)