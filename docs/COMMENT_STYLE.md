# Comment style guide

**Apply this standard to every comment in the repository** (`ml/`, `scripts/`, `tools/`, `tests/`, `app/`). Comments should explain intent, stay concise, and stay consistent.

**No multi-line comments.** Use only single-line comments (`#`). One thought per line. For multiple points use multiple single-line comments, not one block or paragraph across lines.

---

## 1. Write comments like a human explaining intent

- Answer: *"If someone unfamiliar with this code read it, would they understand why this exists?"*
- Explain the reasoning behind the code, not what the line literally does.
- Use plain, clear language. Use technical terms when needed; avoid jargon only you would understand.

| Avoid | Prefer |
|-------|--------|
| `i += 1  # increment i` | `i += 1  # Advance to next index so we do not overwrite previous data.` |
| `# validate the input` | `# Validate input so invalid data never reaches the head.` |

---

## 2. Be concise but complete

- Keep comments to **1–3 sentences** unless explaining a complex algorithm.
- Stick to the **core reason** for the code.
- Cut filler: "this", "the", "thing", "stuff".

| Avoid | Prefer |
|-------|--------|
| Long paragraph describing every step | One line on intent; add a second only for a constraint or trade-off. |
| "This next part is used to handle all sorts of weird edge cases..." | "Normalize user input to lowercase for case-insensitive matching." |

---

## 3. Use active voice

- Active voice is clearer and conveys action and intent.
- Prefer verbs that describe what the code does, in **present tense**.

| Avoid | Prefer |
|-------|--------|
| "This function is used for validating the inputs" | "Validate inputs so all required fields are present." |
| "The buffer was being cleared" | "Clear the buffer before each batch." |

---

## 4. Avoid redundancy

- If the code already clearly expresses something, omit the comment.
- Comment only **why**, **how**, or **non-obvious consequences**.

| Avoid | Prefer |
|-------|--------|
| `x = x + 1  # add 1 to x` | `x = x + 1  # Align index with zero-based array.` |
| Restating variable names or operations | Focus on intent. |

---

## 5. Be consistent

- Use the same style for punctuation, capitalization, tense, and abbreviations.
- **Capitalize** the first letter of a comment: `# Update user cache`
- End full sentences with a period.
- Use **present tense**: "Calculate", "Check", "Return".

---

## 6. Include context, not code translation

- Explain **why** and **how it fits**, not the literal "what."
- When useful, note **dependencies, assumptions, or constraints**.

Example:

```python
# Align buffer indices with the sensor readout (sensor indexing starts at 1).
# Assumes buffer size matches sensor count.
```

---

## 7. Minimalism without sacrificing clarity

- One sentence is often enough.
- For a complex idea, use **short, logical sentences** instead of a long paragraph.

```python
# Split the dataset into training and test sets.
# Use stratified sampling to preserve class distribution.
```

- Avoid long blocks of explanation unless necessary.

---

## 8. No personal notes or humor

- Keep comments **professional and neutral**.
- No jokes, "magic numbers explained later", or vague placeholders ("fix this later").
- For follow-ups, use **TODO** with an actionable note: `# TODO: pass temporal state into encoder when API supports it.`

---

## 9. Explain non-obvious decisions

- Comment any **non-standard choice**, trade-off, or workaround.
- Give enough reasoning for future maintainers.
- Add references when helpful: links, paper titles, RFCs, issue numbers.

Example:

```python
# Use bubble sort here for stability and small dataset size; quicksort is not used.
```

---

## 10. Keep comments up to date

- Outdated comments mislead; remove or update them when you change code.
- Remove placeholder comments that no longer apply.

---

## Summary

1. **Explain intent, not code.**
2. **Keep to 1–3 sentences; use present tense and active voice.**
3. **Avoid redundancy and obvious explanations.**
4. **Be consistent:** capitalization, punctuation, tense.
5. **Add context:** assumptions, constraints, or reasoning for non-obvious choices.
6. **Stay minimal but clear;** split complex thoughts into short sentences.
7. **No jokes, personal notes, or vague placeholders;** use `TODO:` for actionable items.
8. **Document non-obvious decisions** with reasoning or references.
9. **Update or remove comments** when code changes.

---

## Good vs bad (quick reference)

| Avoid | Prefer |
|-------|--------|
| `i += 1  # increment i` | `i += 1  # Advance to next slot so we do not overwrite.` |
| `# This is used to validate the input` | `# Validate input so invalid data never reaches the head.` |
| `# Add 1 to x` | `# Align index with zero-based layout.` |
| Long paragraph explaining every step | One line on intent; second line only for constraint or trade-off. |
| `# CRITICAL: do X` (vague) | `# Enforce contiguous layout so CUDA kernels run without copy.` |
| `# TODO: fix later` | `# TODO: pass temporal state into encoder when API supports it.` |

---

## Examples from this repo

- **Intent:** "Clamp raw box logits before sigmoid to avoid extreme values and gradient explosion."
- **Context:** "Single GPU→CPU sync for the batch; per-item sync would break pipeline parallelism."
- **Constraint:** "Omit nested dict when scene graph is off so JIT trace sees a single value type."
