# Comment style guide

Use this when writing or updating comments anywhere in the repository. Apply the same style in `ml/`, `scripts/`, `tools/`, and `tests/`.

## Rules

1. **Explain intent, not code.** Answer why this exists, not what the line does.
2. **Keep to 1–3 sentences.** Use present tense and active voice.
3. **Avoid redundancy.** If the code is clear, omit the comment.
4. **Be consistent.** Start with a capital letter; end full sentences with a period.
5. **Add context for non-obvious choices.** Document trade-offs, assumptions, or references.
6. **Stay minimal.** One sentence is often enough; split complex ideas into short sentences.
7. **No jokes or personal notes.** Use `TODO:` for actionable follow-ups.
8. **Update comments when code changes.** Remove or fix outdated comments.

## Good vs bad

| Avoid | Prefer |
|-------|--------|
| `i += 1  # increment i` | `i += 1  # Advance to next slot so we do not overwrite.` |
| `# This is used to validate the input` | `# Validate input so invalid data never reaches the head.` |
| `# Add 1 to x` | `# Align index with zero-based layout.` |
| Long paragraph explaining every step | One line on intent; second line only if constraint or trade-off. |
| `# CRITICAL: do X` (vague) | `# Enforce contiguous layout so CUDA kernels can run without copy.` |
| `# TODO: fix later` | `# TODO: pass temporal state into encoder when API supports it.` |

## Examples in this repo

- **Intent:** "Clamp raw box logits before sigmoid to avoid extreme values and gradient explosion."
- **Context:** "Single GPU→CPU sync for the batch; per-item sync would break pipeline parallelism."
- **Constraint:** "Omit nested dict when scene graph is off so JIT trace sees a single value type."
