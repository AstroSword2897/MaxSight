#!/usr/bin/env python3
"""Enforce single-line comments and docstrings with consistent, natural wording across the repo."""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MAX_DOCSTRING_SINGLE_LINE = 200
# Merge consecutive comment lines only when the result stays under this length so we avoid multiline blocks.
MAX_MERGED_COMMENT_LEN = 120


def _make_natural(s: str) -> str:
    """Format comment or docstring so it reads as one sentence and matches repo style (capitalize, period)."""
    s = s.strip()
    if not s:
        return s
    if s.startswith("#"):
        s = s[1:].lstrip()
    low = s.lower()
    if not s or low.startswith("type:") or low.startswith("noqa") or "://" in s:
        return s
    if s.lstrip().startswith("!/") or s.lstrip().startswith("! "):
        return s
    # Rephrase Returns/Args/Raises labels so they read as full sentences instead of "Label: rest".
    for label in ("returns:", "arguments:", "args:", "raises:", "yields:"):
        if low.startswith(label) and len(s) > len(label):
            rest = s[len(label) :].strip()
            if rest:
                s = s[: len(label)].rstrip(":") + " " + rest
            break
    if len(s) > 1 and s[0].isalpha() and not s[0].isupper():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?)'\"" and not s.endswith("..."):
        s = s.rstrip() + "."
    return s


def normalize_inline_comment(s: str) -> str:
    """Apply consistent style to one comment; leave type:, noqa, and URLs unchanged so tooling still works."""
    return _make_natural(s)


def normalize_docstring_content(content: str) -> str:
    """Collapse multiline docstring to one line and apply same style so it matches the rest of the repo."""
    if not content:
        return content
    one = " ".join(line.strip() for line in content.strip().splitlines()).strip()
    one = re.sub(r"\s+", " ", one)
    one = one.rstrip().removesuffix("...").rstrip()
    if not one:
        return one
    return _make_natural(one)


def process_file(path: Path) -> bool:
    """Normalize # comments and convert multiline docstrings to single-line where possible."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Skip {path}: {e}", file=sys.stderr)
        return False
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    changed = False

    def is_shebang(s: str) -> bool:
        t = s.strip()
        return t.startswith("#!") or t.startswith("# !/")

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent_len = len(line) - len(stripped)
        prefix = line[:indent_len]

        # Merge short standalone # blocks into one line so we avoid multiline comments. Skip shebangs so the script still runs.
        if stripped.startswith("#") and not is_shebang(stripped):
            block: list[tuple[str, str]] = []
            while (
                i < len(lines)
                and lines[i].strip().startswith("#")
                and not is_shebang(lines[i].strip())
            ):
                ln = lines[i]
                ind = len(ln) - len(ln.lstrip())
                pfx = ln[:ind]
                part = ln.strip()[1:].strip()
                if (
                    part
                    and not part.strip().lower().startswith("type:")
                    and "://" not in part
                    and not part.strip().lower().startswith("noqa")
                ):
                    block.append((pfx, part))
                    changed = True
                else:
                    if block:
                        merged = " ".join(p for _, p in block)
                        single = normalize_inline_comment(merged)
                        if len(single) <= MAX_MERGED_COMMENT_LEN:
                            out.append(block[0][0] + "# " + single)
                        else:
                            for pfx_b, part_b in block:
                                out.append(pfx_b + "# " + normalize_inline_comment(part_b))
                        block = []
                    out.append(ln)
                i += 1
            if block:
                merged = " ".join(p for _, p in block)
                single = normalize_inline_comment(merged)
                if len(single) <= MAX_MERGED_COMMENT_LEN:
                    out.append(block[0][0] + "# " + single)
                else:
                    for pfx_b, part_b in block:
                        out.append(pfx_b + "# " + normalize_inline_comment(part_b))
            continue

        # Treat as docstring only after def/class or shebang so we do not touch string literals or f-string continuations.
        prev_ok = (
            not out
            or out[-1].rstrip().endswith(":")
            or (len(out) == 1 and is_shebang(out[0].strip()))
        )
        first_dq = line.find('"""')
        first_sq = line.find("'''")
        first_quote = min(first_dq if first_dq >= 0 else 9999, first_sq if first_sq >= 0 else 9999)
        has_assign_before_quote = first_quote < 9999 and "=" in line[:first_quote]
        if (
            prev_ok
            and (stripped.startswith('"""') or stripped.startswith("'''"))
            and not has_assign_before_quote
        ):
            quote = '"""' if stripped.startswith('"""') else "'''"
            # Docstring fits on one line; normalize and keep on one line.
            if stripped.rstrip().endswith(quote) and len(stripped) > len(quote) * 2:
                body = stripped[len(quote) : -len(quote)].strip()
                if body:
                    norm = normalize_docstring_content(body)
                    if norm != body:
                        out.append(prefix + quote + norm + quote)
                        changed = True
                    else:
                        out.append(line)
                else:
                    out.append(line)
                i += 1
                continue
            # Collect lines until the closing quote so we have the full docstring body to collapse.
            doc_lines = [stripped[len(quote) :].rstrip()]
            i += 1
            while i < len(lines):
                rest = lines[i]
                if quote in rest:
                    # Take content before the closing quote so the body is complete.
                    idx = rest.find(quote)
                    doc_lines.append(rest[:idx].rstrip())
                    i += 1
                    break
                doc_lines.append(rest.rstrip())
                i += 1
            body = "\n".join(doc_lines).strip()
            norm = normalize_docstring_content(body)
            if len(norm) <= MAX_DOCSTRING_SINGLE_LINE:
                out.append(prefix + quote + norm + quote)
            else:
                # Use first sentence when docstring is long so the one-line summary stays under the length limit.
                first_sent = (
                    (norm.split(". ")[0] + ".")
                    if ". " in norm
                    else (norm.rstrip() + "." if norm and norm[-1] != "." else norm)
                )
                if len(first_sent) <= MAX_DOCSTRING_SINGLE_LINE:
                    out.append(prefix + quote + first_sent + quote)
                else:
                    out.append(prefix + quote + norm + quote)
            changed = True
            continue

        # Normalize inline # only when it is not inside a string literal so we do not break code.
        if "#" in line:
            idx = line.find("#")
            before = line[:idx]
            if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                comment = line[idx + 1 :].strip()
                clow = comment.strip().lower()
                if (
                    comment
                    and not clow.startswith("type:")
                    and not clow.startswith("noqa")
                    and "://" not in comment
                ):
                    new_comment = normalize_inline_comment(comment)
                    if new_comment != comment:
                        line = line[: idx + 1] + " " + new_comment
                        changed = True
        out.append(line)
        i += 1

    if changed:
        path.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return changed


def main() -> int:
    py_files = [p for p in REPO.rglob("*.py") if ".git" not in p.parts]
    modified = 0
    for path in sorted(py_files):
        if process_file(path):
            modified += 1
            print(path.relative_to(REPO))
    print(f"Modified {modified} files.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
