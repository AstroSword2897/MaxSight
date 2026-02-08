#!/usr/bin/env python3
"""Normalize all comments in .py files: single-line only, capitalize, period. No multi-line blocks."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def normalize_inline_comment(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.startswith("#"):
        s = s[1:].lstrip()
    if not s or s.startswith("type:") or s.startswith("noqa") or "://" in s:
        return s
    if len(s) > 1 and s[0].isalpha() and not s[0].isupper():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?)'\"" and not s.endswith("..."):
        s = s.rstrip() + "."
    return s


def process_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Skip {path}: {e}", file=sys.stderr)
        return False
    lines = text.split("\n")
    out = []
    i = 0
    changed = False
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        # Standalone comment line(s): normalize each to single-line style (cap, period). Skip shebangs.
        def is_shebang(s: str) -> bool:
            t = s.strip()
            return t.startswith("#!") or t.startswith("# !/")
        if stripped.startswith("#") and not is_shebang(stripped):
            while i < len(lines) and lines[i].strip().startswith("#") and not is_shebang(lines[i].strip()):
                ln = lines[i]
                indent = len(ln) - len(ln.lstrip())
                prefix = ln[:indent]
                part = ln.strip()[1:].strip()
                if part and not part.startswith("type:") and "://" not in part:
                    single = normalize_inline_comment(part)
                    out.append(prefix + "# " + single)
                    changed = True
                else:
                    out.append(ln)
                i += 1
            continue
        # Inline comment (only when # is not inside a string)
        if "#" in line:
            idx = line.find("#")
            before = line[:idx]
            if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                comment = line[idx + 1 :].strip()
                if comment and not comment.startswith("type:") and not comment.startswith("noqa") and "://" not in comment:
                    new_comment = normalize_inline_comment(comment)
                    if new_comment != comment:
                        line = line[: idx + 1] + " " + new_comment
                        changed = True
        out.append(line)
        i += 1
    if changed:
        path.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return changed


def main():
    py_files = list(REPO.rglob("*.py"))
    py_files = [p for p in py_files if "archive" not in p.parts and ".git" not in p.parts]
    modified = 0
    for path in sorted(py_files):
        if process_file(path):
            modified += 1
            print(path.relative_to(REPO))
    print(f"Modified {modified} files.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

