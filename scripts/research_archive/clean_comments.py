#!/usr/bin/env python3
"""Remove verbose comments from Python files, keeping minimal essential ones."""

import re
import sys
from pathlib import Path


def clean_docstring(docstring: str) -> str:
    """Remove verbose philosophy/explanation sections, keep brief summary."""
    if not docstring or len(docstring) < 50:
        return docstring

    lines = docstring.split("\n")
    cleaned = []
    skip_section = False

    for line in lines:
        line_lower = line.lower()
        if any(
            phrase in line_lower
            for phrase in [
                "why this",
                "design philosophy",
                "how it connects",
                "relationship to",
                "technical design decision",
                "project philosophy",
                "barrier removal",
            ]
        ):
            skip_section = True
            continue
        if skip_section and (line.strip() == "" or line.strip().startswith("-")):
            continue
        if skip_section and line.strip() and not line.startswith(" "):
            skip_section = False

        if not skip_section:
            cleaned.append(line)

    result = "\n".join(cleaned).strip()
    if len(result) > 200:
        first_line = result.split("\n")[0] if "\n" in result else result[:200]
        return first_line + "..." if len(result) > 200 else result
    return result


def clean_file_comments(content: str) -> str:
    """Remove verbose inline comments, keep essential ones."""
    lines = content.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("#"):
            if any(
                phrase in stripped.lower()
                for phrase in ["todo", "fixme", "hack", "xxx", "note:", "warning:", "important:"]
            ):
                cleaned.append(line)
            elif len(stripped) > 80 or any(
                phrase in stripped.lower()
                for phrase in ["location:", "assumes", "does not handle", "not tested"]
            ):
                continue
            else:
                cleaned.append(line)
        elif "# " in line and not line.strip().startswith("#"):
            parts = line.split("#", 1)
            code_part = parts[0].rstrip()
            comment = parts[1].strip()

            if any(
                phrase in comment.lower() for phrase in ["type:", "type ignore", "noqa", "fmt:"]
            ):
                cleaned.append(line)
            elif len(comment) > 60:
                cleaned.append(code_part)
            elif any(
                phrase in comment.lower()
                for phrase in ["serialize", "thread-safe", "kill switch", "hard disable"]
            ):
                cleaned.append(code_part + "  # " + comment[:50])
            else:
                cleaned.append(line)
        else:
            cleaned.append(line)

    return "\n".join(cleaned)


def process_file(file_path: Path) -> tuple[bool, str]:
    """Process a single Python file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original = content

        if '"""' in content:
            pattern = r'"""(.*?)"""'

            def replace_docstring(match):
                doc = match.group(1)
                cleaned = clean_docstring(doc)
                return f'"""{cleaned}"""' if cleaned else '""""""'

            content = re.sub(pattern, replace_docstring, content, flags=re.DOTALL)

        content = clean_file_comments(content)

        if content != original:
            file_path.write_text(content, encoding="utf-8")
            return True, "cleaned"
        return False, "no changes"
    except Exception as e:
        return False, f"error: {e}"


def main():
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path(__file__).parent.parent

    if target.is_file():
        files = [target]
    else:
        files = list(target.rglob("*.py"))
        files = [
            f
            for f in files
            if ".git" not in str(f) and "venv" not in str(f) and "__pycache__" not in str(f)
        ]

    print(f"Processing {len(files)} Python files...")
    cleaned_count = 0

    for file_path in sorted(files):
        changed, status = process_file(file_path)
        if changed:
            cleaned_count += 1
            print(
                f"  {file_path.relative_to(target.parent if target.is_file() else target)}: {status}"
            )

    print(f"\nCleaned {cleaned_count}/{len(files)} files.")


if __name__ == "__main__":
    main()
