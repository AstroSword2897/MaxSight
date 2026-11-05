"""Ticket 302: the licence register has one row per dataset with the required columns and an explicit approval.

Rules checked:
  * the register table has exactly the columns the ticket asks for;
  * every source in configs/data/sources.yaml, and every dataset named in the ticket, has a row;
  * Approved is Y or N, never blank;
  * no dataset whose terms are unverified is approved.
"""

from __future__ import annotations

from pathlib import Path

from ml.datasets.sources import load_registry

REPO = Path(__file__).resolve().parents[2]
REGISTER = REPO / "docs" / "data-governance" / "licence-register.md"

COLUMNS = [
    "Dataset", "URL", "Version", "Date checked", "Licence", "Commercial use", "Redistribution",
    "Attribution", "Role in MaxSight", "Approved (Y/N)",
]
# Datasets named in ticket 302 that are not (yet) in the registry.
TICKET_EXTRAS = ("RampNet", "Waymo Open Dataset", "NYU Depth V2")


def register_rows() -> list[dict[str, str]]:
    """Rows of the first table whose header starts with 'Dataset'."""
    lines = REGISTER.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Dataset |"))
    header = [cell.strip() for cell in lines[start].strip("|").split("|")]
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]  # ignore bold markup
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def test_register_has_the_ticket_columns():
    first = register_rows()[0]
    assert list(first) == COLUMNS


def test_every_registered_source_and_ticket_dataset_has_a_row():
    names = [row["Dataset"] for row in register_rows()]
    expected = [source.name for source in load_registry().values()] + list(TICKET_EXTRAS)
    missing = [name for name in expected if not any(name in row_name for row_name in names)]
    assert not missing, f"datasets without a register row: {missing}"


def test_approval_is_always_explicit():
    for row in register_rows():
        assert row["Approved (Y/N)"][:1] in ("Y", "N"), row["Dataset"]


def test_no_dataset_with_unverified_terms_is_approved():
    for row in register_rows():
        if row["Approved (Y/N)"].startswith("Y"):
            assert "unverified" not in row["Licence"].lower(), row["Dataset"]
            assert "unverified" not in row["Redistribution"].lower(), row["Dataset"]
