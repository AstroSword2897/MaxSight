"""Central label-string → index mapping for gold builds and validation.

The ``class_map_hash`` property is the canonical way to verify that two
artifacts were compiled with an identical class ordering.  Two mappers whose
``class_map_hash`` values differ must not be mixed in a single training run —
model output indices would silently diverge.
"""

from __future__ import annotations

import hashlib
import json
from typing import List, Sequence

from ml.data.gold.schema import LABEL_SPACE_ACCESSIBILITY_622
from ml.models.maxsight_cnn import COCO_CLASSES


class LabelMapper:
    """Maps raw class names from any adapter into a single target ontology index."""

    def __init__(self, source_space: str | None, target_space: str) -> None:
        # source_space reserved for coco80→622 style remaps without touching adapters.
        self.source_space = source_space
        self.target_space = target_space
        if target_space != LABEL_SPACE_ACCESSIBILITY_622:
            raise ValueError(
                f"LabelMapper: unsupported target_space {target_space!r}; "
                f"only {LABEL_SPACE_ACCESSIBILITY_622!r} is wired today."
            )
        self._name_to_idx = {name: i for i, name in enumerate(COCO_CLASSES)}
        # Stable ordered list guarantees hash is over identical structure.
        self._ordered_classes: List[str] = list(COCO_CLASSES)

    @property
    def num_classes(self) -> int:
        return len(self._name_to_idx)

    @property
    def class_map_hash(self) -> str:
        """SHA-256 over the ordered (index, name) pairs for this mapping.

        Two mappers with the same hash are guaranteed to produce the same integer
        index for every class name.  Embed in artifact meta and verify at load
        time to catch silent class-ordering drift.
        """
        payload = json.dumps(
            list(enumerate(self._ordered_classes)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def map_class_names(self, names: Sequence[str]) -> List[int]:
        """Return integer labels aligned to ``names`` (unknown name → 0)."""

        out: List[int] = []
        for n in names:
            if not isinstance(n, str):
                raise TypeError(f"LabelMapper expects str class names, got {type(n).__name__}")
            out.append(self._name_to_idx.get(n, 0))
        return out
