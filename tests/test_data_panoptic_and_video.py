import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset import MaxSightDataset
from ml.data.data_pipeline import collate_fn


def _write_json(tmp_dir: Path, name: str, data: Dict[str, Any]) -> Path:
    path = tmp_dir / name
    path.write_text(json.dumps(data))
    return path


def test_panoptic_annotations_parsed_to_objects(tmp_path: Path) -> None:
    images = [
        {"id": 1, "file_name": "img1.jpg", "width": 200, "height": 100},
    ]
    categories = [
        {"id": 10, "name": "car"},
    ]
    annotations = [
        {
            "image_id": 1,
            "segments_info": [
                {"id": 1, "category_id": 10, "bbox": [50, 20, 80, 40]},
            ],
        }
    ]
    panoptic_data = {"images": images, "annotations": annotations, "categories": categories}
    ann_path = _write_json(tmp_path, "panoptic.json", panoptic_data)
    img_path = tmp_path / "images"
    img_path.mkdir()
    dummy = torch.randint(0, 255, (100, 200, 3), dtype=torch.uint8).numpy()
    from PIL import Image

    Image.fromarray(dummy).save(img_path / "img1.jpg")
    dataset = MaxSightDataset(
        data_dir=tmp_path,
        annotation_file=ann_path,
        image_dir=img_path,
        audio_dir=None,
        condition_mode=None,
        apply_lighting_augmentation=False,
        max_objects=5,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["num_objects"].item() == 1
    boxes = sample["boxes"]
    cx, cy, w, h = boxes[0].tolist()
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert w > 0.0
    assert h > 0.0
    assert sample["urgency"].item() in (0, 3)


def test_sequence_collate_stacks_frames_and_lengths() -> None:
    b = 2
    t1, t2 = 3, 5
    c, h, w = 3, 4, 4
    batch: List[Dict[str, Any]] = []
    for t in (t1, t2):
        frames = torch.randn(t, c, h, w)
        item = {
            "frames": frames,
            "images": frames[0],
            "labels": torch.zeros(4, dtype=torch.long),
            "boxes": torch.zeros(4, 4, dtype=torch.float32),
            "distance": torch.zeros(4, dtype=torch.long),
            "num_objects": torch.tensor(0, dtype=torch.long),
            "urgency": torch.tensor(0, dtype=torch.long),
        }
        batch.append(item)
    collated = collate_fn(batch)
    assert collated["images"].shape == (b, max(t1, t2), c, h, w)
    assert torch.equal(collated["frame_lengths"], torch.tensor([t1, t2]))

