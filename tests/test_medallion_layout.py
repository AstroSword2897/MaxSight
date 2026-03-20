import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.medallion_layout import (  # noqa: E402
    build_coco_training_index,
    load_training_index,
    merge_video_into_index,
    path_relative_to_repo,
    resolve_coco_for_train,
    resolve_repo_path,
    write_training_index,
)


def test_path_relative_to_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    inner = repo / "a" / "b.txt"
    inner.parent.mkdir(parents=True)
    inner.touch()
    assert path_relative_to_repo(inner, repo) == str(Path("a/b.txt"))


def test_resolve_coco_for_train_roundtrip(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    coco = repo / "datasets" / "coco"
    coco.mkdir(parents=True)
    splits = repo / "silver"
    splits.mkdir(parents=True)
    train = splits / "tr.json"
    val = splits / "va.json"
    train.write_text("{}", encoding="utf-8")
    val.write_text("{}", encoding="utf-8")

    idx = build_coco_training_index(
        repo,
        bronze_coco_data_dir=coco,
        train_annotation=train,
        val_annotation=val,
        image_dir=coco,
    )
    d, ta, va, im = resolve_coco_for_train(idx, repo)
    assert d == coco.resolve()
    assert ta == train.resolve()
    assert va == val.resolve()
    assert im == coco.resolve()


def test_write_load_merge_video(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    p = repo / "gold" / "training_index.json"
    base = {
        "schema_version": "1.0",
        "coco": {
            "data_dir": "d",
            "train_annotation": "t.json",
            "val_annotation": "v.json",
            "test_annotation": None,
            "image_dir": "d",
        },
        "video": {},
    }
    write_training_index(p, base)
    loaded = load_training_index(p)
    m = merge_video_into_index(loaded, {"train_manifest": "silver/video/t.json", "manifest_root": "frames"})
    assert m["video"]["train_manifest"] == "silver/video/t.json"
    assert m["video"]["manifest_root"] == "frames"


def test_resolve_repo_path_absolute(tmp_path: Path) -> None:
    abs_p = tmp_path / "x"
    abs_p.mkdir()
    assert resolve_repo_path(tmp_path, str(abs_p.resolve())) == abs_p.resolve()
