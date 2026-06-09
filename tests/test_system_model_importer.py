from types import SimpleNamespace

import pytest

from system_model_importer.main import load_manifest, upsert_model_record, validate_manifest_entry


def test_load_manifest_reads_explicit_fasterrcnn_profile(tmp_path):
    manifest = tmp_path / "models.json"
    manifest.write_text(
        """
        [
          {
            "name": "Faster R-CNN ResNet50 FPN",
            "file": "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
            "architecture": "faster_rcnn",
            "architecture_profile": "resnet50_fpn",
            "classes": ["person"]
          }
        ]
        """,
        encoding="utf-8",
    )

    entries = load_manifest(manifest)

    assert entries == [
        {
            "name": "Faster R-CNN ResNet50 FPN",
            "file": "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
            "architecture": "faster_rcnn",
            "architecture_profile": "resnet50_fpn",
            "classes": ["person"],
        }
    ]


def test_validate_manifest_entry_rejects_unsupported_fasterrcnn_profile():
    with pytest.raises(ValueError, match="Unsupported Faster R-CNN profile"):
        validate_manifest_entry(
            {
                "name": "bad",
                "file": "bad.pth",
                "architecture": "faster_rcnn",
                "architecture_profile": "default",
            }
        )


class _FakeQuery:
    def __init__(self, existing):
        self.existing = existing

    def filter(self, *_args):
        return self

    def first(self):
        return self.existing


class _FakeSession:
    def __init__(self, existing=None):
        self.existing = existing
        self.added = []
        self.committed = False
        self.refreshed = None

    def query(self, *_args):
        return _FakeQuery(self.existing)

    def add(self, model):
        self.added.append(model)

    def commit(self):
        self.committed = True

    def refresh(self, model):
        self.refreshed = model


def test_upsert_model_record_updates_existing_system_model_without_duplicate():
    existing = SimpleNamespace(
        id="model-id",
        name="Faster R-CNN ResNet50 FPN",
        architecture="unknown",
        architecture_profile="default",
        classes=None,
        minio_model_path="system/old.pth",
        is_system=True,
    )
    session = _FakeSession(existing)

    model_id, created = upsert_model_record(
        session,
        {
            "name": "Faster R-CNN ResNet50 FPN",
            "file": "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
            "architecture": "faster_rcnn",
            "architecture_profile": "resnet50_fpn",
            "classes": ["person"],
        },
        "system/fasterrcnn_resnet50_fpn.pth",
    )

    assert model_id == "model-id"
    assert created is False
    assert session.added == []
    assert session.committed is True
    assert existing.architecture == "faster_rcnn"
    assert existing.architecture_profile == "resnet50_fpn"
    assert existing.classes == ["person"]
    assert existing.minio_model_path == "system/fasterrcnn_resnet50_fpn.pth"
