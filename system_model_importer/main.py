import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import ARRAY, Boolean, Column, DateTime, String, Text, func, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker


DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin@localhost:5432/schemion")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_MODELS_BUCKET", "models")
MODELS_DIR = Path(os.getenv("SYSTEM_MODELS_DIR", "./models"))
MANIFEST_PATH = Path(os.getenv("SYSTEM_MODELS_MANIFEST", "models.json"))

SUPPORTED_ARCHITECTURES = {"yolo", "faster_rcnn"}
SUPPORTED_FASTERRCNN_PROFILES = {
    "resnet50_fpn",
    "resnet50_fpn_v2",
    "mobilenet_v3_large_fpn",
    "mobilenet_v3_large_320_fpn",
}

Base = declarative_base()


class Model(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    architecture = Column(String(50), nullable=False)
    architecture_profile = Column(String(512), nullable=False)
    classes = Column(ARRAY(Text), nullable=True)
    minio_model_path = Column(String(512), nullable=False)
    metrics_path = Column(String(512), nullable=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    is_system = Column(Boolean, default=False)
    base_model_id = Column(UUID(as_uuid=True), nullable=True)
    dataset_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def load_manifest(path: str | Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("System model manifest must be a JSON array")
    entries = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each system model manifest entry must be an object")
        entries.append(validate_manifest_entry(item))
    return entries


def validate_manifest_entry(entry: dict[str, Any]) -> dict[str, Any]:
    required = ("name", "file", "architecture", "architecture_profile")
    for field in required:
        if not str(entry.get(field) or "").strip():
            raise ValueError(f"System model manifest entry is missing '{field}'")

    architecture = str(entry["architecture"]).strip().lower()
    architecture_profile = str(entry["architecture_profile"]).strip()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if architecture == "faster_rcnn" and architecture_profile not in SUPPORTED_FASTERRCNN_PROFILES:
        raise ValueError(f"Unsupported Faster R-CNN profile: {architecture_profile}")

    classes = entry.get("classes")
    if classes is not None:
        if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
            raise ValueError("'classes' must be a list of strings when provided")

    return {
        "name": str(entry["name"]).strip(),
        "file": str(entry["file"]).strip(),
        "architecture": architecture,
        "architecture_profile": architecture_profile,
        "classes": classes,
    }


def build_db_session():
    engine = create_engine(DB_URL)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_local()


def build_minio_client():
    from minio import Minio

    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    if not client.bucket_exists(bucket_name=MINIO_BUCKET):
        client.make_bucket(bucket_name=MINIO_BUCKET)
    return client


def _safe_object_name(entry: dict[str, Any]) -> str:
    ext = Path(entry["file"]).suffix
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", entry["name"]).strip("_").lower()
    return f"system/{slug}{ext}"


def upload_to_minio(client, entry: dict[str, Any], models_dir: str | Path = MODELS_DIR) -> str:
    file_path = Path(models_dir) / entry["file"]
    if not file_path.is_file():
        raise FileNotFoundError(f"System model file not found: {file_path}")

    object_name = _safe_object_name(entry)
    client.fput_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        file_path=str(file_path),
        content_type="application/octet-stream",
    )
    print(f"file: {file_path} uploaded to MinIO: {object_name}")
    return object_name


def upsert_model_record(db, entry: dict[str, Any], minio_path: str) -> tuple[Any, bool]:
    existing = db.query(Model).filter(Model.name == entry["name"], Model.is_system.is_(True)).first()
    created = existing is None
    model = existing or Model(name=entry["name"], is_system=True)

    model.architecture = entry["architecture"]
    model.architecture_profile = entry["architecture_profile"]
    model.classes = entry.get("classes")
    model.minio_model_path = minio_path
    model.user_id = None
    model.is_system = True

    if created:
        db.add(model)
    db.commit()
    db.refresh(model)
    return model.id, created


def import_system_models(manifest_path: str | Path = MANIFEST_PATH, models_dir: str | Path = MODELS_DIR) -> None:
    entries = load_manifest(manifest_path)
    client = build_minio_client()
    db = build_db_session()
    try:
        for entry in entries:
            print(f"\nfile in process: {entry['file']}")
            minio_path = upload_to_minio(client, entry, models_dir)
            model_id, created = upsert_model_record(db, entry, minio_path)
            action = "created" if created else "updated"
            print(f"{action} system model {entry['name']} ({model_id})")
    finally:
        db.close()
    print("\nSuccess")


if __name__ == "__main__":
    import_system_models()
