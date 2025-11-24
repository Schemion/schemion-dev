import os
import uuid
from enum import Enum as DefaultEnum
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Enum, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from minio import Minio

DB_URL = "postgresql://admin:admin@localhost:5432/schemion"
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "models"
MODELS_DIR = "./models"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

if not minio_client.bucket_exists(MINIO_BUCKET):
    minio_client.make_bucket(MINIO_BUCKET)

Base = declarative_base()

class ModelStatus(str, DefaultEnum):
    pending = "pending"
    training = "training"
    completed = "completed"
    failed = "failed"

class Model(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    architecture = Column(String(50), nullable=False)
    architecture_profile = Column(String(50), nullable=False)
    minio_model_path = Column(String(512), nullable=False)
    status = Column(Enum(ModelStatus), default=ModelStatus.completed)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    is_system = Column(Boolean, default=False)
    base_model_id = Column(UUID(as_uuid=True), nullable=True)
    dataset_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())



def add_model_to_db(model_name, version, architecture, minio_path):
    db = SessionLocal()
    try:
        new_model = Model(
            name=model_name,
            version=version,
            architecture=architecture,
            architecture_profile="default",
            minio_model_path=minio_path,
            status=ModelStatus.completed,
            is_system=True
        )
        db.add(new_model)
        db.commit()
        print(f"add {model_name} to db")
        return new_model.id
    finally:
        db.close()


def upload_to_minio(file_path, model_name):
    file_ext = os.path.splitext(file_path)[1]
    new_filename = f"{uuid.uuid4()}_{model_name.replace(' ', '_')}{file_ext}"
    minio_path = f"system/{new_filename}"

    minio_client.fput_object(
        MINIO_BUCKET,
        minio_path,
        file_path,
        content_type="application/octet-stream"
    )

    print(f"file: {file_path} uploaded to MinIO: {minio_path}")
    return minio_path


if __name__ == "__main__":
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(('.pt', '.pth')):
            file_path = os.path.join(MODELS_DIR, filename)
            model_name = os.path.splitext(filename)[0]

            print(f"\nfile in process: {filename}")

            try:
                minio_path = upload_to_minio(file_path, model_name)

                if "yolo" in model_name.lower():
                    architecture = "yolo"
                elif "faster" in model_name.lower():
                    architecture = "faster_rcnn"
                else:
                    architecture = "unknown"

                add_model_to_db(
                    model_name=model_name,
                    version="1.0",
                    architecture=architecture,
                    minio_path=minio_path
                )

            except Exception as e:
                print(f"Error uploading {filename}: {str(e)}")

    print("\nSuccess")