"""Storage utilities for uploading files to S3."""

from pathlib import Path

import fsspec

# Use the same S3 bucket as black_box_evals
BBE_S3_BUCKET = "int8-shared-internal"


def upload_file_to_s3_fixed_path(local_path: Path, s3_dir: str, s3_subpath: str) -> None:
    """
    Upload a local file to S3 with an explicit S3 subpath.

    Prints a warning on failure but does not raise exceptions.

    Args:
        local_path: Full path to the local file to upload
        s3_dir: S3 directory path (relative to BBE_S3_BUCKET)
        s3_subpath: Subpath within s3_dir (e.g., "snapshots/iteration_0.pkl")
    """
    s3_full_path = f"s3://{BBE_S3_BUCKET}/{s3_dir}/{s3_subpath}"

    try:
        with open(local_path, "rb") as local_file:
            with fsspec.open(s3_full_path, "wb") as s3_file:
                s3_file.write(local_file.read())
        print(f"Uploaded to {s3_full_path}")
    except Exception as e:
        print(f"Warning: Failed to upload {s3_subpath} to S3: {e}")


def upload_bytes_to_s3(content: bytes, s3_dir: str, s3_subpath: str) -> None:
    """
    Upload bytes directly to S3 with an explicit S3 subpath.

    Prints a warning on failure but does not raise exceptions.

    Args:
        content: Bytes to upload
        s3_dir: S3 directory path (relative to BBE_S3_BUCKET)
        s3_subpath: Subpath within s3_dir (e.g., "snapshots/iteration_0.pkl")
    """
    s3_full_path = f"s3://{BBE_S3_BUCKET}/{s3_dir}/{s3_subpath}"

    try:
        with fsspec.open(s3_full_path, "wb") as s3_file:
            s3_file.write(content)
        print(f"Uploaded to {s3_full_path}")
    except Exception as e:
        print(f"Warning: Failed to upload {s3_subpath} to S3: {e}")
