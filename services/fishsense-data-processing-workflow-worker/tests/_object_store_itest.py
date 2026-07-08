"""Shared S3 helpers for the data-worker integration tests.

The integration stack (deploy/compose.local.yml) runs real Garage as the
object store. These helpers build a boto3 client pointed at it (path-style
addressing, as Garage requires) and seed the worker's
`E4EFS_OBJECT_STORE__*` env so the activity under test resolves the same
bucket.

Defaults match the `garage` service + the fixed key imported by
`garage-init` in compose.local.yml (in-cluster hostname). CI sets
`E4EFS_OBJECT_STORE__ENDPOINT_URL` to `http://garage:3900` and aliases
`garage` → 127.0.0.1 on the runner (so the same endpoint string resolves
from both the host and the Label Studio container — see integration.yml).
"""

from __future__ import annotations

import os

import boto3
from botocore.config import Config

ENDPOINT_URL = os.environ.get(
    "E4EFS_OBJECT_STORE__ENDPOINT_URL", "http://garage:3900"
)
REGION = os.environ.get("E4EFS_OBJECT_STORE__REGION", "garage")
BUCKET = os.environ.get("E4EFS_OBJECT_STORE__BUCKET", "fishsense")
ACCESS_KEY = os.environ.get(
    "E4EFS_OBJECT_STORE__ACCESS_KEY", "GK31c2f3a4b5c6d7e8f9a0b1c2"
)
SECRET_KEY = os.environ.get(
    "E4EFS_OBJECT_STORE__SECRET_KEY",
    "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
)


def make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(
            signature_version="s3v4", s3={"addressing_style": "path"}
        ),
    )


def set_object_store_env(monkeypatch) -> None:
    monkeypatch.setenv("E4EFS_OBJECT_STORE__ENDPOINT_URL", ENDPOINT_URL)
    monkeypatch.setenv("E4EFS_OBJECT_STORE__REGION", REGION)
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", BUCKET)
    monkeypatch.setenv("E4EFS_OBJECT_STORE__ACCESS_KEY", ACCESS_KEY)
    monkeypatch.setenv("E4EFS_OBJECT_STORE__SECRET_KEY", SECRET_KEY)
