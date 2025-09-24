from __future__ import annotations

import os
from typing import Optional

from google.cloud import secretmanager


def _resolve_project_id() -> Optional[str]:
    return (
        os.getenv("GCP_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
    )


def access_secret_value(secret_id: str, *, version: str = "latest", project_id: Optional[str] = None) -> Optional[str]:
    pid = project_id or _resolve_project_id()
    print(f"Project ID: {pid}")
    if not pid:
        return None
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{pid}/secrets/{secret_id}/versions/{version}"
    print(f"Accessing secret: {name}")
    try:
        response = client.access_secret_version(name=name)
        return response.payload.data.decode("utf-8") if response and response.payload else None
    except Exception:
        return None


def ensure_env_from_secret(env_key: str, secret_id: str, *, version: str = "latest", project_id: Optional[str] = None) -> Optional[str]:
    if os.getenv(env_key):
        return os.getenv(env_key)
    value = access_secret_value(secret_id=secret_id, version=version, project_id=project_id)
    print(f"Value for {env_key}: {value}")
    if value:
        os.environ[env_key] = value
    return value


