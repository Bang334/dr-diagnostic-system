from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from uuid import uuid4

import cloudinary
import cloudinary.uploader


class ImageStorageError(RuntimeError):
    """Raised when an image cannot be stored or removed from cloud storage."""


@dataclass(frozen=True)
class StoredImage:
    url: str
    public_id: str


class CloudinaryImageStorage:
    """Store sanitized PNG images and return their HTTPS delivery URLs."""

    def __init__(
        self,
        *,
        cloud_name: str,
        api_key: str,
        api_secret: str,
        folder: str,
    ) -> None:
        if not all((cloud_name, api_key, api_secret)):
            raise ImageStorageError("Cloudinary credentials are not configured.")

        self._folder = folder.strip("/")
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,
        )

    def upload_png(self, content: bytes, prefix: str) -> StoredImage:
        if not content:
            raise ImageStorageError("Cannot upload an empty image.")

        public_id = f"{prefix}-{uuid4().hex}"
        try:
            response = cloudinary.uploader.upload(
                BytesIO(content),
                resource_type="image",
                folder=self._folder or None,
                public_id=public_id,
                format="png",
                overwrite=False,
            )
        except Exception as exc:
            raise ImageStorageError("Cloudinary image upload failed.") from exc

        secure_url = str(response.get("secure_url") or "")
        stored_public_id = str(response.get("public_id") or "")
        if not secure_url.startswith("https://") or not stored_public_id:
            raise ImageStorageError("Cloudinary returned an invalid upload response.")
        return StoredImage(url=secure_url, public_id=stored_public_id)

    def delete(self, public_id: str) -> None:
        if not public_id:
            return
        try:
            cloudinary.uploader.destroy(
                public_id,
                resource_type="image",
                invalidate=True,
            )
        except Exception as exc:
            raise ImageStorageError("Cloudinary image cleanup failed.") from exc
