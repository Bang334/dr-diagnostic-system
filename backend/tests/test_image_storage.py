import unittest
from unittest.mock import patch

from app.services.image_storage import (
    CloudinaryImageStorage,
    ImageStorageError,
)


class CloudinaryImageStorageTests(unittest.TestCase):
    def _storage(self) -> CloudinaryImageStorage:
        return CloudinaryImageStorage(
            cloud_name="test-cloud",
            api_key="test-key",
            api_secret="test-secret",
            folder="screenings/fundus",
        )

    def test_requires_all_credentials(self):
        with self.assertRaises(ImageStorageError):
            CloudinaryImageStorage(
                cloud_name="test-cloud",
                api_key="",
                api_secret="",
                folder="screenings/fundus",
            )

    @patch("app.services.image_storage.cloudinary.uploader.upload")
    def test_upload_returns_https_url_and_public_id(self, upload):
        upload.return_value = {
            "secure_url": "https://res.cloudinary.com/test/image/upload/example.png",
            "public_id": "screenings/fundus/example",
        }

        stored = self._storage().upload_png(b"png-content", "left-fundus")

        self.assertEqual(
            stored.url,
            "https://res.cloudinary.com/test/image/upload/example.png",
        )
        self.assertEqual(stored.public_id, "screenings/fundus/example")
        self.assertEqual(upload.call_args.kwargs["resource_type"], "image")
        self.assertEqual(upload.call_args.kwargs["format"], "png")
        self.assertFalse(upload.call_args.kwargs["overwrite"])

    @patch("app.services.image_storage.cloudinary.uploader.upload")
    def test_rejects_upload_response_without_https_url(self, upload):
        upload.return_value = {
            "secure_url": "http://example.test/image.png",
            "public_id": "screenings/fundus/example",
        }

        with self.assertRaises(ImageStorageError):
            self._storage().upload_png(b"png-content", "left-fundus")

    @patch("app.services.image_storage.cloudinary.uploader.destroy")
    def test_delete_invalidates_cdn_asset(self, destroy):
        self._storage().delete("screenings/fundus/example")

        destroy.assert_called_once_with(
            "screenings/fundus/example",
            resource_type="image",
            invalidate=True,
        )


if __name__ == "__main__":
    unittest.main()
