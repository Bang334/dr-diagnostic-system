from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageStat, UnidentifiedImageError

from app.clinical.models import ImageQuality


MIN_DIMENSION = 512


def sanitize_fundus_image(image_bytes: bytes) -> bytes:
    """Decode and re-encode pixels as PNG, removing EXIF/text metadata."""
    with Image.open(BytesIO(image_bytes)) as image:
        clean = image.convert("RGB")
        output = BytesIO()
        clean.save(output, format="PNG", optimize=True)
        return output.getvalue()


def assess_technical_quality(image_bytes: bytes) -> ImageQuality:
    """Reject corrupt/undersized files; leave clinical gradability to a human.

    Focus, field definition, illumination and media opacity require a validated
    fundus-quality model or trained reviewer. This function intentionally does
    not label an image "Good".
    """

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            gray = image.convert("L")
            stats = ImageStat.Stat(gray)
            mean = float(stats.mean[0])
            stddev = float(stats.stddev[0])
    except (UnidentifiedImageError, OSError, ValueError):
        return ImageQuality(
            status="Rejected",
            technically_valid=False,
            issues=["Tệp không phải ảnh PNG/JPEG hợp lệ hoặc ảnh đã hỏng."],
        )

    issues = []
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        issues.append(f"Độ phân giải phải tối thiểu {MIN_DIMENSION}x{MIN_DIMENSION} pixel.")
    if mean < 8 or mean > 247:
        issues.append("Ảnh có dấu hiệu thiếu sáng hoặc quá sáng nghiêm trọng.")
    if stddev < 5:
        issues.append("Ảnh có độ tương phản quá thấp để xử lý an toàn.")

    valid = not issues
    return ImageQuality(
        status="ReviewRequired" if valid else "Rejected",
        technically_valid=valid,
        requires_human_review=True,
        width=width,
        height=height,
        issues=issues or [
            "Cần người đọc được đào tạo xác nhận độ nét, chiếu sáng, trường ảnh và khả năng phân loại."
        ],
    )
