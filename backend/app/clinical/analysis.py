from __future__ import annotations

import asyncio
from typing import Dict, Optional

from app.clinical.models import (
    ClinicalContext,
    EyeClinicalAssessment,
    EyeImageSet,
    GradingPort,
    ScreeningAssessment,
    SegmentationPort,
)
from app.clinical.quality import assess_technical_quality


LOW_CONFIDENCE_THRESHOLD = 0.70  # Operational threshold; validate per model/site.
PRIORITY_ORDER = {"routine": 0, "prompt": 1, "urgent": 2, "emergency": 3}

GUIDELINE_IDS = [
    "QD-2557-QD-BYT-2022",
    "QD-2558-QD-BYT-2022",
    "ICDR-Wilkinson-2003",
    "ADA-Standards-of-Care-2026",
]

DISCLAIMER = (
    "Kết quả là dự thảo hỗ trợ sàng lọc, không thay thế khám mắt toàn diện hoặc "
    "chẩn đoán của bác sĩ nhãn khoa. Mọi lịch hẹn và điều trị phải được bác sĩ xác nhận."
)


class InvalidFundusSet(ValueError):
    def __init__(self, quality: Dict[str, object]):
        super().__init__("Bộ ảnh không đạt kiểm tra kỹ thuật tối thiểu.")
        self.quality = quality


def _clinical_rule(eye: str, grading, segmentation, context: ClinicalContext):
    grade = grading.dr_grade
    safety_flags = []
    findings = [f"ICDR Grade {grade}: {grading.dr_label}."]
    actions = ["Bác sĩ xác nhận phân giai đoạn trên bộ ảnh và đối chiếu khám lâm sàng."]

    if grade == 4:
        priority = "urgent"
        follow_up = "Dưới 1 tháng"
        referral = "Chuyển chuyên khoa mắt tuyến tỉnh/trung ương trong vòng dưới 1 tháng."
    elif grade == 3:
        priority = "urgent"
        follow_up = "Không quá 3 tháng; ưu tiên sớm theo bệnh cảnh"
        referral = "Chuyển chuyên khoa mắt tuyến tỉnh/trung ương để đánh giá sớm."
    elif grade == 2:
        priority = "prompt"
        follow_up = "Khoảng 3–6 tháng, bác sĩ cá thể hóa"
        referral = "Theo dõi tại cơ sở đủ năng lực hoặc chuyển chuyên khoa khi giảm thị lực."
    elif grade == 1:
        priority = "routine"
        follow_up = "Khoảng 6–12 tháng, bác sĩ cá thể hóa"
        referral = "Theo dõi định kỳ; chuyển chuyên khoa nếu ảnh không phân loại được hoặc có triệu chứng."
    else:
        priority = "routine"
        follow_up = "Khoảng 12 tháng, có thể điều chỉnh theo nguy cơ"
        referral = "Tiếp tục sàng lọc định kỳ; không coi ảnh đáy mắt là thay thế khám mắt toàn diện."

    if grading.confidence < LOW_CONFIDENCE_THRESHOLD:
        priority = max((priority, "prompt"), key=PRIORITY_ORDER.get)
        safety_flags.append("low_ai_confidence")
        actions.append("Độ tin cậy AI thấp: bác sĩ đọc ảnh hoặc chụp lại; không dùng dự đoán để ra quyết định.")
    if grade >= 3:
        safety_flags.append("specialist_confirmation_required")
        findings.append(
            "Model tổn thương hiện không đủ để tự xác nhận quy tắc 4-2-1, tân mạch hoặc xuất huyết dịch kính."
        )
    if context.hba1c is not None and context.hba1c > 8:
        safety_flags.append("hba1c_above_operational_flag")
        actions.append("HbA1c trên 8% được gắn cờ vận hành; mục tiêu điều trị phải cá thể hóa, không cộng điểm nguy cơ.")

    actions.append("Kiểm soát đái tháo đường theo bác sĩ điều trị; không dùng risk score tự đặt.")
    return priority, follow_up, referral, findings, actions, safety_flags


class ClinicalAnalysisModule:
    """Deep module: one interface for quality, AI aggregation and safe rules."""

    def __init__(self, grading: GradingPort, segmentation: SegmentationPort):
        self._grading = grading
        self._segmentation = segmentation

    async def _analyze_eye(self, images: EyeImageSet, context: ClinicalContext) -> EyeClinicalAssessment:
        quality = {
            "fundus": assess_technical_quality(images.fundus_image),
        }
        if not all(item.technically_valid for item in quality.values()):
            raise InvalidFundusSet({images.eye: quality})

        grading, segmentation = await asyncio.gather(
            self._grading.predict(images.fundus_image, images.eye),
            self._segmentation.predict(images.fundus_image, images.eye),
        )
        rule = _clinical_rule(images.eye, grading, segmentation, context)
        return EyeClinicalAssessment(
            eye=images.eye,
            quality=quality,
            grading=grading,
            segmentation=segmentation,
            review_priority=rule[0],
            follow_up_window=rule[1],
            referral=rule[2],
            findings=rule[3],
            actions=rule[4],
            safety_flags=rule[5],
        )

    async def analyze(
        self,
        left: Optional[EyeImageSet],
        right: Optional[EyeImageSet],
        context: ClinicalContext,
    ) -> ScreeningAssessment:
        requested = [images for images in (left, right) if images is not None]
        if not requested:
            raise InvalidFundusSet({"eyes": "Cần ít nhất một ảnh mắt trái hoặc mắt phải."})
        analyzed = await asyncio.gather(
            *(self._analyze_eye(images, context) for images in requested)
        )
        by_eye = {item.eye: item for item in analyzed}
        left_result = by_eye.get("L")
        right_result = by_eye.get("R")
        overall = max(
            (item.review_priority for item in analyzed),
            key=PRIORITY_ORDER.get,
        )
        recommendation = (
            f"Ưu tiên rà soát: {overall}. Bác sĩ xác nhận riêng từng mắt; "
            "thực hiện mốc sớm hơn khi có triệu chứng, giảm thị lực hoặc ảnh không phân loại được."
        )
        return ScreeningAssessment(
            left_eye=left_result,
            right_eye=right_result,
            overall_priority=overall,
            clinical_recommendation=recommendation,
            guideline_ids=GUIDELINE_IDS,
            disclaimer=DISCLAIMER,
        )
