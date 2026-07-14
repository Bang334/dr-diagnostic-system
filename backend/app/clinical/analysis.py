from __future__ import annotations

import asyncio
from typing import Dict, Iterable

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
    "Kết quả là dự thảo hỗ trợ sàng lọc, không thay thế khám mắt giãn đồng tử, OCT hoặc "
    "chẩn đoán của bác sĩ nhãn khoa. Mọi lịch hẹn và điều trị phải được bác sĩ xác nhận."
)


class InvalidFundusSet(ValueError):
    def __init__(self, quality: Dict[str, object]):
        super().__init__("Bộ ảnh không đạt kiểm tra kỹ thuật tối thiểu.")
        self.quality = quality


def _macular_status(segmentation) -> tuple[str, list[str]]:
    if segmentation.center_involved_confirmed_by_oct is True:
        return "center_involved_dme_confirmed_by_oct", [
            "Có bằng chứng OCT do nguồn tích hợp cung cấp; bác sĩ phải xác nhận chẩn đoán và xử trí."
        ]
    if segmentation.retinal_thickening_confirmed is True:
        near_fovea = any(
            lesion.detected
            and lesion.distance_to_fovea_mm is not None
            and lesion.distance_to_fovea_mm < 1.0
            for lesion in segmentation.lesions
        )
        return (
            "suspected_center_involved_dme" if near_fovea else "suspected_non_center_dme",
            ["Cần bác sĩ/OCT xác nhận mức độ phù hoàng điểm."],
        )
    if any(l.detected and l.key == "hard_exudate" for l in segmentation.lesions):
        return "indeterminate_requires_macular_assessment", [
            "Có xuất tiết cứng; không thể kết luận DME nếu thiếu dày võng mạc, vị trí hố trung tâm hoặc OCT."
        ]
    return "not_assessed", ["Ảnh màu đơn thuần không loại trừ phù hoàng điểm."]


def _clinical_rule(eye: str, grading, segmentation, context: ClinicalContext):
    grade = grading.dr_grade
    visual_acuity = context.visual_acuity_for(eye)
    safety_flags = []
    findings = [f"ICDR Grade {grade}: {grading.dr_label}."]
    actions = ["Bác sĩ xác nhận phân giai đoạn trên bộ ảnh và đối chiếu khám lâm sàng."]

    macular_status, macular_notes = _macular_status(segmentation)
    findings.extend(macular_notes)

    if context.sudden_vision_loss:
        priority = "emergency"
        follow_up = "Đánh giá trực tiếp trong ngày"
        referral = "Chuyển khám mắt/cấp cứu ngay; không chờ kết quả AI."
        safety_flags.append("sudden_vision_loss")
    elif visual_acuity is not None and visual_acuity < 0.5:
        priority = "urgent"
        follow_up = "Đánh giá chuyên khoa sớm"
        referral = "Chuyển chuyên khoa mắt do thị lực < 5/10, kể cả khi ảnh không thấy DR."
        safety_flags.append("visual_acuity_below_5_10")
    elif grade == 4:
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
        referral = "Theo dõi tại cơ sở đủ năng lực hoặc chuyển chuyên khoa khi giảm thị lực/nghi DME."
    elif grade == 1:
        priority = "routine"
        follow_up = "Khoảng 6–12 tháng, bác sĩ cá thể hóa"
        referral = "Theo dõi định kỳ; chuyển chuyên khoa nếu ảnh không phân loại được hoặc có triệu chứng."
    else:
        priority = "routine"
        follow_up = "Khoảng 12 tháng, có thể điều chỉnh theo nguy cơ"
        referral = "Tiếp tục sàng lọc định kỳ; không coi ảnh đáy mắt là thay thế khám mắt toàn diện."

    if macular_status.startswith("center_involved") or macular_status.startswith("suspected_center"):
        priority = max((priority, "urgent"), key=PRIORITY_ORDER.get)
        actions.append("Bác sĩ nhãn khoa đánh giá hoàng điểm và OCT; không tự động chỉ định tiêm/laser.")
    elif "requires_macular_assessment" in macular_status:
        priority = max((priority, "prompt"), key=PRIORITY_ORDER.get)
        actions.append("Đo thị lực và đánh giá hoàng điểm/OCT nếu có chỉ định.")

    if grading.confidence < LOW_CONFIDENCE_THRESHOLD:
        priority = max((priority, "prompt"), key=PRIORITY_ORDER.get)
        safety_flags.append("low_ai_confidence")
        actions.append("Độ tin cậy AI thấp: bác sĩ đọc ảnh hoặc chụp lại; không dùng dự đoán để ra quyết định.")
    if grade >= 3:
        safety_flags.append("specialist_confirmation_required")
        findings.append(
            "Model tổn thương hiện không đủ để tự xác nhận quy tắc 4-2-1, tân mạch hoặc xuất huyết dịch kính."
        )
    if context.pregnant:
        priority = max((priority, "prompt"), key=PRIORITY_ORDER.get)
        safety_flags.append("pregnancy_requires_individual_follow_up")
        actions.append("Thai kỳ với ĐTĐ có từ trước cần lịch khám mắt cá thể hóa bởi bác sĩ.")
    if context.kidney_disease:
        priority = max((priority, "prompt"), key=PRIORITY_ORDER.get)
        safety_flags.append("kidney_disease_systemic_risk")
        actions.append("Bệnh thận là yếu tố nguy cơ toàn thân; phối hợp bác sĩ điều trị để cá thể hóa theo dõi.")
    if context.hba1c is not None and context.hba1c > 8:
        safety_flags.append("hba1c_above_operational_flag")
        actions.append("HbA1c trên 8% được gắn cờ vận hành; mục tiêu điều trị phải cá thể hóa, không cộng điểm nguy cơ.")
    if (
        context.systolic_bp is not None
        and context.diastolic_bp is not None
        and (context.systolic_bp >= 140 or context.diastolic_bp >= 90)
    ):
        safety_flags.append("blood_pressure_above_operational_flag")
        actions.append("Huyết áp từ 140/90 mmHg được gắn cờ để bác sĩ đánh giá; không tự thay đổi thuốc hoặc cộng risk score.")

    actions.append("Tối ưu đường huyết, huyết áp và lipid theo bác sĩ điều trị; không dùng risk score tự đặt.")
    return priority, follow_up, referral, macular_status, findings, actions, safety_flags


class ClinicalAnalysisModule:
    """Deep module: one interface for quality, AI aggregation and safe rules."""

    def __init__(self, grading: GradingPort, segmentation: SegmentationPort):
        self._grading = grading
        self._segmentation = segmentation

    async def _analyze_eye(self, images: EyeImageSet, context: ClinicalContext) -> EyeClinicalAssessment:
        quality = {
            "disc": assess_technical_quality(images.disc_image),
            "posterior_pole": assess_technical_quality(images.posterior_pole_image),
        }
        if not all(item.technically_valid for item in quality.values()):
            raise InvalidFundusSet({images.eye: quality})

        grading, segmentation = await asyncio.gather(
            self._grading.predict(images.posterior_pole_image, images.eye),
            self._segmentation.predict(images.posterior_pole_image, images.eye),
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
            macular_status=rule[3],
            findings=rule[4],
            actions=rule[5],
            safety_flags=rule[6],
        )

    async def analyze(
        self,
        left: EyeImageSet,
        right: EyeImageSet,
        context: ClinicalContext,
    ) -> ScreeningAssessment:
        left_result, right_result = await asyncio.gather(
            self._analyze_eye(left, context),
            self._analyze_eye(right, context),
        )
        overall = max(
            (left_result.review_priority, right_result.review_priority),
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
