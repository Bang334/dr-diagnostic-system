from __future__ import annotations

import json
from typing import Any, Dict, List, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clinical.models import ClinicalContext, ScreeningAssessment


class ClinicalSummaryDraft(BaseModel):
    """Structured, reviewable draft returned by a report-writing provider."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["generated", "fallback"] = "generated"
    provider: str = "gemini"
    model: str
    overview: str
    diagnostic_impression: str
    diagnostic_basis: List[str] = Field(default_factory=list)
    diagnostic_limitations: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    follow_up: str
    safety_note: str


def _unique(values: List[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _assessed_eyes(assessment: ScreeningAssessment) -> list:
    return [eye for eye in (assessment.left_eye, assessment.right_eye) if eye is not None]


def _safe_clinical_payload(
    context: ClinicalContext,
    assessment: ScreeningAssessment,
) -> Dict[str, Any]:
    """Whitelist clinical fields; names, patient codes and contact data never leave the backend."""

    return {
        "patient": {
            "age_years": context.age_years,
            "gender": context.gender,
            "diabetes_type": context.diabetes_type,
            "diabetes_duration_years": context.diabetes_duration_years,
            "hba1c_percent": context.hba1c,
        },
        "screening": {
            "overall_priority": assessment.overall_priority,
            "eyes": [
                {
                    "eye": eye.eye,
                    "dr_grade": eye.grading.dr_grade,
                    "dr_label": eye.grading.dr_label,
                    "confidence": round(eye.grading.confidence, 4),
                    "image_quality": eye.quality["fundus"].status,
                    "lesion_segmentation_status": eye.segmentation.status,
                    "lesions": [
                        {
                            "label": lesion.label,
                            "detected": lesion.detected,
                            "area_pct": lesion.area_pct,
                        }
                        for lesion in eye.segmentation.lesions
                    ],
                    "review_priority": eye.review_priority,
                    "follow_up_window": eye.follow_up_window,
                    "referral": eye.referral,
                    "safety_flags": eye.safety_flags,
                }
                for eye in _assessed_eyes(assessment)
            ],
        },
    }


def _follow_up_text(assessment: ScreeningAssessment) -> str:
    return "; ".join(
        f"Tái khám {'mắt trái' if eye.eye == 'L' else 'mắt phải'}: {eye.follow_up_window}"
        for eye in _assessed_eyes(assessment)
    )


def _eye_label(eye: str) -> str:
    return "Mắt trái" if eye == "L" else "Mắt phải"


def _diagnostic_support(assessment: ScreeningAssessment) -> tuple[str, List[str], List[str]]:
    eyes = _assessed_eyes(assessment)
    positive = [eye for eye in eyes if eye.grading.dr_grade > 0]
    if positive:
        stages = "; ".join(
            f"{_eye_label(eye.eye).lower()} phù hợp với {eye.grading.dr_label} (Grade {eye.grading.dr_grade})"
            for eye in eyes
        )
        impression = (
            "Nhận định hỗ trợ chẩn đoán bệnh võng mạc đái tháo đường: "
            f"{stages}. Kết quả cần bác sĩ nhãn khoa xác nhận."
        )
    else:
        impression = (
            "Chưa thấy dấu hiệu bệnh võng mạc đái tháo đường trên các ảnh đã phân tích "
            "(Grade 0); kết quả này không loại trừ đái tháo đường hoặc tổn thương ngoài vùng ảnh."
        )

    basis: List[str] = []
    for eye in eyes:
        basis.append(
            f"{_eye_label(eye.eye)}: mô hình grading dự đoán Grade {eye.grading.dr_grade} – "
            f"{eye.grading.dr_label}, độ tin cậy {eye.grading.confidence:.1%}."
        )
        detected = [lesion.label for lesion in eye.segmentation.lesions if lesion.detected]
        if detected:
            basis.append(
                f"{_eye_label(eye.eye)}: mô hình phân đoạn ghi nhận " + ", ".join(detected) + "."
            )
        elif eye.segmentation.status == "not_available":
            basis.append(
                f"{_eye_label(eye.eye)}: chưa có kết quả phân đoạn tổn thương; "
                "nhận định hiện dựa chủ yếu trên mô hình grading."
            )
        else:
            basis.append(
                f"{_eye_label(eye.eye)}: phân đoạn không ghi nhận tổn thương trong các lớp đang hỗ trợ; "
                "điều này không loại trừ tổn thương ngoài khả năng mô hình."
            )

    limitations = [
        "Ảnh fundus và DR grade không chẩn đoán hoặc loại trừ đái tháo đường; "
        "khi nghi ngờ cần xét nghiệm glucose/HbA1c theo tiêu chuẩn chẩn đoán.",
        "Đây là nhận định hỗ trợ DR từ ảnh, không thay thế khám đáy mắt giãn đồng tử "
        "và chẩn đoán phân biệt của bác sĩ nhãn khoa.",
    ]
    if any(eye.grading.dr_grade >= 3 for eye in eyes):
        limitations.append(
            "Grade 3–4 cần bác sĩ kiểm chứng quy tắc 4-2-1, tân mạch và xuất huyết "
            "trước võng mạc/dịch kính; các lớp phân đoạn hiện tại chưa bao phủ đầy đủ các dấu hiệu này."
        )
    return impression, basis, limitations


def _rule_summary(
    context: ClinicalContext,
    assessment: ScreeningAssessment,
    *,
    status: Literal["generated", "fallback"],
    model: str,
) -> ClinicalSummaryDraft:
    eyes = _assessed_eyes(assessment)
    screening_scope = "một mắt" if len(eyes) == 1 else "hai mắt"
    findings = [
        f"Mắt {'trái' if eye.eye == 'L' else 'phải'}: Grade {eye.grading.dr_grade} – "
        f"{eye.grading.dr_label}, độ tin cậy {eye.grading.confidence:.1%}."
        for eye in eyes
    ]
    risk_factors: List[str] = []
    if context.diabetes_duration_years is not None:
        risk_factors.append(f"Thời gian mắc đái tháo đường: {context.diabetes_duration_years:g} năm.")
    if context.hba1c is not None:
        risk_factors.append(f"HbA1c gần nhất: {context.hba1c:g}%.")
    diagnostic_impression, diagnostic_basis, diagnostic_limitations = _diagnostic_support(assessment)
    return ClinicalSummaryDraft(
        status=status,
        provider="local-rules",
        model=model,
        overview=(
            f"Kết quả sàng lọc {screening_scope} có mức ưu tiên rà soát '{assessment.overall_priority}'. "
            "Đây là bản tổng hợp tự động, chưa phải chẩn đoán cuối cùng."
        ),
        diagnostic_impression=diagnostic_impression,
        diagnostic_basis=diagnostic_basis,
        diagnostic_limitations=diagnostic_limitations,
        key_findings=findings,
        risk_factors=risk_factors,
        recommended_actions=_unique(
            [action for eye in eyes for action in eye.actions]
        ),
        follow_up=_follow_up_text(assessment),
        safety_note=assessment.disclaimer,
    )


def build_rule_summary(
    context: ClinicalContext,
    assessment: ScreeningAssessment,
) -> ClinicalSummaryDraft:
    """Build the deterministic comparison shown alongside the Gemini draft."""

    return _rule_summary(
        context,
        assessment,
        status="generated",
        model="clinical-rules-v1",
    )


def _fallback_summary(
    context: ClinicalContext,
    assessment: ScreeningAssessment,
    model: str,
) -> ClinicalSummaryDraft:
    return _rule_summary(
        context,
        assessment,
        status="fallback",
        model=model,
    )


class GeminiClinicalSummaryAdapter:
    """Gemini is a report writer only; grading remains owned by the local DR model."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        timeout_seconds: float = 45,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ):
        self.api_key = api_key.strip()
        self.model = model.strip() or "gemini-3.1-flash-lite"
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _prompt(payload: Dict[str, Any]) -> str:
        return (
            "Bạn là trợ lý soạn thảo báo cáo sàng lọc bệnh võng mạc đái tháo đường. "
            "Bạn chỉ hỗ trợ chẩn đoán bệnh võng mạc đái tháo đường (DR) từ grade và bằng chứng tổn thương "
            "được cung cấp; không chẩn đoán đái tháo đường từ ảnh fundus và không được diễn giải Grade 0 "
            "thành không mắc đái tháo đường. "
            "Chỉ tổng hợp dữ liệu JSON được cung cấp; không tự đổi grade, confidence, mức ưu tiên, "
            "không chẩn đoán DME khi chưa có OCT, không kê đơn và không khẳng định thay bác sĩ. "
            "Viết tiếng Việt ngắn gọn, dễ duyệt. Trả về đúng một JSON object gồm các khóa: "
            "overview (string), diagnostic_impression (string), diagnostic_basis (array string), "
            "diagnostic_limitations (array string), key_findings (array string), risk_factors (array string), "
            "recommended_actions (array string), follow_up (string), safety_note (string). "
            "Trường follow_up bắt buộc ghi rõ thời gian tái khám cho từng mắt và phải sao chép nguyên văn "
            "giá trị follow_up_window tương ứng, theo dạng 'Tái khám mắt trái: ...; Tái khám mắt phải: ...'; "
            "không tự rút ngắn hoặc kéo dài thời gian do quy tắc hệ thống đã chỉ định. "
            "Nếu dữ liệu phân đoạn có status not_available, phải nói rõ chưa có kết quả phân đoạn, "
            "không được diễn giải là không có tổn thương. Dữ liệu đã loại bỏ định danh trực tiếp:\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    async def generate(
        self,
        context: ClinicalContext,
        assessment: ScreeningAssessment,
    ) -> ClinicalSummaryDraft:
        if not self.api_key:
            return _fallback_summary(context, assessment, self.model)

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        body = {
            "contents": [{"parts": [{"text": self._prompt(_safe_clinical_payload(context, assessment))}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                    json=body,
                )
                response.raise_for_status()
                data = response.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            generated = json.loads(raw_text)
            _, diagnostic_basis, diagnostic_limitations = _diagnostic_support(assessment)
            generated["diagnostic_basis"] = diagnostic_basis
            generated["diagnostic_limitations"] = diagnostic_limitations
            generated["follow_up"] = _follow_up_text(assessment)
            return ClinicalSummaryDraft(
                status="generated",
                provider="gemini",
                model=self.model,
                **generated,
            )
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, ValidationError):
            # A report-writing outage must not discard a valid local DR analysis.
            return _fallback_summary(context, assessment, self.model)
