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
    diabetes_assessment_level: Literal[
        "low",
        "moderate",
        "high",
        "known_diabetes",
        "discordant",
        "insufficient_data",
    ]
    diabetes_assessment: str
    diabetes_evidence: List[str] = Field(default_factory=list)
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
        "prior_screenings": [
            {
                "screening_date": prior.screening_date.isoformat(),
                "eyes": [eye.model_dump(mode="json") for eye in prior.eyes],
            }
            for prior in context.prior_screenings[:2]
        ],
    }


def _follow_up_text(assessment: ScreeningAssessment) -> str:
    return "; ".join(
        f"Tái khám {'mắt trái' if eye.eye == 'L' else 'mắt phải'}: {eye.follow_up_window}"
        for eye in _assessed_eyes(assessment)
    )


def _eye_label(eye: str) -> str:
    return "Mắt trái" if eye == "L" else "Mắt phải"


def _prior_screening_evidence(context: ClinicalContext) -> List[str]:
    evidence: List[str] = []
    for prior in context.prior_screenings[:2]:
        eye_summaries: List[str] = []
        for eye in prior.eyes:
            lesion_text = (
                f", tổn thương {', '.join(eye.detected_lesions)}"
                if eye.detected_lesions
                else ""
            )
            eye_summaries.append(
                f"{_eye_label(eye.eye).lower()} Grade {eye.dr_grade} – "
                f"{eye.dr_label}{lesion_text}"
            )
        if eye_summaries:
            evidence.append(
                f"Lần khám {prior.screening_date.date().isoformat()}: "
                + "; ".join(eye_summaries)
                + "."
            )
    return evidence


def _diabetes_support(
    context: ClinicalContext,
    assessment: ScreeningAssessment,
) -> tuple[str, str, List[str]]:
    """Assess diabetes context from the record; retinal AI is supporting evidence only."""

    evidence: List[str] = []
    diabetes_type = (context.diabetes_type or "").strip()
    hba1c = context.hba1c
    history_parts: List[str] = []

    positive_eyes = [
        eye for eye in _assessed_eyes(assessment) if eye.grading.dr_grade > 0
    ]

    if diabetes_type:
        level = "known_diabetes"
        history_parts.append(f"mắc đái tháo đường {diabetes_type}")
        if context.diabetes_duration_years is not None:
            history_parts.append(
                f"trong {context.diabetes_duration_years:g} năm"
            )
        evidence.append(f"Tiền sử trong hồ sơ: đái tháo đường {diabetes_type}.")
    elif hba1c is None:
        level = "insufficient_data"
    elif hba1c < 5.7:
        level = "discordant" if positive_eyes else "low"
    elif hba1c < 6.5:
        level = "high" if positive_eyes else "moderate"
    else:
        level = "high"

    if hba1c is not None:
        evidence.append(f"HbA1c gần nhất: {hba1c:g}%.")

    retinal_sentence: str
    if positive_eyes:
        grades = "; ".join(
            f"{_eye_label(eye.eye).lower()} Grade {eye.grading.dr_grade} – {eye.grading.dr_label}"
            for eye in positive_eyes
        )
        detected_lesions = _unique(
            [
                lesion.label
                for eye in positive_eyes
                for lesion in eye.segmentation.lesions
                if lesion.detected
            ]
        )
        lesion_text = (
            f", đồng thời ghi nhận {', '.join(detected_lesions)}"
            if detected_lesions
            else ""
        )
        retinal_sentence = (
            f"Kết quả ảnh võng mạc cho thấy {grades}{lesion_text}, "
            "là bằng chứng bổ sung về biến chứng võng mạc liên quan đến đái tháo đường."
        )
        evidence.append(
            "Bằng chứng bổ sung từ ảnh: "
            f"{grades}. Kết quả này phản ánh biến chứng võng mạc, không tự xác nhận đái tháo đường."
        )
    else:
        retinal_sentence = (
            "Ảnh võng mạc chưa ghi nhận DR (Grade 0), nhưng kết quả này không loại trừ "
            "đái tháo đường."
        )
        evidence.append(
            "Ảnh chưa ghi nhận DR (Grade 0); kết quả này không loại trừ đái tháo đường."
        )

    if diabetes_type:
        history = " ".join(history_parts)
        hba1c_sentence = (
            f"HbA1c gần nhất là {hba1c:g}%. " if hba1c is not None else ""
        )
        statement = (
            f"Hồ sơ ghi nhận người bệnh {history}. {hba1c_sentence}{retinal_sentence} "
            "Các dữ liệu cần được bác sĩ tổng hợp để đánh giá kiểm soát đường huyết và mức độ biến chứng."
        )
    elif hba1c is None:
        statement = (
            "Chưa có tiền sử hoặc HbA1c để đánh giá tình trạng đái tháo đường. "
            f"{retinal_sentence} Cần bổ sung xét nghiệm và bác sĩ xác nhận."
        )
    elif hba1c < 5.7:
        risk_text = (
            "Các dữ liệu đang không thống nhất"
            if positive_eyes
            else "Nguy cơ sàng lọc hiện ở mức thấp"
        )
        statement = (
            f"{risk_text}: HbA1c {hba1c:g}% thấp hơn ngưỡng tiền đái tháo đường. "
            f"{retinal_sentence} Cần bác sĩ đối chiếu vì một kết quả HbA1c đơn lẻ không loại trừ bệnh."
        )
    elif hba1c < 6.5:
        risk_text = (
            "Nguy cơ sàng lọc cao"
            if positive_eyes
            else "Nguy cơ sàng lọc trung bình"
        )
        statement = (
            f"{risk_text}: HbA1c {hba1c:g}% nằm trong khoảng tiền đái tháo đường. "
            f"{retinal_sentence} "
            "Kết quả tổng hợp cho thấy cần đánh giá thêm bằng xét nghiệm và thăm khám lâm sàng."
        )
    else:
        statement = (
            f"Nguy cơ sàng lọc cao: HbA1c {hba1c:g}% nằm trong ngưỡng đái tháo đường. "
            f"{retinal_sentence} "
            "Kết quả ảnh làm tăng bằng chứng về biến chứng võng mạc nhưng không thay thế xét nghiệm "
            "xác nhận và kết luận của bác sĩ."
        )

    prior_evidence = _prior_screening_evidence(context)
    evidence.extend(prior_evidence)
    if prior_evidence:
        statement += (
            " Diễn tiến võng mạc được đối chiếu với 1–2 lần khám trước và có thể dùng để gợi ý xu hướng "
            "kiểm soát đường huyết theo thời gian; đây là suy luận gián tiếp cần đối chiếu HbA1c hoặc glucose "
            "nối tiếp vì huyết áp, lipid, thời gian mắc bệnh và điều trị mắt cũng có thể ảnh hưởng."
        )
    else:
        statement += " Chưa có lần khám trước để đánh giá diễn tiến biến chứng võng mạc."

    return level, statement, evidence


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
    diabetes_level, diabetes_assessment, diabetes_evidence = _diabetes_support(
        context, assessment
    )
    diagnostic_impression, diagnostic_basis, diagnostic_limitations = _diagnostic_support(assessment)
    return ClinicalSummaryDraft(
        status=status,
        provider="local-rules",
        model=model,
        overview=(
            f"Kết quả sàng lọc {screening_scope} có mức ưu tiên rà soát '{assessment.overall_priority}'. "
            "Đây là bản tổng hợp tự động, chưa phải chẩn đoán cuối cùng."
        ),
        diabetes_assessment_level=diabetes_level,
        diabetes_assessment=diabetes_assessment,
        diabetes_evidence=diabetes_evidence,
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
        model: str = "gemini-3.6-flash",
        timeout_seconds: float = 45,
        max_output_tokens: int = 16384,
    ):
        self.api_key = api_key.strip()
        self.model = model.strip() or "gemini-3.6-flash"
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _prompt(payload: Dict[str, Any]) -> str:
        return (
            "Bạn là trợ lý soạn thảo báo cáo hỗ trợ đánh giá đái tháo đường và biến chứng võng mạc. "
            "Mục tiêu trung tâm là đánh giá hỗ trợ chẩn đoán tình trạng đái tháo đường từ tiền sử, "
            "loại đái tháo đường đã ghi nhận, thời gian mắc bệnh, HbA1c và dữ liệu lâm sàng được cung cấp. "
            "Nếu chưa có tiền sử mà HbA1c >= 6.5%, phải nêu rằng kết quả đạt ngưỡng xét nghiệm gợi ý "
            "đái tháo đường nhưng cần xét nghiệm xác nhận khi không có tăng đường huyết rõ ràng; không được "
            "tự ghi là đã chẩn đoán xác định. HbA1c 5.7–6.4% phải được mô tả là khoảng tiền đái tháo đường. "
            "HbA1c < 5.7% không được dùng để loại trừ bệnh nếu dữ liệu khác không thống nhất. "
            "Hãy đánh giá tình trạng đái tháo đường chủ yếu từ tiền sử, loại đái tháo đường, HbA1c và "
            "dữ liệu lâm sàng được cung cấp. Dùng DR Grade và tổn thương MA/HE/EX làm bằng chứng bổ sung "
            "về biến chứng võng mạc, không dùng chúng độc lập để xác nhận hoặc loại trừ đái tháo đường. "
            "Không được diễn giải Grade 0 thành không mắc đái tháo đường, không đưa ra tỷ lệ chắc chắn mắc "
            "bệnh nếu payload không chứa xác suất từ một mô hình đái tháo đường đã được thẩm định. "
            "Nếu HbA1c và ảnh võng mạc không thống nhất, phải nêu sự không thống nhất và khuyến nghị bác sĩ "
            "đánh giá hoặc xét nghiệm xác nhận. "
            "Phải phân loại diabetes_assessment_level đúng một trong các mã sau và không tạo mã khác: "
            "known_diabetes nếu hồ sơ đã ghi nhận loại đái tháo đường; insufficient_data nếu chưa có tiền sử "
            "và thiếu HbA1c; high nếu chưa có tiền sử nhưng HbA1c >= 6.5%; moderate nếu HbA1c từ 5.7% đến "
            "dưới 6.5% và ảnh không ghi nhận DR; high nếu HbA1c từ 5.7% đến dưới 6.5% đồng thời DR Grade > 0 "
            "hoặc có MA/HE/EX; low nếu HbA1c < 5.7% và ảnh Grade 0; discordant nếu HbA1c < 5.7% nhưng "
            "DR Grade > 0 hoặc có MA/HE/EX. Không hạ mức nguy cơ chỉ vì Grade 0. "
            "Khi không có tiền sử và mức là high, phải dùng cụm 'nguy cơ sàng lọc cao' hoặc 'nghi ngờ cao', "
            "không dùng cụm 'kiểm soát đường huyết kém'. Nếu chưa có loại bệnh trong hồ sơ, phải nói rõ chưa "
            "xác định được Type 1, Type 2 hay loại khác; không tự suy đoán loại bệnh. "
            "Trường diabetes_assessment phải là một đoạn văn liền mạch kết hợp tiền sử, thời gian mắc bệnh, "
            "HbA1c, DR Grade và tổn thương được phát hiện; không trình bày các nguồn này thành những kết luận "
            "rời rạc và không được nói DR Grade thuộc ngưỡng chẩn đoán đái tháo đường. "
            "Nếu prior_screenings có dữ liệu, bắt buộc so sánh lần hiện tại với 1–2 lần khám trước theo ngày, "
            "từng mắt, DR Grade và tổn thương để mô tả ổn định, tiến triển hoặc cải thiện. Nếu không có lịch sử, "
            "phải nói rõ chưa đủ dữ liệu đánh giá xu hướng. Được đưa ra nhận định xu hướng kiểm soát đường huyết ở mức "
            "gợi ý: DR tiến triển có thể phù hợp với phơi nhiễm tăng đường huyết kéo dài hoặc kiểm soát chưa tối ưu; "
            "DR ổn định/cải thiện có thể phù hợp với kiểm soát tốt hơn. Luôn ghi rõ đây là suy luận gián tiếp, không phải "
            "đo đường huyết; cần đối chiếu HbA1c/glucose nối tiếp và các yếu tố gây nhiễu như huyết áp, lipid, thời gian "
            "mắc bệnh, điều trị võng mạc hoặc cải thiện đường huyết nhanh. Đưa nhận xét dọc thời gian vào diabetes_assessment, "
            "diabetes_evidence và key_findings. "
            "Chỉ tổng hợp dữ liệu JSON được cung cấp; không tự đổi grade, confidence, mức ưu tiên, "
            "không kê đơn, không tự chỉ định điều trị và không khẳng định thay bác sĩ. "
            "Viết tiếng Việt ngắn gọn, dễ duyệt. Trả về đúng một JSON object gồm các khóa: "
            "overview (string), diabetes_assessment_level (string), diabetes_assessment (string), "
            "diabetes_evidence (array string), diagnostic_impression (string), diagnostic_basis (array string), "
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
                "responseMimeType": "application/json",
            },
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                trust_env=False,
            ) as client:
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
        except (
            httpx.HTTPError,
            httpx.InvalidURL,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            ValidationError,
        ):
            # A report-writing outage must not discard a valid local DR analysis.
            return _fallback_summary(context, assessment, self.model)
