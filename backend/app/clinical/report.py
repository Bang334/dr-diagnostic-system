from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict

from fpdf import FPDF


def _font_path() -> Path:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError("Không tìm thấy font Unicode để xuất báo cáo tiếng Việt.")


class _ClinicalPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Clinical", size=8)
        self.cell(0, 8, f"Trang {self.page_no()}", align="C")


def clinical_report_pdf(patient: Dict[str, Any], assessment: Dict[str, Any]) -> bytes:
    """Render the stored draft exactly; never recompute clinical rules in a report."""

    pdf = _ClinicalPDF()
    pdf.add_font("Clinical", "", str(_font_path()))
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def line(text: str, size: int = 10, gap: int = 5):
        pdf.set_font("Clinical", size=size)
        pdf.multi_cell(0, gap, str(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    line("BÁO CÁO HỖ TRỢ SÀNG LỌC VÕNG MẠC ĐÁI THÁO ĐƯỜNG", 14, 7)
    line("TRẠNG THÁI: DỰ THẢO – BẮT BUỘC BÁC SĨ XÁC NHẬN", 11, 6)
    line(f"Bệnh nhân: {patient.get('patient_code', '—')} – {patient.get('full_name', '—')}")
    line(f"Ưu tiên rà soát: {assessment.get('overall_priority', '—')}")

    for key, label in (("left_eye", "MẮT TRÁI"), ("right_eye", "MẮT PHẢI")):
        eye = assessment.get(key)
        if not eye:
            continue
        grading = eye["grading"]
        pdf.ln(3)
        line(label, 12, 6)
        line(f"Phân loại AI: Grade {grading['dr_grade']} – {grading['dr_label']} ({grading['confidence']:.1%})")
        line(f"Khoảng theo dõi tham khảo: {eye['follow_up_window']}")
        line(f"Chuyển tuyến: {eye['referral']}")
        line("Tổn thương do model phân đoạn ghi nhận:")
        lesions = eye["segmentation"].get("lesions", [])
        if not lesions:
            line("  • Chưa có tổn thương được trả về; không đồng nghĩa loại trừ bệnh.")
        for lesion in lesions:
            state = "có" if lesion.get("detected") else "không"
            line(f"  • {lesion.get('label', lesion.get('key'))}: {state}; area={lesion.get('area_pct', 0)}% (chỉ số kỹ thuật)")
        for action in eye.get("actions", []):
            line(f"  • {action}")

    summary = assessment.get("clinical_summary")
    if summary:
        pdf.ln(3)
        line("TÓM TẮT HỒ SƠ LÂM SÀNG – DỰ THẢO AI", 12, 6)
        line(summary.get("overview", ""))
        for finding in summary.get("key_findings", []):
            line(f"  • {finding}")
        line(f"Theo dõi tham khảo: {summary.get('follow_up', '—')}")
        line(
            f"Nguồn soạn thảo: {summary.get('provider', '—')} / {summary.get('model', '—')}",
            9,
            5,
        )

    pdf.ln(3)
    line("Nguồn áp dụng: " + ", ".join(assessment.get("guideline_ids", [])))
    line(assessment.get("disclaimer", ""), 9, 5)
    output = BytesIO()
    pdf.output(output)
    return output.getvalue()
