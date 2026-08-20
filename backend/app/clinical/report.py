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
    line("TRẠNG THÁI: DỰ THẢO - BẮT BUỘC BÁC SĨ XÁC NHẬN", 11, 6)
    line(f"Bệnh nhân: {patient.get('patient_code', '-')} - {patient.get('full_name', '-')}")
    line(f"Ưu tiên rà soát: {assessment.get('overall_priority', '-')}")

    for key, label in (("left_eye", "MẮT TRÁI"), ("right_eye", "MẮT PHẢI")):
        eye = assessment.get(key)
        if not eye:
            continue
        grading = eye["grading"]
        pdf.ln(3)
        line(label, 12, 6)
        line(f"Phân loại AI: Grade {grading['dr_grade']} - {grading['dr_label']} ({grading['confidence']:.1%})")
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
        line("TÓM TẮT HỒ SƠ LÂM SÀNG - DỰ THẢO AI", 12, 6)
        line(summary.get("overview", ""))
        for finding in summary.get("key_findings", []):
            line(f"  • {finding}")
        line(f"Theo dõi tham khảo: {summary.get('follow_up', '-')}")
        line(
            f"Nguồn soạn thảo: {summary.get('provider', '-')} / {summary.get('model', '-')}",
            9,
            5,
        )

    pdf.ln(3)
    line("Nguồn áp dụng: " + ", ".join(assessment.get("guideline_ids", [])))
    line(assessment.get("disclaimer", ""), 9, 5)
    output = BytesIO()
    pdf.output(output)
    return output.getvalue()


def screening_report_pdf(report: Dict[str, Any]) -> bytes:
    """Render a report exclusively from screening data already stored in the database."""

    pdf = _ClinicalPDF()
    pdf.add_font("Clinical", "", str(_font_path()))
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def line(text: Any, size: int = 10, gap: int = 5) -> None:
        pdf.set_font("Clinical", size=size)
        pdf.multi_cell(0, gap, str(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    patient = report.get("patient") or {}
    reviewed = report.get("status") == "Reviewed"
    status_label = "ĐÃ ĐƯỢC BÁC SĨ XÁC NHẬN" if reviewed else "DỰ THẢO - CHƯA ĐƯỢC BÁC SĨ XÁC NHẬN"

    line("BÁO CÁO HỖ TRỢ SÀNG LỌC VÕNG MẠC ĐÁI THÁO ĐƯỜNG", 14, 7)
    line(f"TRẠNG THÁI: {status_label}", 11, 6)
    line(f"Mã lần khám: #{report.get('screening_id', '-')}")
    line(f"Ngày khám: {report.get('screening_date') or '-'}")
    line(
        f"Bệnh nhân: {patient.get('patient_code', '-')} - "
        f"{patient.get('full_name', '-')}"
    )
    line(f"Bác sĩ phụ trách: {report.get('doctor_name') or 'Chưa cập nhật'}")

    for eye in report.get("eyes") or []:
        eye_label = "MẮT TRÁI" if eye.get("eye") == "L" else "MẮT PHẢI"
        ai_result = eye.get("ai_result")
        doctor_review = eye.get("doctor_review")
        pdf.ln(3)
        line(eye_label, 12, 6)

        if ai_result:
            confidence = float(ai_result.get("confidence") or 0)
            line(
                "Nhận định hỗ trợ AI: "
                f"Grade {ai_result.get('dr_grade', '-')} - "
                f"{ai_result.get('dr_label', '-')} ({confidence:.1%})"
            )
            line(f"Phiên bản mô hình: {ai_result.get('model_version') or '-'}", 9, 5)
        else:
            line("Chưa có nhận định hỗ trợ AI.")

        lesions = eye.get("lesions") or []
        line("Tổn thương do mô hình phân đoạn ghi nhận:")
        if not lesions:
            line("  • Chưa có dữ liệu phân đoạn; không đồng nghĩa loại trừ bệnh.")
        for lesion in lesions:
            state = "có" if lesion.get("detected") else "không"
            area_pct = float(lesion.get("area_pct") or 0)
            line(f"  • {lesion.get('label', 'Tổn thương')}: {state}; area={area_pct:.4f}%")

        if doctor_review:
            agreement = "Đồng ý với AI" if doctor_review.get("is_agree_with_ai") else "Điều chỉnh kết quả AI"
            line(
                "Kết luận bác sĩ: "
                f"Grade {doctor_review.get('final_dr_grade', '-')} - "
                f"{doctor_review.get('final_dr_label', '-')}"
            )
            line(f"Đối chiếu AI: {agreement}")
            line(f"Ghi chú lâm sàng: {doctor_review.get('clinical_notes') or 'Không có'}")
        else:
            line("Kết luận bác sĩ: Chưa xác nhận.")

    recall = report.get("recall")
    if recall:
        risk_labels = {
            "Low": "Thấp",
            "Medium": "Trung bình",
            "High": "Cao",
            "Urgent": "Khẩn cấp",
        }
        recall_status_labels = {
            "Scheduled": "Đã lên lịch",
            "Completed": "Đã hoàn thành",
            "Overdue": "Quá hạn",
            "Cancelled": "Đã huỷ",
        }
        pdf.ln(3)
        line("KẾ HOẠCH TÁI KHÁM", 12, 6)
        line(f"Ngày tái khám: {recall.get('recall_date') or '-'}")
        risk = recall.get("risk_stratification")
        line(f"Phân tầng nguy cơ: {risk_labels.get(risk, risk or '-')}")
        line(f"Khuyến nghị: {recall.get('recommendation') or 'Chưa cập nhật'}")
        recall_status = recall.get("status")
        line(f"Trạng thái: {recall_status_labels.get(recall_status, recall_status or '-')}")

    pdf.ln(3)
    line(
        "Báo cáo này hỗ trợ sàng lọc bệnh võng mạc đái tháo đường và không thay thế "
        "chẩn đoán, thăm khám hoặc quyết định điều trị của bác sĩ chuyên khoa.",
        9,
        5,
    )
    output = BytesIO()
    pdf.output(output)
    return output.getvalue()
