# Bản đồ chuyển `dr-screening` vào hệ thống chính

| Nguồn cũ | Đích/thay thế trong hệ thống chính | Quyết định |
| --- | --- | --- |
| `ai/semi_supervised/semi_supervised_training.py` | `ai/semi_supervised/semi_supervised_training.py` | Đã chuyển và sửa output checkpoint. |
| `ai/semi_supervised/few_shot_demo.py` | `ai/semi_supervised/few_shot_demo.py` | Đã chuyển, ghi rõ dữ liệu giả lập. |
| `docs/semi_supervised_research.md` | `docs/semi_supervised_research.md` | Đã chuyển và bỏ tuyên bố “tốt nhất” chưa có bằng chứng. |
| `docs/clinical_rules_traceability.md` | `docs/clinical_rules_traceability.md` | Viết lại theo module/rule mới. |
| `clinical/core/*`, `clinical/services/*` | `backend/app/clinical/*` | Đã thay thế bằng module tích hợp; không sao chép code legacy. |
| `clinical/api/*` | `backend/app/api/screenings.py`, `reviews.py` | Đã thay thế bằng API chính và database thật. |
| `clinical/tests/test_tv3.py` | `backend/tests/test_clinical_analysis.py` | Thay bằng test tại interface mới; không giữ test rule đã loại bỏ. |
| `clinical/mocks`, `fixtures` | Adapter in-memory trong test mới | Không chuyển vào production. |
| `docs/api_contract.md`, `system_flow.md` | Tài liệu cùng tên trong `docs/` | Đã cập nhật theo API bốn ảnh và rule an toàn. |
| `requirements*.txt` | `backend/requirements.txt`, `ai/semi_supervised/requirements-research.txt` | Đã tách runtime và nghiên cứu. |
| `YeuCau.md`, `TASKS_TV3.md` | `README.md`, `docs/tv3_integration_status.md` | Nội dung cần thiết đã tổng hợp; file checklist tự đánh dấu cũ không chuyển. |

Sau khi test/build độc lập đạt và không còn tham chiếu đường dẫn
`dr-screening`, folder legacy có thể được lưu trữ hoặc xóa. Việc xóa không đồng
nghĩa Semi-supervised/Few-shot đã được validation trên dữ liệu thật.
