# Đối chiếu model grading và pretrained backbone

## Kết luận từ mã nguồn project

- Model grading tích hợp mặc định dùng kiến trúc `vit_large_patch14_dinov2.lvd142m` và nạp checkpoint retina-specific `YukunZhou/RETFound_dinov2_meh`, sau đó fine-tune thành bộ phân loại 5 mức ICDR.
- Dữ liệu fine-tune là bộ Fundus gộp APTOS, DDR, IDRiD, EyePACS và Messidor; checkpoint được chọn bằng validation QWK rồi mới đánh giá test.
- Kết quả được báo cáo trong project: supervised Accuracy 83,91%, Macro-F1 0,8381, Balanced Accuracy 0,8444, QWK 0,9122; semi-supervised Accuracy 84,10%, Macro-F1 0,8429, Balanced Accuracy 0,8486, QWK 0,9144.
- Checkpoint và `test_metrics.json` không có trong workspace hiện tại, nên các con số trên là số liệu báo cáo, chưa được tái kiểm chứng trực tiếp từ artifact chạy.

## Pretrained model đã làm đề tài gì và đạt bao nhiêu?

Checkpoint chính xác mà project dùng là [RETFound DINOv2 MEH](https://huggingface.co/YukunZhou/RETFound_dinov2_meh). Model card mô tả đây là vision foundation model được pretrain tự giám sát bằng DINOv2 trên một phần dữ liệu AlzEye; model card không công bố một Accuracy downstream duy nhất.

Paper gốc được model card dẫn là [A foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x). Đề tài của paper là học biểu diễn tổng quát từ ảnh võng mạc không gán nhãn rồi thích nghi cho nhiều bài toán: chẩn đoán DR/glaucoma, tiên lượng AMD và dự đoán một số bệnh hệ thống. RETFound gốc được pretrain trên 1,6 triệu ảnh võng mạc. Sau fine-tune cho phân loại DR, paper báo AUROC 0,943 trên APTOS-2019, 0,822 trên IDRiD và 0,884 trên Messidor-2. Đây là AUROC sau fine-tune, không phải Accuracy của checkpoint pretrained và không được so trực tiếp với Accuracy 83,91%.

Backbone DINOv2 ViT-L/14 tổng quát mà kiến trúc kế thừa có ImageNet-1k k-NN Accuracy 83,5% và linear-probe Accuracy 86,3% theo [model card chính thức của DINOv2](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md). Đây là benchmark ảnh tự nhiên, chỉ dùng để trình bày nguồn gốc backbone, không phải benchmark DR.

## Mô hình gần bài toán để tham khảo

| Công trình | Thiết lập/kết quả công bố | Cách dùng để so sánh |
| --- | --- | --- |
| [DDR: Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening](https://www.sciencedirect.com/science/article/pii/S0020025519305377) | 13.673 ảnh từ 147 bệnh viện; Accuracy phân loại DR 0,8284 | Gần bài toán grading hơn benchmark ImageNet; kết quả supervised của project cao hơn 1,07 điểm %, nhưng không phải cùng split nên không được tuyên bố thắng trực tiếp. |
| [DeepDRiD challenge](https://www.sciencedirect.com/science/article/pii/S2666389922001040) | Weighted kappa grading của các đội nằm trong khoảng 0,82–0,93 | QWK 0,9122 của project nằm trong dải cạnh tranh, nhưng project phải đánh giá trên đúng DeepDRiD protocol mới có xếp hạng công bằng. |
| [Deep learning versus human graders](https://www.nature.com/articles/s41746-019-0099-8) | Trên 25.326 ảnh sàng lọc tại Thái Lan: QWK thuật toán 0,85, người chấm vùng 0,78 | Cho thấy QWK project là hứa hẹn; khác dữ liệu và chuẩn tham chiếu nên không chứng minh tương đương lâm sàng. |
| [DRMamba](https://doi.org/10.1007/s44163-026-01330-z) | DDR: Accuracy/QWK 0,847/0,851; APTOS-2019: 0,905/0,938 | Model gần bài toán 5 lớp, nhưng chênh lệch giữa hai dataset cho thấy không nên so số mà bỏ qua dataset/split. |

## Cách kết luận “độ chính xác có ổn không”

Kết quả hiện tại tốt ở mức đồ án/nghiên cứu thử nghiệm: Accuracy khoảng 84% và QWK trên 0,91, đồng thời gần hoặc nằm trong vùng kết quả của các công trình liên quan. Chưa đủ để kết luận sẵn sàng lâm sàng. Cần có checkpoint, manifest test, dự đoán từng ảnh, kiểm tra trùng bệnh nhân/nội dung, khoảng tin cậy 95%, kiểm định chênh lệch giữa supervised và semi-supervised, và external test trên một dataset giữ kín như DDR hoặc DeepDRiD.

Mức tăng semi-supervised chỉ là 0,19 điểm phần trăm (84,10% so với 83,91%). Không nên gọi là cải thiện có ý nghĩa nếu chưa làm kiểm định ghép cặp trên cùng ảnh, chẳng hạn McNemar, và báo khoảng tin cậy.
