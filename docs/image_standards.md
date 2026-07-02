# Tiêu Chuẩn Chất Lượng Ảnh Võng Mạc Và Quy Trình Kiểm Soát Đầu Vào

Để đảm bảo các mô hình học sâu (AI Grading & Segmentation) đạt độ chính xác tối ưu và giảm thiểu tỷ lệ chẩn đoán sai lệch, hệ thống thiết lập bộ tiêu chuẩn chất lượng ảnh chụp đáy mắt (Fundus Image) và quy trình kiểm soát chất lượng (Quality Control - QC) đầu vào.

---

## 1. Yêu Cầu Kỹ Thuật Đối Với Ảnh Đáy Mắt

Ảnh chụp võng mạc tải lên hệ thống cần đáp ứng các thông số kỹ thuật tối thiểu sau:

| Thông số | Yêu cầu tối thiểu | Khuyến nghị tối ưu |
| :--- | :--- | :--- |
| **Độ phân giải** | >= 1024 x 1024 pixels | >= 2048 x 2048 pixels |
| **Định dạng file** | JPEG (Lossless), PNG | PNG (16-bit) |
| **Trường quan sát (FOV)** | 45 độ | 50 độ |
| **Vùng trung tâm** | Đĩa thị (Optic Disc) và Hoàng điểm (Macula) phải nằm trọn vẹn trong ảnh. | Đĩa thị nằm cách biên trái/phải khoảng 1DD (Optic Disc Diameter). |
| **Hệ màu** | RGB màu đầy đủ (24-bit) | RGB màu đầy đủ (24-bit) |

---

## 2. Tiêu Chí Đánh Giá Chất Lượng Ảnh (Quality Control Criteria)

Trước khi thực hiện phân tích, hệ thống tự động hoặc Kỹ thuật viên (KTV) cần đánh giá ảnh theo 3 cấp độ chất lượng:

### Cấp độ 1: Tốt (Good)
* **Mô tả:** Ảnh sắc nét, độ tương phản cao. Mạch máu võng mạc nhìn rõ đến các nhánh nhỏ ngoại vi.
* **Hành động:** Chuyển thẳng sang AI phân tích.

### Cấp độ 2: Trung bình (Fair)
* **Mô tả:** Ảnh hơi mờ do đục thủy tinh thể nhẹ hoặc nhiễu sáng nhẹ, tuy nhiên đĩa thị và hoàng điểm vẫn có thể phân biệt được. Mạch máu lớn nhìn thấy rõ.
* **Hành động:** AI xử lý kèm cảnh báo độ tin cậy thấp hơn. Hệ thống tự động kích hoạt bộ tiền xử lý tăng cường độ tương phản (CLAHE).

### Cấp độ 3: Kém (Poor)
* **Mô tả:** Ảnh bị mờ hoàn toàn, mất nét, tối cục bộ hoặc chói sáng cực đại che khuất >30% diện tích võng mạc. Không phân biệt được đĩa thị hoặc hoàng điểm.
* **Hành động:** **TỪ CHỐI PHÂN TÍCH**. Yêu cầu KTV nhỏ thêm thuốc giãn đồng tử và chụp lại ảnh võng mạc cho bệnh nhân.

---

## 3. Quy Trình Chụp Ảnh Võng Mạc Chuẩn Lâm Sàng

KTV thực hiện chụp ảnh đáy mắt cần tuân thủ quy trình các bước sau để đảm bảo chất lượng hình ảnh đồng đều:

```
[BƯỚC 1: Chuẩn bị bệnh nhân]
  └── Thích nghi bóng tối 5 phút (hoặc nhỏ thuốc giãn đồng tử nếu có chỉ định của bác sĩ)
  └── Đặt cằm và trán bệnh nhân áp sát vào khung máy chụp ảnh đáy mắt

[BƯỚC 2: Căn chỉnh máy chụp]
  └── Chọn chế độ chụp Macula-centered (Hoàng điểm ở trung tâm) hoặc Disc-centered (Đĩa thị ở trung tâm)
  └── Điều chỉnh tiêu cự (focus) cho đến khi các mạch máu quanh hoàng điểm hiện rõ sắc nét nhất

[BƯỚC 3: Chụp ảnh và Kiểm tra tức thời]
  └── Chụp ảnh từng mắt (Mắt Phải trước - R, Mắt Trái sau - L)
  └── Kiểm tra nhanh ảnh trên màn hình: loại bỏ ảnh bị nhắm mắt, ảnh bị rung/nhòe do bệnh nhân cử động

[BƯỚC 4: Upload lên hệ thống]
  └── Nhập đúng mã định danh bệnh nhân (Patient Code)
  └── Đóng gói ảnh và gửi lên server để phân tích
```
---

## 4. Xử Lý Tiền Xử Lý Ảnh Chuyên Biệt Bằng Thuật Toán

Đối với các ảnh có chất lượng trung bình (Fair), Module Preprocessing của phân hệ AI sẽ tự động áp dụng các bước xử lý sau trước khi đưa vào mô hình học sâu:

1. **Trích xuất kênh màu xanh lá (Green Channel):** Lọc bỏ kênh Red (bị bão hòa ánh sáng đỏ của võng mạc) và kênh Blue (nhiều nhiễu), giữ lại kênh Green giúp các vi phình mạch (màu đỏ sẫm) nổi bật rõ rệt trên nền võng mạc.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Giúp nâng cao độ tương phản cục bộ một cách tự nhiên mà không làm phóng đại nhiễu ở các vùng biên ảnh võng mạc.
3. **Kỹ thuật Ben Graham:** Trừ đi ảnh được làm mờ Gaussian với bán kính lớn ($\sigma = 30$) khỏi ảnh gốc, sau đó scale độ sáng để triệt tiêu hiện tượng phân bố ánh sáng không đồng đều do độ cong của nhãn cầu.
