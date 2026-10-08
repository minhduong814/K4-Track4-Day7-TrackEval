# Báo cáo bài lab: theo dõi người trên 5 cảnh

**Ngày chạy:** 2026-10-09  
**Môi trường:** Python 3.12.13, Ultralytics 8.4.158, PyTorch 2.14.1+cu130, RTX 3050 Ti Laptop GPU  
**Detector:** `yolo26n.pt`, kích thước ảnh 640 px, chỉ theo dõi lớp person (COCO class 0)

## 1. Kiểm tra dữ liệu và pipeline

`scripts/check_data.py` tìm thấy đủ 5 chuỗi: `video_1` có 600 frame, `video_2` 1.050, `video_3` 837, `video_4` 900 và `video_5` 750. Chỉ `video_1` có ground truth. YOLO phát hiện 7 bounding box người trong frame đầu tiên; lệnh `predict` không gán ID.

Đã chạy hai tracker trên 150 frame đầu của mỗi chuỗi để so sánh preview. Với `video_1`, đã đo metric trên 150 frame đầu và sau đó chạy lại cấu hình chọn trên đủ 600 frame.

## 2. Tinh chỉnh trên `video_1`

Các lượt dưới đây dùng ByteTrack, trừ dòng BoT-SORT Re-ID. Mỗi metric ở bảng tinh chỉnh được tính trên cùng 150 frame đầu; `IDSW` là số lần đổi ID.

| Tracker | Conf | IoU | HOTA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| ByteTrack | 0.30 | 0.50 | 32.23 | 16.93 | 30.17 | 4 |
| BoT-SORT Re-ID | 0.30 | 0.50 | 29.56 | 16.49 | 30.06 | 9 |
| ByteTrack | 0.15 | 0.50 | **33.80** | **17.84** | **33.26** | 5 |
| ByteTrack | 0.50 | 0.50 | 26.97 | 13.76 | 24.47 | 1 |
| ByteTrack | 0.30 | 0.40 | 32.19 | 17.21 | 30.49 | 2 |
| ByteTrack | 0.30 | 0.70 | 31.41 | 16.80 | 29.52 | 5 |

`conf=0.15, iou=0.50` cho HOTA, MOTA và IDF1 cao nhất trong các cấu hình thử trên đoạn này. Nó cũng tạo nhiều false positive hơn cấu hình `conf=0.30`, nên kết quả là lựa chọn theo metric tổng thể, không phải cấu hình ít false positive nhất. BoT-SORT Re-ID không cải thiện điểm trên đoạn `video_1` này.

Với bốn chuỗi không có ground truth, đã xem preview của các mức `conf` và `iou` trên 150 frame đầu. Bảng sau ghi số dòng track được xuất, chỉ để đối chiếu lượng detection trong preview; đây không phải điểm chất lượng.

| Video | Tracker | Conf 0.15 / 0.30 / 0.50 (dòng) | IoU 0.40 / 0.50 / 0.70 (dòng) |
|---|---|---:|---:|
| `video_2` | ByteTrack | 1785 / 1581 / 1122 | 1581 / 1581 / 1582 |
| `video_3` | BoT-SORT Re-ID | 921 / 773 / 562 | 772 / 773 / 795 |
| `video_4` | BoT-SORT Re-ID | 1071 / 978 / 782 | 978 / 978 / 978 |
| `video_5` | BoT-SORT Re-ID | 1181 / 966 / 526 | 966 / 966 / 967 |

Ở các preview này, `conf=0.15` thêm nhiều box yếu và `conf=0.50` bỏ nhiều người nhỏ hoặc xa. `iou` thay đổi rất ít số box. Vì không có ground truth để phân biệt false positive với người bị bỏ sót, giữ `conf=0.30, iou=0.50` cho các chuỗi này.

## 3. Cấu hình đã chọn và kết quả toàn chuỗi

| Video | Tracker | Conf | IoU | Lý do chọn |
|---|---|---:|---:|---|
| `video_1` | ByteTrack | 0.15 | 0.50 | Điểm HOTA/MOTA/IDF1 cao nhất trong lượt tinh chỉnh có nhãn |
| `video_2` | ByteTrack | 0.30 | 0.50 | Góc nhìn cao, camera gần như tĩnh; preview ByteTrack và Re-ID tương tự |
| `video_3` | BoT-SORT Re-ID | 0.30 | 0.50 | Camera thay đổi góc nhìn và người ở gần máy quay che khuất phần lớn khung hình |
| `video_4` | BoT-SORT Re-ID | 0.30 | 0.50 | Cảnh trong nhà có nhiều người đi cắt ngang; Re-ID là lựa chọn thử cho duy trì ID |
| `video_5` | BoT-SORT Re-ID | 0.30 | 0.50 | Góc nhìn từ phương tiện có chuyển động camera; người nhỏ và xe chiếm phần lớn khung hình |

Metric toàn bộ 600 frame của `video_1`:

| HOTA | MOTA | IDF1 | IDSW | Detection recall | Detection precision |
|---:|---:|---:|---:|---:|---:|
| 29.71 | 20.48 | 31.93 | 33 | 22.90% | 91.07% |

Recall thấp cho thấy YOLO bỏ sót nhiều người, nhất là người ở xa. Precision cao cho thấy phần lớn các detection được tạo ra khớp với người được chấm. Chỉ `video_1` có ground truth nên không tính điểm cho các video còn lại. Các lựa chọn cho `video_2` đến `video_5` dựa trên cảnh quan sát và preview 150 frame, không phải kết quả chấm ground truth.

## 4. Kết quả đầu ra

Năm file kết quả đầy đủ nằm trong `runs/nop_bai/`:

- `video_1.txt`
- `video_2.txt`
- `video_3.txt`
- `video_4.txt`
- `video_5.txt`

Video xem trước và file thử nghiệm được giữ cục bộ để đối chiếu; gói nộp chỉ gồm báo cáo và năm file kết quả cuối.
