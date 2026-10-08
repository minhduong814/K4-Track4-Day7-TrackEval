# Báo cáo bài lab: cấu hình và đánh giá bộ theo dõi người


**Thành viên:**
1. Nguyễn Minh Dương - 2A202602920
2. Hoàng Trung Khải - 2A202602947

**Ngày thực hiện:** 09/10/2026

## Môi trường thực nghiệm

- Python 3.12.13; Ultralytics 8.4.158; PyTorch 2.14.1+cu130.
- GPU NVIDIA GeForce RTX 3050 Ti Laptop GPU, VRAM 4 GB.
- Detector YOLO26n (`yolo26n.pt`), `imgsz=640`, chỉ theo dõi lớp person (COCO class 0).
- Bộ đánh giá TrackEval; kết quả xuất theo định dạng MOTChallenge.

Dữ liệu gồm năm chuỗi, tổng cộng 4.137 frame. Chỉ `video_1` có ground truth; do đó chỉ chuỗi này được tính metric.

| Chuỗi | Số frame | Đặc điểm cảnh trong hướng dẫn | Ground truth |
|---|---:|---|---|
| `video_1` | 600 | Quảng trường, camera tĩnh, mật độ người vừa | Có |
| `video_2` | 1.050 | Góc camera cao, nhiều người, ánh sáng đèn mạnh | Không |
| `video_3` | 837 | Camera di chuyển giữa phố, người gần camera có thể che khuất | Không |
| `video_4` | 900 | Trung tâm mua sắm trong nhà, đông người, bề mặt phản chiếu | Không |
| `video_5` | 750 | Góc nhìn từ phương tiện, camera và xe cộ chuyển động | Không |

## CP1 — Xem cảnh và nêu giả thuyết

Chúng tôi kiểm tra các frame ở đầu, giữa và cuối từng chuỗi. Lệnh YOLO `predict` trên frame đầu của `video_1` phát hiện 7 bounding box người. Ở bước predict, box chưa có ID; ID được tạo khi chạy chế độ tracking.

### Giả thuyết trước khi so sánh

Camera tĩnh và mật độ người vừa, so sánh với thuật toán có Re-ID. Cảnh ít có chuyển động camera, BoT-SORT Re-ID có thể không cải thiện rõ metric so với ByteTrack

## CP2 — Chạy baseline ByteTrack trên `video_1`

Baseline dùng ByteTrack với `conf=0.30`, `iou=0.50` trên 150 frame đầu của `video_1`. Kết quả được đánh giá bằng ground truth tương ứng với 150 frame này.

| Tracker | Số frame | Conf | IoU | HOTA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack (baseline) | 150 | 0.30 | 0.50 | 32.23 | 16.93 | 30.17 | 4 |

ID 2 có detection ở cả 150 frame. ID 3 cũng xuất hiện từ đầu đến cuối đoạn baseline, với 146 detection trong 150 frame. Trong các frame xem trước, một số người ở xa trong nền nhỏ và khó nhìn thấy box ổn định. Điều này phù hợp với khả năng detector bỏ sót người nhỏ/xa.

## CP3 — So sánh tracker và tinh chỉnh tham số

### 3.1. So sánh tracker trên `video_1`

Metric được tính trên cùng 150 frame đầu. So sánh ở ngưỡng `conf=0.30`, `iou=0.50`:

| Tracker | HOTA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| ByteTrack | 32.23 | 16.93 | 30.17 | 4 |
| BoT-SORT Re-ID | 29.56 | 16.49 | 30.06 | 9 |

Trong lượt so sánh này, BoT-SORT Re-ID không đạt điểm cao hơn ByteTrack trên `video_1`.

### 3.2. Tinh chỉnh ngưỡng trên `video_1`

Các lượt dưới đây dùng ByteTrack. Mỗi lượt chỉ thay một ngưỡng so với cấu hình baseline.

| Conf | IoU | HOTA | MOTA | IDF1 | IDSW | Thay đổi so với baseline |
|---:|---:|---:|---:|---:|---:|---|
| 0.30 | 0.50 | 32.23 | 16.93 | 30.17 | 4 | Baseline |
| 0.15 | 0.50 | 33.80 | 17.84 | 33.26 | 5 | Thay conf |
| 0.50 | 0.50 | 26.97 | 13.76 | 24.47 | 1 | Thay conf |
| 0.30 | 0.40 | 32.19 | 17.21 | 30.49 | 2 | Thay IoU |
| 0.30 | 0.70 | 31.41 | 16.80 | 29.52 | 5 | Thay IoU |

Trong các cấu hình đã thử, `conf=0.15, iou=0.50` đạt HOTA, MOTA và IDF1 cao nhất trên đoạn này. `conf=0.50` có IDSW thấp nhất trong bảng nhưng các metric HOTA, MOTA và IDF1 thấp hơn. Với các mức IoU đã thử, số lượng dòng kết quả thay đổi rất ít trong bốn chuỗi không có ground truth.

### 3.3. Quan sát preview trên `video_2` đến `video_5`

Bảng ghi số dòng kết quả tracker trên 150 frame đầu cho mỗi cấu hình. Đây chỉ là số lượng detection/track được xuất, không phải metric chất lượng.

| Video | Tracker | Conf 0.15 / 0.30 / 0.50 (số dòng) | IoU 0.40 / 0.50 / 0.70 (số dòng) |
|---|---|---:|---:|
| `video_2` | ByteTrack | 1.785 / 1.581 / 1.122 | 1.581 / 1.581 / 1.582 |
| `video_3` | BoT-SORT Re-ID | 921 / 773 / 562 | 772 / 773 / 795 |
| `video_4` | BoT-SORT Re-ID | 1.071 / 978 / 782 | 978 / 978 / 978 |
| `video_5` | BoT-SORT Re-ID | 1.181 / 966 / 526 | 966 / 966 / 967 |

Khi xem preview, chúng tôi ghi nhận `conf=0.15` thêm nhiều box yếu, còn `conf=0.50` có thể bỏ người nhỏ hoặc ở xa. Do không có ground truth cho các video này, không thể dùng số dòng để kết luận cấu hình nào chính xác hơn. Cấu hình `conf=0.30, iou=0.50` được giữ cho các video này.

### 3.4. Cấu hình được chọn sau tinh chỉnh

| Video | Tracker | Conf | IoU | Cơ sở chọn cấu hình |
|---|---|---:|---:|---|
| `video_1` | ByteTrack | 0.15 | 0.50 | HOTA, MOTA và IDF1 cao nhất trong các lượt tinh chỉnh có nhãn |
| `video_2` | ByteTrack | 0.30 | 0.50 | Preview hai tracker tương tự; cảnh có góc nhìn cao và camera gần tĩnh |
| `video_3` | BoT-SORT Re-ID | 0.30 | 0.50 | Thử tracker có Re-ID trong cảnh camera di chuyển và có che khuất |
| `video_4` | BoT-SORT Re-ID | 0.30 | 0.50 | Thử tracker có Re-ID trong cảnh trong nhà đông người |
| `video_5` | BoT-SORT Re-ID | 0.30 | 0.50 | Thử tracker có Re-ID trong cảnh có chuyển động camera |

## Chạy kết quả cuối trên toàn bộ dữ liệu

Sau khi chọn cấu hình, chúng tôi chạy tracker trên toàn bộ frame của từng chuỗi. Năm file kết quả MOT được lưu trong `runs/nop_bai/`.

| Video | Tracker | Conf | IoU | Số frame | File kết quả |
|---|---|---:|---:|---:|---|
| `video_1` | ByteTrack | 0.15 | 0.50 | 600 | `runs/nop_bai/video_1.txt` |
| `video_2` | ByteTrack | 0.30 | 0.50 | 1.050 | `runs/nop_bai/video_2.txt` |
| `video_3` | BoT-SORT Re-ID | 0.30 | 0.50 | 837 | `runs/nop_bai/video_3.txt` |
| `video_4` | BoT-SORT Re-ID | 0.30 | 0.50 | 900 | `runs/nop_bai/video_4.txt` |
| `video_5` | BoT-SORT Re-ID | 0.30 | 0.50 | 750 | `runs/nop_bai/video_5.txt` |

### Đánh giá toàn bộ `video_1`

TrackEval chấm đủ 600 frame có ground truth:

| HOTA | MOTA | IDF1 | IDSW | Detection recall | Detection precision |
|---:|---:|---:|---:|---:|---:|
| 29.71 | 20.48 | 31.93 | 33 | 22.90% | 91.07% |
