# Hướng Dẫn Bài Lab: Cấu Hình & Đánh Giá Tracker

## 1. Tổng Quan Bài Lab

* **Thời lượng:** 2 giờ.
* **Hình thức:** Mỗi nhóm 2 học viên sử dụng 1 máy tính.
* **Nhiệm vụ:** Chọn cấu hình theo dõi người (object tracking) phù hợp cho 5 cảnh quay. Hệ thống đã tích hợp sẵn **YOLO detector** và **5 trackers**.
* **Mục tiêu:** Trả lời bằng kết quả quan sát: *Tracker nào giữ ID ổn định hơn trong từng cảnh, và vì sao?*
* **Sản phẩm nộp:** Cuối buổi, nộp 5 file kết quả (`video_1.txt` đến `video_5.txt`) cùng báo cáo hoàn chỉnh.

### Bảng tổng hợp các Video

| Video | Đặc điểm cần chú ý | Cách đánh giá trong Lab |
| :--- | :--- | :--- |
| `video_1` | Quảng trường ban ngày, camera tĩnh, mật độ vừa | **Có nhãn**; xem video và đọc điểm $HOTA$, $MOTA$, $IDF1$ |
| `video_2` | Phố đêm, camera tĩnh trên cao, rất đông | **Xem bằng mắt** |
| `video_3` | Camera di chuyển, ảnh nhỏ, ít frame/giây (FPS thấp) | **Xem bằng mắt** |
| `video_4` | Trong nhà, camera tiến tới, có phản chiếu kính | **Xem bằng mắt** |
| `video_5` | Góc nhìn trên xe bus, giao lộ đông, rung lắc | **Xem bằng mắt** |

> **Lưu ý:** Chỉ `video_1` có nhãn dữ liệu (ground truth). Với các video từ `video_2` đến `video_5`, học viên mô tả bằng quan sát thực tế, **không tự điền** các điểm $HOTA$, $MOTA$, $IDF1$.

### Gợi Ý Phân Bổ Thời Gian (120 phút)
* **15 phút:** Cài đặt và kiểm tra dữ liệu.
* **20 phút:** Ôn lại kiến thức metrics và chạy thử detector.
* **10 phút:** Chạy baseline.
* **35 phút:** Thử nghiệm & so sánh cấu hình các tracker.
* **30 phút:** Chấm điểm `video_1` và viết báo cáo.
* **10 phút:** Demo và thảo luận.

---

## 2. Thiết Lập & Kiểm Tra Môi Trường

Tải gói dữ liệu 5 video theo tập tin `README`, giải nén và thiết lập biến môi trường `LAB_DATA` trỏ đến thư mục chứa trực tiếp từ `video_1` đến `video_5`.

Thực hiện lệnh trong Terminal:

```bash
export LAB_DATA=/đường/dẫn/lab_data
python scripts/check_data.py --lab-data-root "$LAB_DATA"
```

* **Kết quả mong đợi:** Cả 5 thư mục video đều chứa ảnh đuôi `.jpg` trong `img1/`; riêng `video_1` báo có nhãn.
* **Xử lý lỗi:** Nếu báo thiếu ảnh, hãy kiểm tra lại đường dẫn và gói giải nén.
* **Khởi chạy Notebook:** Dùng kernel `cv_robotics_lab21` khi mở `on_tap_metrics.ipynb`. Luôn mở notebook từ cửa sổ terminal đã set biến `LAB_DATA`.

---

## 3. Kiến Thức Nền — Detector, Tracker và Các Chỉ Số

1. **Detector (YOLO):** Tìm các khung chứa người (bounding boxes) ở từng frame. Một hộp phát hiện ban đầu **chưa có ID**.
2. **Tracker:** Nối các hộp qua thời gian để cùng một đối tượng giữ nguyên một ID.
3. **Chỉ số $IoU$ (Intersection over Union):** Đo độ chồng lấp của hai hộp.
   * *Lưu ý trong Lab này:* Tham số `--iou` đại diện cho ngưỡng $IoU$ của thuật toán NMS trong **Detector**, không phải ngưỡng ghép ID của Tracker.
4. **Tham số `--conf`:** Ngưỡng tin cậy của YOLO.
   * Đặt quá cao: Dễ bỏ sót người ở khoảng cách xa, bị khuất hoặc tối.
   * Đặt quá thấp: Dễ sinh ra các hộp giả (false positives).
5. **Phân loại Tracker:**
   * **Chỉ dùng thông tin chuyển động:** `bytetrack`, `ocsort`.
   * **Dùng thêm đặc trưng ngoại hình (Re-ID):** `botsort`, `strongsort`, `deepocsort` (giúp giữ ID tốt hơn khi đối tượng bị che khuất).
6. **Ý nghĩa các Metric:**
   * **$MOTA$:** Bị ảnh hưởng nhiều bởi bỏ sót (FN), hộp giả (FP) và số lần đổi ID (IDSW).
   * **$IDF1$:** Phản ánh khả năng giữ đúng danh tính liên tục qua các frame.
   * **$HOTA$:** Cân bằng giữa chất lượng phát hiện (detection) và chất lượng liên kết (association).
7. **Cấu hình cố định của Lab:**
   * Model YOLO: `yolo26n.pt`
   * Kích thước ảnh: $640\text{ px}$
   * Trọng số Re-ID: `osnet_x0_25_msmt17.pt`
   * *Nhiệm vụ chính:* Học viên chỉ tinh chỉnh các tham số `--tracker`, `--conf`, và `--iou` cho từng video.

---

## 4. Các Bước Thực Hiện (Checkpoints)

### CP1: Quan Sát Cảnh và Nêu Giả Thuyết
1. Xem trước các đoạn phim tại `preview/video_1.mp4` … `preview/video_5.mp4` để nhận diện các thách thức (đông đúc, camera di chuyển, thiếu sáng, bị che khuất).
2. Mở `on_tap_metrics.ipynb`, đọc bảng chỉ số $MOTA$ / $IDF1$ / $HOTA$ và trả lời 3 câu hỏi True/False. Chạy phần YOLO trên 1 ảnh của `video_1` (xác nhận hộp phát hiện đã có nhưng chưa có ID).
3. Đưa ra dự đoán ngắn cho ít nhất 2 cảnh. 
   * *Ví dụ:* *"Cảnh đông dễ lẫn người khi họ cắt ngang nhau; tôi muốn so sánh ByteTrack với một tracker có Re-ID."*
* **Điều kiện hoàn thành CP1:** Kiểm tra xong dữ liệu, notebook chạy thành công và có giả thuyết ban đầu.

---

### CP2: Chạy Baseline Đầu Tiên Trên `video_1`
Chạy `ByteTrack` trên tối đa 150 frame đầu tiên để kiểm tra toàn bộ pipeline:

```bash
python scripts/run_tracking.py \
  --source "$LAB_DATA/video_1/img1" \
  --seq-name video_1 \
  --tracker bytetrack --conf 0.3 --iou 0.5 \
  --out runs/thu_nhanh --save-video --max-frames 150
```

* **Kiểm tra:** Mở file `runs/thu_nhanh/video_1_preview.mp4`. Quan sát màu sắc và ID của từng người. Nếu người đang di chuyển mà bị đổi ID hoặc đổi màu, đó là lỗi tráo danh tính (ID switch).
* **Lưu ý:** 
  * File `runs/thu_nhanh/video_1.txt` chỉ là bản thử nghiệm, chưa phải file nộp.
  * Nếu máy có GPU, bổ sung `--device cuda:0` để tăng tốc độ xử lý.
* **Điều kiện hoàn thành CP2:** Chỉ ra được ít nhất 1 đoạn ID giữ ổn định hoặc 1 lỗi nhận diện quan sát được từ video baseline.

---

### CP3: Thí Nghiệm Chính — Chọn Cấu Hình Cho 5 Video
Với từng video, thực hiện các bước sau:

1. **Thử nghiệm Tracker:** Thử ít nhất 1 tracker chuyển động (`bytetrack`/`ocsort`) và 1 tracker Re-ID (`botsort`/`strongsort`/`deepocsort`).
   * *Chú ý:* Dùng các thư mục `--out` riêng biệt để tránh ghi đè kết quả.
2. **Tinh chỉnh Tham số:** Sau khi chọn được tracker phù hợp, thử nghiệm điều chỉnh:
   * `--conf`: `0.15` / `0.3` / `0.5`
   * `--iou`: `0.4` / `0.5` / `0.7`
   * *Nguyên tắc:* Mỗi lượt thử **chỉ đổi 1 tham số** để đánh giá chính xác tác động. Ghi lại các cấu hình không hiệu quả và lý do.
3. **Xuất kết quả nộp bài:** Khi đã chốt cấu hình tối ưu cho từng video, chạy lại **toàn bộ frame** (bỏ tham số `--max-frames 150`) lưu vào thư mục `runs/nop_bai/`.

**Ví dụ lệnh chạy hoàn chỉnh cho `video_2`:**
```bash
python scripts/run_tracking.py \
  --source "$LAB_DATA/video_2/img2" \
  --seq-name video_2 \
  --tracker strongsort --conf 0.25 --iou 0.5 \
  --out runs/nop_bai --save-video
```

*(Lặp lại quy trình trên cho từ `video_1` đến `video_5` với tên sequence và tham số tương ứng).*

* **Điều kiện hoàn thành CP3:** Thư mục `runs/nop_bai/` chứa đủ 5 file text `video_1.txt` đến `video_5.txt` trùng khớp với các cấu hình được trình bày trong báo cáo.