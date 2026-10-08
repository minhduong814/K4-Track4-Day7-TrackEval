# Bài lab: cấu hình và đánh giá bộ theo dõi người

## Mục tiêu

Chạy YOLO26n cùng nhiều bộ theo dõi trên năm chuỗi ảnh, so sánh độ ổn định ID bằng quan sát và chọn cấu hình phù hợp cho từng cảnh.

Chỉ `video_1` có ground truth. Chỉ tính HOTA, MOTA và IDF1 cho chuỗi này. Với `video_2` đến `video_5`, ghi nhận xét từ video xem trước; không tự gán điểm metric.

| Chuỗi | Số frame | Đặc điểm cần quan sát |
|---|---:|---|
| `video_1` | 600 | Quảng trường, camera tĩnh, mật độ người vừa; có ground truth |
| `video_2` | 1050 | Góc camera cao, nhiều người, ánh sáng đèn mạnh |
| `video_3` | 837 | Camera di chuyển giữa phố, người ở gần máy quay có thể che khuất |
| `video_4` | 900 | Trung tâm mua sắm trong nhà, nhiều người và bề mặt phản chiếu |
| `video_5` | 750 | Góc nhìn từ phương tiện trên phố, có chuyển động camera và xe cộ |

## 1. Cài đặt và kiểm tra dữ liệu

Từ thư mục gốc của repo:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export LAB_DATA="$PWD/lab_data"
python scripts/check_data.py --lab-data-root "$LAB_DATA"
```

Lệnh kiểm tra cần tìm thấy `img1/` cho cả năm chuỗi và `video_1/gt/gt.txt`. Các ảnh trong repo đã được giải nén sẵn vào `lab_data/`. Nếu dùng bộ dữ liệu khác, đặt `LAB_DATA` tới thư mục chứa trực tiếp các thư mục `video_1` đến `video_5`.

## 2. CP1: xem cảnh và nêu giả thuyết

Mở một vài ảnh ở đầu, giữa và cuối mỗi thư mục `img1/`. Các ảnh được đánh số theo frame, ví dụ `lab_data/video_3/img1/000280.jpg`. Repo không có notebook hoặc video preview dựng sẵn; có thể xem trực tiếp các ảnh này.

Chạy YOLO trên một ảnh để xem bounding box trước khi tracker gán ID:

```bash
python -c 'import os; from ultralytics import YOLO; image=os.path.join(os.environ["LAB_DATA"], "video_1/img1/000001.jpg"); r=YOLO("yolo26n.pt").predict(image, imgsz=640, classes=[0], verbose=False)[0]; print("boxes:", r.boxes.xyxy.cpu().tolist()); print("IDs:", r.boxes.id)'
```

Kết quả ở bước predict không có ID. ID chỉ xuất hiện khi chạy `track`.

Ghi giả thuyết trước khi thử nghiệm. Ví dụ: cảnh có camera di chuyển có thể hưởng lợi từ BoT-SORT với bù chuyển động camera; cảnh đông người có thể cần Re-ID để giảm đổi ID khi người che khuất nhau.

## 3. CP2: chạy baseline ByteTrack trên `video_1`

```bash
python scripts/run_tracking.py \
  --source "$LAB_DATA/video_1/img1" \
  --seq-name video_1 \
  --tracker bytetrack --conf 0.3 --iou 0.5 \
  --out runs/thu_nhanh/bytetrack \
  --save-video --max-frames 150 --device 0
```

Lệnh tạo `runs/thu_nhanh/bytetrack/video_1.txt` và video xem trước `runs/thu_nhanh/bytetrack/video_1_preview.mp4`. Nếu VRAM không đủ, đổi thành `--device cpu`.

Mở video xem trước, chọn một vài người dễ theo dõi và ghi lại frame có ID ổn định hoặc ID bị đổi. Để tính điểm baseline trên đúng 150 frame đầu:

```bash
python scripts/evaluate_lab.py \
  --lab-data-root "$LAB_DATA" \
  --tracker-dir runs/thu_nhanh/bytetrack \
  --seq-length 150
```

## 4. CP3: so sánh tracker và tinh chỉnh

Script hỗ trợ ByteTrack, BoT-SORT, BoT-SORT có Re-ID, OC-SORT và Deep OC-SORT. Chạy ít nhất một tracker chuyển động và một tracker có Re-ID. Mỗi lần chạy cần một thư mục đầu ra riêng.

Ví dụ chạy ByteTrack và BoT-SORT có Re-ID trên cả năm chuỗi:

```bash
for i in 1 2 3 4 5; do
  seq="video_$i"
  python scripts/run_tracking.py \
    --source "$LAB_DATA/$seq/img1" --seq-name "$seq" \
    --tracker bytetrack --conf 0.3 --iou 0.5 \
    --out "runs/experiments/$seq/bytetrack" --save-video --max-frames 150 --device 0
  python scripts/run_tracking.py \
    --source "$LAB_DATA/$seq/img1" --seq-name "$seq" \
    --tracker botsort-reid --conf 0.3 --iou 0.5 \
    --out "runs/experiments/$seq/botsort-reid" --save-video --max-frames 150 --device 0
done
```

Sau khi so sánh video xem trước, chọn một tracker cho từng cảnh rồi đổi từng tham số một:

| Thử nghiệm | Giá trị |
|---|---|
| `--conf` | `0.15`, `0.3`, `0.5` |
| `--iou` | `0.4`, `0.5`, `0.7` |

`--iou` là ngưỡng NMS của detector. Mỗi lần thử, lưu vào thư mục riêng và ghi số ID đổi, người bị mất, phát hiện giả cùng các frame tiêu biểu.

## 5. Xuất kết quả cuối

Chạy lại mỗi chuỗi với toàn bộ frame, dùng tracker và ngưỡng đã chọn. Không truyền `--max-frames`. Thêm `--save-video` nếu cần video để xem lại.

Ví dụ cho `video_1`:

```bash
python scripts/run_tracking.py \
  --source "$LAB_DATA/video_1/img1" --seq-name video_1 \
  --tracker bytetrack --conf 0.15 --iou 0.5 \
  --out runs/nop_bai --device 0
```

Ví dụ trên chọn ByteTrack với `conf=0.15` sau khi so sánh metric trên 150 frame đầu. Lặp lại lệnh cho bốn chuỗi còn lại, thay `--source`, `--seq-name`, tracker và ngưỡng đã chọn; giữ chung `--out runs/nop_bai` để tạo đủ năm file `video_1.txt` đến `video_5.txt`.

Chấm điểm chuỗi có ground truth:

```bash
python scripts/evaluate_lab.py \
  --lab-data-root "$LAB_DATA" \
  --tracker-dir runs/nop_bai
```

Lặp lại cho bốn chuỗi còn lại. Các file `.txt` là kết quả tracker; không dùng chúng làm ground truth.

## 6. Mẫu báo cáo

| Video | Tracker/cấu hình | Quan sát ID và lỗi | Điểm (chỉ video_1) | Lý do chọn |
|---|---|---|---|---|
| `video_1` |  |  | HOTA: ; MOTA: ; IDF1: |  |
| `video_2` |  |  | Không có ground truth |  |
| `video_3` |  |  | Không có ground truth |  |
| `video_4` |  |  | Không có ground truth |  |
| `video_5` |  |  | Không có ground truth |  |

Kết luận ngắn: tracker nào phù hợp nhất cho từng cảnh, cấu hình nào tạo khác biệt rõ nhất, và metric video_1 có khớp với quan sát hay không?

Báo cáo đã hoàn thành cho dữ liệu hiện có: [LAB_REPORT.md](LAB_REPORT.md).
