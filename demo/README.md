# Demo: dự báo năng lượng pin UAV trước chuyến bay

Product demo cho đề tài RBL môn DTA301 (Nhóm 6): *Phân tích và dự đoán mức năng lượng pin của UAV giao hàng*.

Người dùng nhập kế hoạch bay (tải trọng, tốc độ hành trình, độ cao hành trình), ứng dụng trả về năng
lượng pin dự báo cho cả chuyến (Wh), khoảng dự báo, tỉ lệ so với dung lượng pin và số chuyến bay được
với một viên pin. Mô hình chỉ dùng tham số biết trước khi cất cánh.

## Chạy demo

**Cách nhanh (Windows):** bấm đúp `run_demo.bat`. Lần đầu máy sẽ tự tạo môi trường Python và cài thư
viện (cần Python 3.12 và Internet), các lần sau mở ngay. Trình duyệt mở tại http://localhost:8501.

**Cách thủ công:**

```bash
cd demo
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Trong demo có gì

| Tab | Nội dung | Trả lời |
|---|---|---|
| Dự báo | Năng lượng dự báo, khoảng dự báo, so với pin; đối chiếu 3 mô hình với các chuyến bay thực cùng cấu hình; hồ sơ công suất của một chuyến bay thực | Sản phẩm chính |
| What-if | Năng lượng thay đổi thế nào theo từng tham số; tốc độ ít tốn năng lượng nhất; bản đồ tốc độ × độ cao; hệ số hồi quy kèm 95% CI | RQ1 |
| So sánh mô hình | MAE, RMSE, R², MAPE của Linear Regression, Random Forest, Polynomial Ridge; kiểm định thống kê | RQ2 |
| Phân tích sai số | Sai số theo mức tải trọng, tốc độ, độ cao; phép thử với cấu hình chưa từng bay; các chuyến dự báo sai nhất | RQ3 |
| Dữ liệu & giới hạn | Tổng quan 181 chuyến bay, nguồn dữ liệu, phạm vi áp dụng | Limitations |

Mở thẳng một kịch bản bằng đường link, tiện khi thuyết trình:
`http://localhost:8501/?tab=2&payload=300&speed=7.5&altitude=60`
(`tab` nhận số 1 đến 5).

## Số liệu được tính thế nào

`build_artifacts.py` đánh giá mô hình và ghi kết quả vào `artifacts/` (CSV và JSON). Chuyến bay là đơn
vị phân tích ở mọi bước:

1. Giữ kín 20% chuyến bay (37 chuyến, seed 42) cho lần kiểm tra cuối; danh sách ở `artifacts/split_holdout.json`.
2. Cross-validation 5-fold lặp 10 lần trên 144 chuyến còn lại.
3. So sánh mô hình bằng corrected resampled t-test (Nadeau & Bengio, 2003) trên 50 fold, và bằng sai số
   từng chuyến trên tập giữ kín với 95% CI bootstrap theo chuyến bay (10,000 lần).
4. Hai phép thử thêm trên toàn bộ 181 chuyến: bỏ hẳn từng cấu hình tải × tốc độ × độ cao khỏi huấn
   luyện rồi dự báo cấu hình đó; và bỏ nguyên từng ngày bay (16 ngày) rồi dự báo các chuyến của ngày đó.

Mô hình dùng để dự báo trực tiếp trong app được huấn luyện lại trên toàn bộ 181 chuyến mỗi khi app khởi
động (khoảng 1 giây), nên không cần lưu file mô hình.

Chạy lại phần đánh giá sau khi đổi dữ liệu hoặc mô hình:

```bash
python build_artifacts.py
```

## File

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit |
| `energy_models.py` | Biến đầu vào, định nghĩa 3 mô hình, hàm nạp dữ liệu (dùng chung cho app và phần đánh giá) |
| `build_artifacts.py` | Đánh giá mô hình, tạo `artifacts/` |
| `artifacts/` | Kết quả đánh giá: `summary.json`, `cv_folds.csv`, `per_flight.csv`, `flight_profiles.csv`, `split_holdout.json` |
| `requirements.txt` | Phiên bản thư viện |
| `run_demo.bat` | Chạy demo bằng một cú bấm đúp trên Windows |

Dữ liệu đầu vào là `../data/processed/flights_full.csv` do `src/01_build_dataset.py` tạo ra.

## Giới hạn

- Chỉ đúng cho DJI Matrice 100, tuyến R1, trong phạm vi tải 0–500 g, tốc độ 4–12 m/s, độ cao 25–100 m.
- Chưa tính gió: bộ dữ liệu không có nguồn gió biết trước khi bay.
- Là công cụ tham khảo khi lập kế hoạch; dùng cho quyết định vận hành thật cần kiểm chứng thêm.

Dữ liệu: Rodrigues, T. A. et al. (2021). *Data Collected with Package Delivery Quadcopter Drone.*
Carnegie Mellon University, KiltHub. DOI 10.1184/R1/12683453 (CC BY 4.0).
# drone-energy
