# Drone Energy RBL (DTA301, Nhóm 6)

Đề tài: **Phân tích và dự đoán mức năng lượng pin của UAV giao hàng**
(Predicting Whole-Flight Battery Energy Consumption of a Package-Delivery Drone from Mission Parameters
Known Before Take-off).

Dữ liệu: Rodrigues, T. A. et al. (2021). *Data Collected with Package Delivery Quadcopter Drone.*
Carnegie Mellon University, KiltHub. DOI 10.1184/R1/12683453 (CC BY 4.0).

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `../data/raw/` | Dữ liệu gốc: `flights/*.csv` (209 bản ghi bay), `parameters.csv`, `README.txt` của tác giả |
| `data/processed/` | Dữ liệu đã xử lý: `flights_full.csv` (181 chuyến tuyến R1, mỗi chuyến một dòng), `samples.csv`, `cleaning_log.json` |
| `src/` | Pipeline phân tích ban đầu, chạy theo thứ tự `01` đến `06`. Bài toán năng lượng cả chuyến ở đây dùng 8 biến, gồm cả thống kê gió đo trong chuyến; có thêm bài toán công suất tức thời và phân cụm |
| `outputs/` | Hình và bảng do `src/` tạo ra |
| `demo/` | Product demo (Streamlit) và phần đánh giá mô hình theo thiết kế chỉ dùng 3 biến biết trước khi bay; xem `demo/README.md` |
| `docs/` | Kịch bản thuyết trình và ảnh chụp demo dự phòng |

Số liệu trong `demo/artifacts/` và trong `outputs/tables/` khác nhau vì hai nơi dùng bộ biến đầu vào khác
nhau (xem cột Nội dung ở trên).

## Chạy demo

Bấm đúp `demo/run_demo.bat` (Windows), hoặc xem hướng dẫn thủ công trong `demo/README.md`.

## Chạy lại pipeline

Cần Python 3.12 và các thư viện ghi trong `environment.txt` (thêm `seaborn`).

```bash
python src/01_build_dataset.py
python src/02_eda.py
python src/03_baseline.py
python src/04_experiment.py
python src/05_analysis.py
python src/06_repro_check.py
```

Mọi bước dùng seed cố định là 42.
