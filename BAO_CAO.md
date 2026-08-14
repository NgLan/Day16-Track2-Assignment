# Báo cáo Đánh giá Môi trường Cloud AI trên GCP (Lab 16)

## 1. Thông tin tổng quan
- **Môi trường:** Google Cloud Platform (GCP) Compute Engine
- **Cấu hình máy chủ:** `e2-medium` (2 vCPU, 4GB RAM)
- **Mô hình:** LightGBM Classifier (phát hiện gian lận thẻ tín dụng - Credit Card Fraud Detection)
- **Bộ dữ liệu:** Kaggle Credit Card Fraud Detection (284,807 giao dịch, 31 đặc trưng)

## 2. Bảng kết quả Benchmark

| Chỉ số / Metric | Kết quả thực tế |
|---|---|
| **Thời gian load data** | `2.6309 s` |
| **Thời gian training (100 trees)** | `9.5111 s` |
| **Best iteration** | `100` |
| **AUC-ROC** | `0.806111` |
| **Accuracy** | `0.998455` |
| **F1-Score** | `0.584906` |
| **Precision** | `0.543860` |
| **Recall** | `0.632653` |
| **Inference latency (1 row)** | `1.2385 ms` |
| **Inference throughput (1000 rows)** | `266,576.70 rows/s` |

## 3. Nhận xét và Đánh giá (Báo cáo ngắn)

> **Nhận xét kết quả:**
> Mô hình LightGBM được triển khai và huấn luyện thành công trên cấu hình CPU phổ thông `e2-medium` (2 vCPU, 4 GB RAM) trong môi trường VPC riêng trên GCP. Thời gian nạp dữ liệu hơn 284,000 mẫu chỉ mất khoảng 2.63 giây và thời gian huấn luyện hoàn chỉnh 100 cây quyết định diễn ra nhanh chóng trong 9.51 giây. Về hiệu năng dự đoán, mô hình đạt độ chính xác cao (Accuracy 99.85%) và chỉ số AUC-ROC đạt 0.8061 trên tập dữ liệu lệch pha nặng (imbalanced). Đặc biệt, tốc độ suy luận cực kỳ ấn tượng với độ trễ chỉ 1.24 ms cho mỗi giao dịch đơn lẻ và thông lượng xử lý đạt hơn 266,000 giao dịch/giây khi dự đoán theo batch. Kết quả này chứng minh cấu hình CPU nhỏ trên GCP hoàn toàn đáp ứng tốt cho các bài toán huấn luyện và phục vụ suy luận thời gian thực cho các mô hình Machine Learning truyền thống với chi phí tối ưu (~$0.033/giờ).
