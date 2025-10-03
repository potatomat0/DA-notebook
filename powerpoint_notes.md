 1. Trang bìa
      - Thông tin: Tên đề tài “Phân tích sống sót Titanic”, họ tên sinh viên, mã lớp/học phần, giảng viên hướng
  dẫn, thời gian thực hiện.
      - Hình: ảnh tàu Titanic hoặc icon dữ liệu.
      - Ghi chú: câu tagline ngắn “EDA → Feature Engineering → Modeling”.
  2. Mục lục
      - Bullet: Giới thiệu → Dữ liệu & chất lượng → Khám phá & trực quan → Feature Engineering → Mô hình hóa →
  Kết luận & Hướng phát triển.
      - Hình: icon timeline/phân đoạn.
      - Ghi chú: “Tổng quan các bước phân tích”.
  3. Giới thiệu bài toán
      - Nội dung: mục tiêu dự đoán xác suất sống sót, lý do chọn dataset, nguồn dữ liệu (Kaggle).
      - Hình: infographic tóm tắt 891 hành khách, 12 biến.

  - Ghi chú: “Bài toán phân loại nhị phân – sống sót/không sống sót”.

  4. Tổng quan dữ liệu & dữ liệu thiếu
      - Nội dung: liệt kê biến chính, số lượng bản ghi.
      - Bảng/hình: bảng nhỏ tóm tắt tỷ lệ thiếu (Age 19.9%, Cabin 77.1%, Embarked 0.2%).
      - Ghi chú: “Tác động của thiếu dữ liệu”.
  5. Xử lý thiếu dữ liệu
      - Nội dung: điền Embarked theo mode; dự đoán Age bằng median theo (Pclass, SibSp, Parch); Cabin -> deck
  + Unknown.
      - Hình: bảng missing_summary + sơ đồ minh họa cách ước lượng tuổi.
      - Ghi chú: “Giữ lại thông tin quan trọng thay vì bỏ dữ liệu”.
  6. Biểu đồ phân bố chính
      - Nội dung: highlight Age, Fare sau khi xử lý.
      - Biểu đồ: histogram Age, histogram/log Fare.
      - Ghi chú: “Age lệch trái, Fare lệch phải → cần bin/log khi đưa vào mô hình”.
  7. Tín hiệu sống sót theo biến phân loại
      - Nội dung: so sánh Sex, Pclass, Embarked, SibSp, Parch.
      - Biểu đồ: barplot Survival theo Sex, Pclass, bảng tỷ lệ từ groupby.
      - Ghi chú: “Nữ & hạng 1 có xác suất cao; gia đình vừa phải lợi thế”.
  8. Đặc trưng mới & insight
      - Nội dung: Title, Family size, IsAlone, Ticket prefix, CabinDeck.
      - Biểu đồ: barplot Survival by Title, IsAlone; bảng top ticket groups.
      - Ghi chú: “Rút đặc trưng giúp mô hình hiểu vị thế xã hội & cấu trúc gia đình”.
  9. Tương quan & mức độ ảnh hưởng
      - Nội dung: point-biserial cho số, Cramér’s V cho phân loại.
      - Biểu đồ: heatmap tương quan (Age–Pclass–SibSp–Parch), bảng ranking Cramér’s V.
      - Ghi chú: “Sex, Pclass, Title, Fare là biến mạnh”.
  10. Quy trình mô hình hóa
      - Nội dung: train/test split, baseline Logistic, so sánh Decision Tree & Random Forest, grid search +
  stratified K-fold.
      - Hình: sơ đồ pipeline hoặc flowchart.
      - Ghi chú: “Thiết lập baseline và tối ưu hyperparameter”.
  11. Kết quả mô hình
      - Nội dung: bảng/biểu đồ accuracy CV của 3 mô hình chính; kết quả test.
      - Biểu đồ: bar chart so sánh accuracy (Logistic vs Decision Tree vs Random Forest).
      - Ghi chú: “Random Forest (hoặc mô hình tốt nhất) đạt accuracy cao nhất”.
  12. Tổng kết & hướng phát triển
      - Nội dung: các phát hiện chính (vai trò của Sex, Pclass, gia đình, deck), mô hình khuyến nghị.
      - Hướng phát triển: thử thêm thang điểm bất cân đối, feature selection, trình bày trên dashboard.
      - Hình: icon checklist/kết luận.
  13. Phụ lục (tùy chọn nếu còn chỗ)
      - Nội dung: danh sách thư viện, môi trường chạy, link notebook GitHub/Kaggle.
      - Hình: mã QR hoặc biểu tượng Git.
      - Ghi chú: “Tài liệu tham khảo & tái hiện”.

