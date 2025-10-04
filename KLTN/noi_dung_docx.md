ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH
TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN

PHÂN TÍCH TỈ LỆ SỐNG SÓT TITANIC – PHÂN TÍCH THĂM DÒ VÀ XÂY DỰNG MÔ HÌNH

Nhóm 14
Sinh viên thực hiện:
| STT | Họ tên | MSSV | Ngành |
| --- | --- | --- | --- |
| 1 | Hoàng Minh Nhật | 24550031 | CNTT |
| 2 | Đoàn Chí Hưng | 24550014 | CNTT |

TP. HỒ CHÍ MINH – 05/2024

# GIỚI THIỆU
Phần nghiên cứu tập trung vào việc phân tích dữ liệu hành khách tàu Titanic nhằm xác định các yếu tố chi phối khả năng sống sót. Nhóm triển khai chu trình chuẩn của môn Phân tích Dữ liệu IE224: làm sạch, khám phá, xây dựng đặc trưng và mô hình hóa. Bộ công cụ chính gồm pandas cho tiền xử lý, seaborn/matplotlib để trực quan hóa và scikit-learn cho mô hình học máy. Bộ dữ liệu được tham khảo từ kho Kaggle [1]; nhóm không sử dụng lại đồ án trước đây và chủ động xây dựng quy trình xử lý riêng. Các bước chính bao gồm đánh giá chất lượng dữ liệu (đặc biệt là thiếu vắng ở Age và Cabin), trích xuất đặc trưng mở rộng (Title, Family Size, Deck, Ticket Group) và thử nghiệm ba thuật toán giám sát (Logistic Regression, Decision Tree, Random Forest) trước khi ensemble. Kết quả hiện tại cho thấy độ chính xác trên tập kiểm tra đạt 83.16% với Logistic Regression baseline và 83.51% với mô hình VotingClassifier sử dụng các mô hình tốt nhất sau tinh chỉnh. Báo cáo cam kết minh bạch nguồn dữ liệu, nêu rõ các công cụ áp dụng và các phát hiện chính đạt được sau giai đoạn EDA.

# MÔ TẢ BỘ DỮ LIỆU (BẮT BUỘC)
Bộ dữ liệu Titanic gồm 891 dòng và 12 thuộc tính mô tả thông tin nhận dạng, nhân khẩu học và vé của hành khách. Trong đó có 3 biến số (`Age`, `Fare`, `PassengerId`) và 9 biến phân loại/định danh (`Survived`, `Pclass`, `Name`, `Sex`, `SibSp`, `Parch`, `Ticket`, `Cabin`, `Embarked`). Bộ dữ liệu được tham khảo tại Kaggle [1]; quá trình xử lý trong notebook đọc bản sao trên GitHub với cấu trúc giữ nguyên so với nguồn gốc. Tóm tắt thống kê ban đầu: tuổi trung bình 29.7 (độ lệch chuẩn 14.5), giá vé trung bình 32.2 bảng với giá trị lớn nhất 512.33. Tỷ lệ sống sót chung là 38.38%.

Tình trạng thiếu dữ liệu nổi bật:
| Biến | Số giá trị thiếu | Tỷ lệ | Ghi chú |
| --- | --- | --- | --- |
| Age | 177 | 19.9% | Ảnh hưởng trực tiếp đến phân tầng độ tuổi và cần ước lượng lại theo nhóm `Pclass`, `SibSp`, `Parch`. |
| Cabin | 687 | 77.1% | Dữ liệu cabin rất rời rạc; giải pháp là trích deck từ ký tự đầu và gán `Unknown` khi khuyết. |
| Embarked | 2 | 0.2% | Thiếu không đáng kể; có thể điền giá trị phổ biến `S`. |

Các biến khác đầy đủ giá trị. Bộ dữ liệu không phát hiện bản ghi trùng `PassengerId`. Nhóm sử dụng mô tả ý nghĩa từng biến như trong notebook để đảm bảo người đọc hiểu rõ phạm vi thông tin trước khi phân tích.

# PHƯƠNG PHÁP PHÂN TÍCH
Nhóm tuân theo quy trình gồm ba giai đoạn chính, được thể hiện lại dưới dạng pipeline: kiểm tra dữ liệu → tạo đặc trưng → mô hình hóa và đánh giá. Hình minh họa sẽ được bổ sung trong phiên bản Word.
[Infographic quy trình: Thu thập → Làm sạch → EDA → Feature Engineering → Modeling → Đánh giá]

## 3.1 Kiểm tra và tiền xử lý
- Đánh giá cấu trúc dữ liệu, kiểu biến và tỉ lệ khuyết dựa trên `info()` và thống kê mô tả.
- Điền `Embarked` bằng mode (`S`), ước lượng `Age` theo trung vị của các nhóm phân tầng bởi `Pclass`, `SibSp`, `Parch`, chuyển `Cabin` thành deck và gán `Unknown` cho phần thiếu.
- Chuẩn hóa định dạng vé: lấy prefix chữ cái và mã nhóm để dễ gom cụm.
- Tạo các cờ nhị phân như `IsAlone` (từ `SibSp + Parch`) và `FamilySize`.

## 3.2 Trực quan hóa thăm dò
- Sử dụng bar chart, histogram và boxplot để khám phá phân phối biến số và phân loại.
- Quan sát tương quan giữa biến đầu vào với `Survived` thông qua biểu đồ tỷ lệ sống sót theo từng nhóm.
- Ghi chú những điểm bất thường (ví dụ: `Fare` lệch phải mạnh, `Age` tập trung ở 18–35).

## 3.3 Xây dựng đặc trưng và mã hóa
- One-hot encoding cho `Sex`, `Embarked`, `Pclass` và các nhóm vé.
- Trích `Title` từ họ tên và chuẩn hóa về 4 nhóm chính nhằm phản ánh địa vị xã hội.
- Chuẩn hóa `Fare` bằng log hoặc phân vị (áp dụng trong notebook thông qua các cột mới).
- Xử lý các biến dạng số để mô hình tuyến tính tiếp nhận dễ dàng.

## 3.4 Mô hình hóa và đánh giá
- Chia dữ liệu sau xử lý thành 67% train và 33% test với `train_test_split`.
- Huấn luyện Logistic Regression làm baseline; sau đó tinh chỉnh Decision Tree, Random Forest và Logistic Regression bằng GridSearchCV (Stratified K-Fold 10 lần).
- Tổng hợp kết quả cross-validation, chọn top estimator cho mỗi mô hình.
- Kết hợp ba mô hình tốt nhất (Decision Tree, Random Forest, Logistic Regression) bằng VotingClassifier (soft voting) và đo lường accuracy trên tập test.

# PHÂN TÍCH THĂM DÒ/SƠ BỘ
Các trực quan hóa chính:
- [Biểu đồ cột: Tần suất `Survived`, `Sex`, `Pclass`, `Embarked`, `Parch`]
- [Histogram/KDE: Phân phối `Age`, `Fare`]
- [Biểu đồ tỷ lệ sống sót: `Pclass` vs `Survived`, `Sex` vs `Survived`, `SibSp`/`Parch` vs `Survived`]
- [Heatmap hoặc biểu đồ cột: Tỷ lệ sống sót theo `Embarked` kết hợp giới tính]

Nhận định tiêu biểu rút ra từ notebook:
- Tỷ lệ tử vong cao (62.29%) cho thấy dữ liệu hơi mất cân bằng về lớp.
- 64.75% hành khách là nam nhưng nhóm nữ đạt tỷ lệ sống sót 74% so với 19% ở nam.
- Hạng vé 1 có 63% sống sót, hạng 3 chỉ 24%, khẳng định vai trò của địa vị kinh tế.
- Hành khách đi một mình (`Parch=0`, `SibSp=0`) chiếm đa số nhưng có xác suất sống sót thấp hơn các nhóm gia đình nhỏ (1–2 người đi kèm).
- `Fare` tương quan thuận với sống sót; cần xử lý ngoại lệ khi đưa vào mô hình.
- Trẻ em dưới 10 tuổi được ưu tiên cứu hộ; nhóm tuổi 20–35 chịu thiệt hại nặng nhất.

# KẾT QUẢ PHÂN TÍCH
Bảng tổng hợp các kết quả chính:
| Mô hình | Cấu hình nổi bật | Mean CV Accuracy | Độ chính xác trên tập test |
| --- | --- | --- | --- |
| Logistic Regression (tuned) | Penalty {l1, l2}, C từ 1e-3 đến 1e3, max_iter 200–500 | 0.8153 | 83.16% (baseline) |
| Decision Tree (tuned) | `max_depth` 1–19, `min_samples_split` 10–490 | 0.8322 | (chưa ghi nhận riêng) |
| Random Forest (tuned) | `n_estimators` 100–300, `max_features` {1,3,10} | 0.8407 | (chưa ghi nhận riêng) |
| VotingClassifier (soft) | Kết hợp DT + RF + LR tốt nhất | - | 83.51% |

Ghi chú:
- Các giá trị cross-validation lấy từ kết quả GridSearchCV ghi nhận trong notebook.
- Mô hình ensemble mang lại cải thiện nhẹ so với Logistic Regression nhưng ổn định hơn so với từng mô hình lẻ.
- Chỉ số sử dụng là accuracy; cần bổ sung F1 hoặc ROC-AUC khi mở rộng.

# CHỈNH SỬA SAU BÁO CÁO
- Bổ sung đánh giá độ lệch lớp bằng confusion matrix và F1-score để phản ánh tốt hơn chi phí sai lầm.
- Kiểm tra lại ảnh hưởng của chuẩn hóa `Fare` (log vs quantile) nhằm đảm bảo mô hình tuyến tính không bị ảnh hưởng bởi ngoại lệ.
- Thử nghiệm thêm mô hình Gradient Boosting hoặc XGBoost để so sánh với Random Forest.
- Cập nhật hình ảnh trực quan hóa có chú thích song ngữ nhằm tăng tính trình bày trong slide.
- Rà soát trích xuất `Ticket` để gộp những nhóm hiếm nhằm tránh ma trận đặc trưng thưa.

# KẾT LUẬN
Nghiên cứu đã hoàn thành chu trình phân tích dữ liệu Titanic: từ kiểm tra dữ liệu đầu vào, xử lý thiếu, tạo đặc trưng nâng cao đến huấn luyện và tinh chỉnh mô hình. Các yếu tố ảnh hưởng mạnh đến sống sót gồm giới tính, hạng vé, cấu trúc gia đình và mức giá vé. Các bước làm sạch giúp khắc phục 19.9% dữ liệu thiếu ở `Age` và 77.1% dữ liệu thiếu ở `Cabin` mà vẫn giữ toàn bộ mẫu. Mô hình Random Forest sau tinh chỉnh đạt độ chính xác cross-validation 84.07%, và ensemble cuối cùng đạt 83.51% trên tập kiểm tra, phù hợp với mục tiêu bài toán phân loại nhị phân. Hạn chế hiện tại là chưa đánh giá sâu các chỉ số khác ngoài accuracy và chưa khai thác các kỹ thuật xử lý mất cân bằng lớp. Nhóm đề xuất tiếp tục mở rộng sang đánh giá đường ROC, cân bằng lớp bằng SMOTE và triển khai dashboard tương tác để trình bày kết quả.

# TÀI LIỆU THAM KHẢO
[1] Kaggle. Titanic - Machine Learning from Disaster. Link: https://www.kaggle.com/c/titanic/data (Truy cập: 05/2024).
[2] F. Pedregosa et al. Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research 12, 2011.

# PHỤ LỤC PHÂN CÔNG NHIỆM VỤ
| STT | Thành viên | Nhiệm vụ |
| --- | --- | --- |
| 1 | Hoàng Minh Nhật | Thu thập và kiểm tra dữ liệu, xây dựng đặc trưng (Title, Family Size, Deck), viết phần mô tả và phương pháp. |
| 2 | Đoàn Chí Hưng | Thực hiện EDA, trực quan hóa, huấn luyện mô hình và tổng hợp kết quả, chuẩn bị slide trình bày. |
