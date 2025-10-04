# Mô hình hóa Titanic – Giải thích từng bước

## 1. Mục tiêu chung
Phần *Modeling* trong notebook tiến hành biến tập dữ liệu Titanic sau tiền xử lý thành các mô hình dự đoán xác suất sống sót. Quy trình bao gồm: tách tập train/test, huấn luyện mô hình baseline Logistic Regression, sau đó so sánh với các mô hình khác (Decision Tree, SVM, Random Forest, Logistic Regression, KNN) bằng Grid Search kết hợp Stratified K-Fold. Cuối cùng ghép các mô hình tốt nhất bằng VotingClassifier để kiểm tra hiệu quả ensemble.

## 2. Các khái niệm quan trọng
- **Train/Test Split**: chia dữ liệu thành hai phần (huấn luyện và kiểm tra) để đánh giá mô hình trên dữ liệu chưa từng nhìn thấy.
- **Logistic Regression**: mô hình tuyến tính trả về xác suất (0–1) cho lớp “sống sót”. Dễ huấn luyện và dễ giải thích, nên được dùng làm baseline.
- **Decision Tree**: mô hình dạng cây "nếu – thì", tự động tìm ngưỡng tốt nhất để chia dữ liệu; trực quan nhưng dễ overfit.
- **Random Forest**: tập hợp nhiều cây quyết định, mỗi cây huấn luyện trên mẫu dữ liệu khác nhau rồi bỏ phiếu; thường ổn định và chính xác hơn cây đơn.
- **SVM, KNN**: hai mô hình được đưa vào để tham khảo thêm; nếu chỉ cần mức cơ bản có thể dừng ở Logistic Regression, Decision Tree và Random Forest.
- **Hyperparameter**: tham số phải cấu hình trước khi huấn luyện (ví dụ độ sâu cây, số lượng cây, hệ số phạt C của logistic). Không được mô hình tự học nên cần dò tìm.
- **Grid Search**: duyệt qua mọi tổ hợp hyperparameter trong một “lưới” giá trị do người dùng định nghĩa, huấn luyện từng mô hình rồi chọn tổ hợp tốt nhất.
- **Stratified K-Fold Cross Validation**: chia dữ liệu train thành *k* phần cân bằng tỉ lệ sống sót; lặp lại huấn luyện/đánh giá *k* lần và lấy trung bình, giúp kết quả ổn định.
- **Accuracy**: tỉ lệ dự đoán đúng trên tổng mẫu; đây là tiêu chí được dùng trong notebook để so sánh mô hình.
- **VotingClassifier (Ensemble)**: kết hợp nhiều mô hình con (ở đây là Decision Tree, Random Forest, Logistic Regression) và lấy trung bình xác suất (soft voting) để ra dự đoán cuối cùng.

## 2.5 Feature Engineering quan trọng
- `Sex`: mã hóa one-hot (`Sex_female`, `Sex_male`) để mô hình phản ánh rõ tỷ lệ sống sót cao hơn của nữ.
- `Pclass`: one-hot (`Pclass_1`, `Pclass_2`, `Pclass_3`) giúp tách riêng tác động từng hạng vé đến xác suất sống sót.
- `Age`: điền thiếu bằng median theo nhóm (`Pclass`, `SibSp`, `Parch`), giữ dạng liên tục cho mô hình; nếu cần dễ trình bày có thể tạo thêm `AgeQuantile` nhưng phải kiểm chứng vì binning có thể làm giảm độ chính xác.
- `Fare`: chia thành 5 nhóm theo quantile (`FareQuantile`) để giảm ảnh hưởng ngoại lệ và làm nổi bật xu hướng “giá vé cao → cơ hội sống sót lớn”.
- `SibSp` + `Parch`: gộp thành `FamilySize` và cờ `IsAlone` nêu bật lợi thế của gia đình nhỏ (1–3 người đi cùng).
- `Cabin`: rút ký tự đầu thành `Deck`, gán `Unknown` khi thiếu để vẫn giữ thông tin vị trí khoang.
- `Embarked`: one-hot (`Embarked_C`, `_Q`, `_S`) do khách lên tàu ở Cherbourg có tỷ lệ sống sót cao hơn.
- `Ticket`: chuẩn hóa prefix và one-hot các nhóm lặp lại nhằm nhận diện những nhóm hành khách đi chung.


## 3. Giải thích từng ô code chính
### 3.1. Import thư viện (Cell 125)
```python
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
```
- Gọi các lớp cần dùng cho bước tách dữ liệu, mô hình cơ bản và các thuật toán sẽ so sánh. Việc import đầy đủ giúp tái sử dụng trong các ô tiếp theo.

### 3.2. Tách tập train/test (Cell 127)
```python
train = train_df[:train_df_len]
X_train = train.drop(labels="Survived", axis=1)
y_train = train["Survived"]
X_test = test_df.drop(labels="PassengerId", axis=1)
```
- `train_df_len` nhớ lại kích thước ban đầu trước khi nối dữ liệu test, bảo đảm chỉ dùng phần train thật sự.
- `X_train` chứa toàn bộ đặc trưng, `y_train` là nhãn sống sót.
- `X_test` chuẩn bị cho việc dự đoán cuối cùng (ẩn trong notebook).

### 3.3. Logistic Regression baseline (Cell 129)
```python
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
acc_log_train = round(log_reg.score(X_train, y_train) * 100, 2)
```
- Khởi tạo Logistic Regression mặc định, huấn luyện bằng `fit`.
- `score` với dữ liệu train cho biết accuracy ban đầu. Đây là mốc để so sánh với các mô hình khác.

### 3.4. Chuẩn bị hyperparameter & mô hình (Cell 131)
- Tạo danh sách `classifier` gồm 5 mô hình.
- Định nghĩa `dt_param_grid`, `svc_param_grid`, `rf_param_grid`, `logreg_param_grid`, `knn_param_grid`: mỗi biến là tập giá trị hyperparameter cần dò.
- Mục tiêu: đưa vào GridSearchCV ở bước kế tiếp.

### 3.5. Grid Search + Stratified K-Fold (Cell 132)
```python
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(
        classifier[i],
        param_grid=classifier_param[i],
        cv=StratifiedKFold(n_splits=10),
        scoring="accuracy",
        n_jobs=1,
        verbose=1
    )
    clf.fit(X_train, y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
```
- Vòng lặp lần lượt áp dụng Grid Search cho từng mô hình.
- `StratifiedKFold(10)` đảm bảo mỗi fold giữ tỉ lệ sống sót tương tự toàn bộ dữ liệu.
- Sau khi `fit`, ta lưu lại điểm cross-validation tốt nhất và mô hình đã tối ưu hyperparameter (`best_estimator_`).

### 3.6. Bảng so sánh kết quả (Cell 133–134)
```python
cv_results = pd.DataFrame({
    "Cross Validation Accuracy Means": cv_result,
    "ML Models": ["DecisionTreeClassifier", "SVM", "RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier"]
})
sns.barplot(cv_results, x="ML Models", y="Cross Validation Accuracy Means")
```
- Tạo DataFrame để xem nhanh accuracy trung bình của từng mô hình sau khi tối ưu.
- Vẽ biểu đồ cột giúp trực quan mô hình nào hoạt động tốt nhất.

### 3.7. Ensemble Voting (Cell 136)
```python
votingC = VotingClassifier(
    estimators=[
        ("dt", best_estimators[0]),
        ("rfc", best_estimators[2]),
        ("lr", best_estimators[3])
    ],
    voting="soft",
    n_jobs=1
)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test), y_test))
```
- Lấy ba mô hình hiệu quả nhất (Decision Tree, Random Forest, Logistic Regression) sau Grid Search để tạo ensemble.
- `voting="soft"` nghĩa là lấy trung bình xác suất dự đoán.
- Huấn luyện lại trên toàn bộ `X_train`, sau đó đánh giá accuracy trên `X_test` (hoặc hold-out test nếu có nhãn).

## 4. Nhật xét & Kết luận
- Logistic Regression là baseline dễ hiểu; Decision Tree và Random Forest cho phép nắm bắt quan hệ phi tuyến và tương tác giữa các đặc trưng.
- Grid Search kết hợp Stratified K-Fold đảm bảo mỗi mô hình được điều chỉnh công bằng, tránh phụ thuộc vào một lần chia dữ liệu duy nhất.
- Kết quả cross-validation cho biết mô hình nào đáng tin cậy nhất; nếu Random Forest dẫn đầu, ta có thể dùng nó hoặc ensemble với Logistic để cân bằng giữa độ chính xác và khả năng giải thích.
- VotingClassifier giúp tổng hợp ưu điểm của từng mô hình con. Tuy nhiên với báo cáo nhập môn, bạn có thể dừng ở việc so sánh 3 mô hình cơ bản rồi lựa chọn mô hình có accuracy cao nhất và giải thích lý do.

Tổng thể, phần Modeling trong notebook minh họa đầy đủ các bước chuẩn: baseline → dò hyperparameter → so sánh → (tùy chọn) ensemble. Đây là khung quy trình bạn có thể tái sử dụng cho những bộ dữ liệu phân loại khác.


Mục tiêu của phần huấn luyện mô hình:

Sau khi khám phá và tiền xử lý dữ liệu, mục tiêu của chúng ta là xây dựng một mô hình máy học có khả năng dự đoán liệu một hành khách trên tàu Titanic có sống sót hay không (Survived) dựa trên các đặc trưng khác của họ (tuổi, giới tính, hạng vé, v.v.).

Các mô hình được sử dụng:

Chúng ta đã chọn ba loại mô hình phân loại phổ biến để thử nghiệm:

Logistic Regression (Hồi quy Logistic): Đây là một mô hình tuyến tính đơn giản, thường được sử dụng cho các bài toán phân loại nhị phân (ví dụ: sống sót hoặc không sống sót). Nó ước tính xác suất một trường hợp thuộc về một lớp nhất định.
Decision Tree Classifier (Cây Quyết định): Mô hình này hoạt động bằng cách tạo ra một cấu trúc giống cây, trong đó mỗi nút biểu diễn một quyết định dựa trên một đặc trưng, và các nhánh dẫn đến các kết quả khác nhau. Nó dễ hiểu và diễn giải.
Random Forest Classifier (Rừng Ngẫu nhiên): Đây là một mô hình ensemble (kết hợp nhiều mô hình nhỏ). Nó xây dựng nhiều cây quyết định độc lập và kết hợp kết quả của chúng để đưa ra dự đoán cuối cùng. Random Forest thường mạnh mẽ và ít bị overfitting hơn Decision Tree đơn lẻ.
Tại sao lại chia dữ liệu thành Train - Test?

Để đánh giá xem mô hình của chúng ta hoạt động tốt như thế nào trên dữ liệu chưa từng thấy, chúng ta chia tập dữ liệu ban đầu thành hai phần:

Tập huấn luyện (Training set): Sử dụng để "dạy" mô hình cách học từ dữ liệu.
Tập kiểm tra (Testing set): Sử dụng để đánh giá hiệu suất của mô hình sau khi đã được huấn luyện. Mô hình chưa từng thấy dữ liệu này trong quá trình học, vì vậy nó cho ta biết mô hình có khả năng tổng quát hóa tốt đến mức nào.
Tỷ lệ chia 67%/33% là một tỷ lệ phổ biến, cung cấp đủ dữ liệu cho cả hai tập.

Mô hình Simple Logistic Regression:

Đây là bước đầu tiên để có cái nhìn nhanh về hiệu suất của một mô hình cơ bản (Logistic Regression) mà chưa tinh chỉnh bất kỳ cài đặt nào. Kết quả độ chính xác trên tập huấn luyện (81.53%) và tập kiểm tra (83.16%) cho ta một điểm tham chiếu ban đầu.

Tinh chỉnh các siêu tham số (Hyperparameter Tuning):

Mỗi mô hình máy học có các "siêu tham số" (hyperparameters) là các cài đặt mà chúng ta cần chọn trước khi huấn luyện mô hình (ví dụ: độ sâu tối đa của cây quyết định, số lượng cây trong rừng ngẫu nhiên). Việc chọn đúng siêu tham số có thể ảnh hưởng đáng kể đến hiệu suất của mô hình.

Tại sao cần tinh chỉnh siêu tham số?

Mục tiêu là tìm ra bộ siêu tham số tốt nhất giúp mô hình đạt hiệu suất cao nhất trên dữ liệu mới, tránh tình trạng overfitting (mô hình học quá kỹ dữ liệu huấn luyện mà không tổng quát hóa được) hoặc underfitting (mô hình quá đơn giản để học được quy luật).

Grid Search (Tìm kiếm lưới):

Grid Search là một kỹ thuật phổ biến để tinh chỉnh siêu tham số. Chúng ta định nghĩa một "lưới" các giá trị có thể có cho các siêu tham số của mô hình. Grid Search sau đó sẽ thử nghiệm tất cả các kết hợp có thể có của các giá trị này và đánh giá hiệu suất của mô hình với từng sự kết hợp đó.

Cross-Validation (Kiểm định chéo):

Để đánh giá hiệu suất của mỗi sự kết hợp siêu tham số một cách đáng tin cậy hơn, chúng ta sử dụng Cross-Validation, cụ thể ở đây là Stratified K-Fold Cross-Validation.

K-Fold: Chúng ta chia tập huấn luyện thành K (ở đây là 10) phần nhỏ bằng nhau.
Stratified: Đảm bảo rằng tỷ lệ của lớp mục tiêu (Survived) trong mỗi phần nhỏ tương đương với tỷ lệ trong toàn bộ tập huấn luyện. Điều này đặc biệt quan trọng với các tập dữ liệu không cân bằng (ví dụ: số người sống sót ít hơn số người không sống sót).
Quá trình kiểm định chéo diễn ra như sau: Mô hình được huấn luyện trên K-1 phần và đánh giá trên phần còn lại. Quá trình này lặp lại K lần, mỗi lần sử dụng một phần khác nhau làm tập đánh giá.
Điểm cuối cùng của một sự kết hợp siêu tham số là điểm trung bình trên K lần lặp này.
Tại sao lại sử dụng Cross-Validation?

Cross-Validation giúp chúng ta có một ước lượng đáng tin cậy hơn về hiệu suất thực sự của mô hình trên dữ liệu chưa thấy, giảm thiểu sự phụ thuộc vào một lần chia tập huấn luyện/đánh giá cụ thể.

random_state = 42:

Việc đặt random_state bằng một số cố định (ví dụ: 42) đảm bảo rằng các quá trình ngẫu nhiên (như chia tập dữ liệu, khởi tạo mô hình) sẽ luôn tạo ra kết quả giống nhau mỗi lần bạn chạy code. Điều này rất quan trọng cho khả năng tái lập (reproducibility) của kết quả. Nếu không đặt random_state, mỗi lần chạy code, bạn có thể nhận được kết quả hơi khác nhau do yếu tố ngẫu nhiên.

Tại sao các tham số trong từng model lại cụ thể là như vậy?

Các tham số cụ thể được chọn trong dt_param_grid, rf_param_grid, và logreg_param_grid là các giá trị phổ biến và thường mang lại hiệu suất tốt cho các mô hình tương ứng.

Decision Tree (dt_param_grid):
min_samples_split: Số lượng mẫu tối thiểu cần có để một nút có thể được chia nhỏ hơn. Các giá trị từ 10 đến 500 với bước nhảy 20 được chọn để khám phá xem kích thước nút tối thiểu nào là phù hợp.
max_depth: Độ sâu tối đa của cây. Các giá trị lẻ từ 1 đến 19 được chọn để kiểm soát độ phức tạp của cây.
Random Forest (rf_param_grid):
max_features: Số lượng đặc trưng tối đa được xem xét khi tìm kiếm phân tách tốt nhất. Các giá trị 1, 3, 10 là các lựa chọn phổ biến.
min_samples_split: Tương tự như Decision Tree.
min_samples_leaf: Số lượng mẫu tối thiểu cần có ở một nút lá.
bootstrap: Có lấy mẫu bootstrap khi xây dựng cây con hay không. False được chọn để thử nghiệm.
n_estimators: Số lượng cây trong rừng. 100 và 300 là các giá trị thường được thử nghiệm.
criterion: Tiêu chí để đo chất lượng phân tách. gini là một lựa chọn phổ biến.
Logistic Regression (logreg_param_grid):
C: Tham số nghịch đảo của sức mạnh điều chuẩn (regularization). Giá trị nhỏ hơn thể hiện điều chuẩn mạnh hơn (ngăn overfitting). np.logspace(-3, 3, 7) tạo ra một loạt các giá trị C trải đều trên thang logarit.
penalty: Loại điều chuẩn được áp dụng (l1 hoặc l2).
max_iter: Số lần lặp tối đa để thuật toán hội tụ. Các giá trị lớn hơn được chọn để đảm bảo thuật toán có đủ thời gian tìm ra lời giải tốt nhất.
Các giá trị này được chọn dựa trên kinh nghiệm và thực tiễn trong machine learning. Grid Search giúp chúng ta tìm ra sự kết hợp tốt nhất trong phạm vi các giá trị đã chọn này.

Mô hình Ensemble (VotingClassifier):

Sau khi tìm được bộ siêu tham số tốt nhất cho từng mô hình đơn lẻ, chúng ta sử dụng VotingClassifier để kết hợp chúng.

estimators: Danh sách các mô hình đã được huấn luyện với siêu tham số tốt nhất.
voting="soft": Sử dụng xác suất dự đoán của từng mô hình thay vì chỉ dự đoán lớp cuối cùng. Lớp có xác suất trung bình cao nhất sẽ được chọn. Điều này thường cho kết quả tốt hơn so với "hard" voting (chỉ đếm số phiếu).
Mô hình Ensemble sẽ đưa ra dự đoán cuối cùng dựa trên "phiếu bầu" của các mô hình thành phần.
Kết luận:

Quá trình này giúp chúng ta không chỉ tìm ra mô hình đơn lẻ hoạt động tốt nhất mà còn xem xét liệu việc kết hợp các mô hình có cải thiện hiệu suất hay không. Độ chính xác trên tập kiểm tra cuối cùng (0.8351) cho ta biết ước lượng về hiệu suất của mô hình Ensemble trên dữ liệu mới.
