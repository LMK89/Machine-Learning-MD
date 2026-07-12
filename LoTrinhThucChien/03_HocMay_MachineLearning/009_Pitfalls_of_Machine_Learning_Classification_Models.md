## Cam bẫy của máy phân loại mô hình
Trang trình bày 1: Mặt tối của phân loại công việc trong máy học

Phân loại là một nhiệm vụ cơ bản trong máy học nhưng không phải là không có những bẫy bẫy. Bài trình bày này khám phá các vấn đề phổ biến có thể ảnh hưởng đến tính hiệu quả của các loại phân loại mô hình, cùng với các giải pháp thực tế sử dụng Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Sample Classification Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.show()
```

Slide 2: Mất cân bằng giai cấp

Sự mất cân bằng giữa các lớp xảy ra khi một lớp đông hơn đáng kể so với các lớp khác. Điều này có thể dẫn đến các mô hình bị sai lệch hoạt động kém hơn so với số lượng tối thiểu của các tầng.

```python
from sklearn.utils import resample

# Create an imbalanced dataset
X_imbalanced, y_imbalanced = make_classification(n_samples=1000, n_classes=2,
                                                 weights=[0.9, 0.1], n_informative=3,
                                                 random_state=42)

# Upsample the minority class
X_minority = X_imbalanced[y_imbalanced == 1]
y_minority = y_imbalanced[y_imbalanced == 1]
X_minority_upsampled, y_minority_upsampled = resample(X_minority, y_minority,
                                                      n_samples=len(X_imbalanced[y_imbalanced == 0]),
                                                      random_state=42)

# Combine the upsampled minority class with the majority class
X_balanced = np.vstack((X_imbalanced[y_imbalanced == 0], X_minority_upsampled))
y_balanced = np.hstack((y_imbalanced[y_imbalanced == 0], y_minority_upsampled))

print(f"Original class distribution: {np.bincount(y_imbalanced)}")
print(f"Balanced class distribution: {np.bincount(y_balanced)}")
```

Trình bày 3: Trang bị quá trình

Quá trình xảy ra khi một mô hình học dữ liệu huấn luyện quá tốt, bao gồm tiếng ồn và các đặc tính của nó, dẫn đến thông báo hóa học gần hơn và dữ liệu không được tìm thấy.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree with different max_depths
depths = range(1, 20)
train_scores = []
test_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

# Plot the results
plt.plot(depths, train_scores, label='Training Accuracy')
plt.plot(depths, test_scores, label='Testing Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Overfitting in Decision Trees')
plt.legend()
plt.show()
```

Trang trình bày 4: Xu hướng lựa chọn tính năng

Xu hướng lựa chọn đặc tính xảy ra khi chúng ta chọn các tính năng dựa trên hiệu suất của chúng trên toàn bộ dữ liệu, dẫn đến tính toán quá lạc quan về hiệu suất mô hình.

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Generate a dataset with irrelevant features
X_biased, y_biased = make_classification(n_samples=1000, n_features=20, n_informative=5,
                                         n_redundant=5, n_repeated=0, n_classes=2,
                                         random_state=42)

# Incorrect way: Feature selection on entire dataset
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X_biased, y_biased)

# Correct way: Feature selection only on training data
X_train, X_test, y_train, y_test = train_test_split(X_biased, y_biased, test_size=0.3, random_state=42)
selector = SelectKBest(f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

print(f"Number of features before selection: {X_biased.shape[1]}")
print(f"Number of features after selection: {X_selected.shape[1]}")
```

Trang trình bày 5: Bỏ qua các mối tương quan về tính năng

Mối quan hệ tương quan cao giữa các đặc điểm có thể dẫn đến đa tuyến, gây khó khăn cho việc diễn giải tầm quan trọng của từng đặc điểm và có khả năng ảnh hưởng đến hiệu suất mô hình.

```python
import seaborn as sns

# Generate correlated features
n_samples = 1000
X_corr = np.random.randn(n_samples, 3)
X_corr[:, 1] = X_corr[:, 0] + np.random.randn(n_samples) * 0.1
X_corr[:, 2] = X_corr[:, 0] + np.random.randn(n_samples) * 0.1

# Calculate correlation matrix
corr_matrix = np.corrcoef(X_corr.T)

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

Trang trình bày 6: Xác thực chéo không chính xác

Cách xác thực chéo không chính xác có thể dẫn đến sai lệch hiệu suất ước tính. Các lỗi phổ biến bao gồm rò rỉ dữ liệu và sử dụng sai chiến lược cho thời gian chuỗi dữ liệu.

```python
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Generate time series data
np.random.seed(42)
X_ts = np.array([i + np.random.randn() for i in range(1000)]).reshape(-1, 1)
y_ts = (X_ts > 0).astype(int).ravel()

# Incorrect: standard k-fold cross-validation
incorrect_cv_scores = cross_val_score(LogisticRegression(), X_ts, y_ts, cv=5)

# Correct: time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
correct_cv_scores = cross_val_score(LogisticRegression(), X_ts, y_ts, cv=tscv)

print(f"Incorrect CV scores: {incorrect_cv_scores.mean():.3f} (+/- {incorrect_cv_scores.std() * 2:.3f})")
print(f"Correct CV scores: {correct_cv_scores.mean():.3f} (+/- {correct_cv_scores.std() * 2:.3f})")
```

Trang trình bày 7: Bỏ qua quá trình xử lý dữ liệu trước

Việc không xử lý trước dữ liệu đúng có thể dẫn đến hiệu suất màn hình thấp hơn. Các bước tiền xử lý phổ biến bao gồm tỷ lệ chia, xử lý thiếu giá trị và mã hóa các loại phân loại.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create a dataset with mixed types and missing values
X_mixed = np.column_stack([
    np.random.randn(100),  # Continuous feature
    np.random.choice(['A', 'B', 'C'], 100),  # Categorical feature
    np.random.randn(100)  # Continuous feature with missing values
])
X_mixed[np.random.choice(100, 10), 2] = np.nan  # Introduce missing values

# Define preprocessing steps
numeric_features = [0, 2]
categorical_features = [1]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Fit and transform the data
X_preprocessed = preprocessor.fit_transform(X_mixed)

print(f"Shape before preprocessing: {X_mixed.shape}")
print(f"Shape after preprocessing: {X_preprocessed.shape}")
```

Trang trình bày 8: Bỏ qua các giả định về mô hình

Nhiều loại phân loại thuật toán được đưa ra các giả định về dữ liệu. Vi phạm các định nghĩa này có thể dẫn đến hiệu suất màn hình thấp hơn hoặc trình giải mã không chính xác.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Generate non-linearly separable data
X_nonlinear, y_nonlinear = make_classification(n_samples=1000, n_features=2, n_classes=2,
                                               n_clusters_per_class=2, random_state=42)

# Fit LDA (assumes linearly separable classes)
lda = LinearDiscriminantAnalysis()
lda.fit(X_nonlinear, y_nonlinear)

# Plot decision boundary
x_min, x_max = X_nonlinear[:, 0].min() - 1, X_nonlinear[:, 0].max() + 1
y_min, y_max = X_nonlinear[:, 1].min() - 1, X_nonlinear[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c=y_nonlinear, alpha=0.8)
plt.title('LDA on Non-linearly Separable Data')
plt.show()
```

Trang trình bày 9: Giải thích sai số liệu của mô hình

Chỉ dựa vào độ chính xác có thể gây nhầm lẫn, đặc biệt đối với các bộ mất cân bằng dữ liệu. Điều quan trọng là phải xem xét nhiều số liệu để đánh giá toàn diện.

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Create an imbalanced dataset
X_imbalanced, y_imbalanced = make_classification(n_samples=1000, n_classes=2,
                                                 weights=[0.9, 0.1], n_informative=3,
                                                 random_state=42)

# Split the data and train a model
X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.3, random_state=42)
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculate various metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = clf.score(X_test, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

Trang trình bày 10: Không xử lý các ngoại lệ giá trị

Các ngoại lệ có thể tác động đáng kể đến hiệu suất của mô hình, đặc biệt đối với các thuật toán nhạy cảm với các giá trị cực đoan như mô hình tuyến tính hoặc k lân cận gần nhất.

```python
from sklearn.neighbors import KNeighborsClassifier

# Generate data with outliers
X_outliers, y_outliers = make_classification(n_samples=1000, n_features=2, n_informative=2,
                                             n_redundant=0, n_clusters_per_class=1, random_state=42)
X_outliers[0] = [10, 10]  # Add an outlier

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_outliers, y_outliers)

# Plot decision boundary
x_min, x_max = X_outliers[:, 0].min() - 1, X_outliers[:, 0].max() + 1
y_min, y_max = X_outliers[:, 1].min() - 1, X_outliers[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c=y_outliers, alpha=0.8)
plt.title('KNN Classification with Outlier')
plt.show()
```

Trang trình bày 11: Bỏ qua việc chỉnh sửa hiệu suất của lớp

Một số mô hình có thể tạo ra hiệu suất được hiệu chỉnh hợp lý, dẫn đến tính toán đáng tin cậy không đáng tin cậy cho những kỳ vọng.

```python
from sklearn.calibration import calibration_curve
from sklearn.naive_bayes import GaussianNB

# Generate data and split into train and test sets
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Naive Bayes classifier (known for poor probability calibration)
nb = GaussianNB()
nb.fit(X_train, y_train)

# Calculate calibration curve
prob_true, prob_pred = calibration_curve(y_test, nb.predict_proba(X_test)[:, 1], n_bins=10)

# Plot calibration curve
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.plot(prob_pred, prob_true, marker='.', label='Naive Bayes')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.show()
```

Trang trình bày 12: Chưa xem xét khả năng giải mô hình

Các mô hình phức tạp như mạng lưới thần kinh sâu có thể đạt được độ chính xác cao nhưng khó diễn giải, điều này có thể gây ra vấn đề trong các lĩnh vực cần giải quyết cho các giải pháp quyết định.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# Generate and split data
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple decision tree (more interpretable)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Train a more complex Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Compare accuracies
dt_accuracy = dt.score(X_test, y_test)
rf_accuracy = rf.score(X_test, y_test)

print(f"Decision Tree Accuracy: {dt_accuracy:.3f}")
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")

# Visualize the decision tree
plt.figure(figsize=(15,10))
plot_tree(dt, filled=True, feature_names=[f'F{i}' for i in range(X.shape[1])], class_names=['0', '1'])
plt.title('Decision Tree Visualization')
plt.show()
```

Trang trình bày 13: Bỏ qua dữ liệu trôi dạt và giám sát mô hình

Các mô hình có thể trở nên chính xác hơn theo thời gian khi phân phối dữ liệu đến thay đổi. Việc không giám sát và cập nhật các mô hình có thể dẫn đến hiệu suất bị suy giảm.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Simulate data drift
np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)

# Initial data distribution
X_initial = np.random.randn(n_samples // 2, 1)
y_initial = (X_initial > 0).astype(int).ravel()

# Drifted data distribution
X_drift = np.random.randn(n_samples // 2, 1) + 1  # Mean shift
y_drift = (X_drift > 1).astype(int).ravel()

# Combine data
X = np.vstack((X_initial, X_drift))
y = np.hstack((y_initial, y_drift))

# Train model on initial data
model = LogisticRegression()
model.fit(X_initial, y_initial)

# Predict on all data
y_pred = model.predict(X)

# Calculate rolling accuracy
window = 100
rolling_acc = np.array([np.mean(y[i:i+window] == y_pred[i:i+window]) for i in range(n_samples - window)])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time[window:], rolling_acc)
plt.axvline(x=n_samples // 2, color='r', linestyle='--', label='Drift Point')
plt.xlabel('Time')
plt.ylabel('Rolling Accuracy')
plt.title('Model Performance Over Time with Data Drift')
plt.legend()
plt.show()
```

Slide 14: Ví dụ thực tế: Phân loại hình ảnh

Trong các loại hình ảnh phân tích, một phổ biến phổ biến không được tính đến các sai lệch trong huấn luyện dữ liệu. Ví dụ: một mô hình được đào tạo để phân loại động vật có thể hoạt động miễn phí trên các hình ảnh có nền hoặc điều kiện ánh sáng bất thường.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Predict on test set
y_pred = svm.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Digit Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Display some misclassified images
misclassified = X_test[y_test != y_pred]
mis_pred = y_pred[y_test != y_pred]
mis_true = y_test[y_test != y_pred]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i < len(misclassified):
        ax.imshow(misclassified[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'True: {mis_true[i]}, Pred: {mis_pred[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 15: Ví dụ thực tế: Phân loại văn bản

Trong phân loại văn bản, một phổ biến phổ biến là phạm vi công việc được trang bị quá nhiều cho các từ hoặc cụm từ có thể không tốt. Điều này có thể dẫn đến hiệu suất thấp trên bản văn bản mới của dữ liệu, nhưng không được nhìn thấy.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample text data
texts = [
    "I love this product", "Great service", "Terrible experience",
    "Awful customer support", "Amazing quality", "Disappointing results",
    "Outstanding performance", "Waste of money", "Highly recommended",
    "Never buying again"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Evaluate the model
train_score = clf.score(X_train_vec, y_train)
test_score = clf.score(X_test_vec, y_test)

print(f"Training accuracy: {train_score:.2f}")
print(f"Testing accuracy: {test_score:.2f}")

# Show most important features for each class
feature_names = vectorizer.get_feature_names_out()
for i, category in enumerate(["Negative", "Positive"]):
    top_features = sorted(zip(clf.feature_log_prob_[i], feature_names), reverse=True)[:5]
    print(f"\nTop 5 features for {category}:")
    for score, word in top_features:
        print(f"{word}: {score:.2f}")
```

Trang trình bày 16: Tài nguyên bổ sung

Để khám phá thêm về những góc bẫy của máy học và các phương pháp hay nhất, hãy xem xét các tài nguyên sau:

1. "Một số điều hữu ích cần biết về học máy" của Pedro Domingos ArXiv link: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
2. "Học máy: Thẻ tín dụng lãi suất cao cho nợ kỹ thuật" của D. Sculley và cộng sự. Liên kết ArXiv: [https://arxiv.org/abs/1410.5244](https://arxiv.org/abs/1410.5244)
3. "Nợ kỹ thuật ẩn trong hệ thống máy học" của D. Sculley và cộng đồng. Liên kết ArXiv: [https://arxiv.org/abs/1412.6564](https://arxiv.org/abs/1412.6564)
4. "Khắc phục sự cố mạng thần kinh sâu" của Josh Tobin Có tại: [http://josh-tobin.com/assets/pdf/troubleshooting-deep-neural-networks-01-19.pdf](http://josh-tobin.com/assets/pdf/troubleshooting-deep-neural-networks-01-19.pdf)

Những tài nguyên này cung cấp những hiểu biết sâu sắc có giá trị về các công thức chung trong máy học và các chiến lược để vượt qua chúng.
