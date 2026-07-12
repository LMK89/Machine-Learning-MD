## So sánh Rừng ngẫu nhiên và XGBoost trong Python
Slide 1: Giới thiệu về Rừng ngẫu nhiên và XGBoost

Rừng ngẫu nhiên và XGBoost là các thuật toán tổng hợp mạnh mẽ được sử dụng trong máy học. Cả hai phương pháp đều kết hợp nhiều quyết định để tạo ra các mô hình dự kiến ​​mạnh mẽ. Bài trình bày này sẽ khám phá những điểm tương đồng, khác biệt và cách phát triển chúng trong Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Visualize decision boundaries
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()

# Example usage (to be used in later slides)
# plot_decision_boundary(model, X, y)
```

Slide 2: Rừng ngẫu nhiên: Khái niệm cơ bản

Rừng ngẫu nhiên là một tập hợp các cây được quyết định. Nó tạo ra nhiều quyết định về các mẫu dữ liệu được chọn ngẫu nhiên, nhận được kỳ vọng từ mỗi cây và chọn giải pháp tốt nhất bằng cách bỏ phiếu. Nó cung cấp độ chính xác và ổn định tốt hơn so với việc quyết định đơn lẻ.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
```

Trang trình bày 3: Rừng ngẫu nhiên: Các tính năng chính

Rừng ngẫu nhiên sử dụng đóng bao (tập hợp bootstrap) để tạo ra nhiều dạng huấn luyện dữ liệu cho mỗi cây. Nó cũng sử dụng tính năng ngẫu nhiên khi xây dựng cây, làm tăng tính đa dạng của khu rừng và giúp ngăn chặn tình trạng trang web quá mạnh.

```python
# Demonstrate feature importance in Random Forest
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, sorted_idx)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()
```

Trang trình bày 4: XGBoost: Khái niệm cơ bản

XGBoost (Tăng cường độ dốc eXtreme) là một cấp độ tăng cường thuật toán được tối ưu hóa. Nó xây dựng các cây theo cách tuần tự, trong đó mỗi cây mới sẽ sửa các lỗi mà cây mắc phải trước đó. XGBoost được biết đến với tốc độ và hiệu suất, đặc biệt là với dữ liệu có cấu hình cấu trúc/bảng bảng.

```python
# Create and train an XGBoost model
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate the model
xgb_accuracy = xgb_model.score(X_test, y_test)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
```

Trang trình bày 5: XGBoost: Các tính năng chính

XGBoost sử dụng quy định xác định chính xác hóa mô hình hóa hơn để kiểm soát Kiểm soát công việc của trang đang diễn ra. Nó bao gồm các tính năng nâng cao như xử lý các giá trị bị thiếu, cắt cây và phân tích chéo. XGBoost cũng hỗ trợ bài hát và phân tích bài hát tính toán, giúp nó có khả năng mở rộng cao.

```python
# Demonstrate feature importance in XGBoost
xgb_feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(xgb_feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10, 6))
plt.barh(pos, xgb_feature_importance[sorted_idx], align='center')
plt.yticks(pos, sorted_idx)
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()
```

Slide 6: Xử lý các giá trị bị thiếu

Rừng ngẫu nhiên có thể xử lý các giá trị bị thiếu trong khi XGBoost có sẵn phương thức tích hợp để xử lý chúng. Vui lòng so sánh hiệu suất của chúng trên dữ liệu với các giá trị còn thiếu.

```python
import pandas as pd

# Create a dataset with missing values
X_missing = X.()
X_missing[np.random.choice(X.shape[0], 100), np.random.choice(X.shape[1], 5)] = np.nan

# Convert to pandas DataFrame
X_missing_df = pd.DataFrame(X_missing)

# Random Forest with missing values
rf_missing = RandomForestClassifier(n_estimators=100, random_state=42)
rf_missing.fit(X_missing_df, y)
rf_missing_score = rf_missing.score(X_missing_df, y)

# XGBoost with missing values
xgb_missing = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_missing.fit(X_missing_df, y)
xgb_missing_score = xgb_missing.score(X_missing_df, y)

print(f"Random Forest Score with missing values: {rf_missing_score:.4f}")
print(f"XGBoost Score with missing values: {xgb_missing_score:.4f}")
```

Slide 7: Điều chỉnh siêu thông số

Cả Random Forest và XGBoost đều có nhiều siêu tham số có thể được điều chỉnh để đạt được hiệu suất tối ưu. Hãy sử dụng GridSearchCV để tìm các thông tin tốt nhất cho từng hình.

```python
from sklearn.model_selection import GridSearchCV

# Random Forest parameter grid
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3)
rf_grid.fit(X_train, y_train)

print("Best Random Forest parameters:", rf_grid.best_params_)
print("Best Random Forest score:", rf_grid.best_score_)

# XGBoost parameter grid
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

xgb_grid = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=3)
xgb_grid.fit(X_train, y_train)

print("Best XGBoost parameters:", xgb_grid.best_params_)
print("Best XGBoost score:", xgb_grid.best_score_)
```

Slide 8: Lựa chọn tính năng

Cả Rừng ngẫu nhiên và XGBoost đều có thể được sử dụng để lựa chọn tính năng. Vui lòng so sánh các tính năng cấp độ quan trọng của chúng.

```python
from sklearn.feature_selection import SelectFromModel

# Random Forest feature selection
rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
rf_selector.fit(X_train, y_train)
rf_selected_features = X_train.columns[rf_selector.get_support()].tolist()

# XGBoost feature selection
xgb_selector = SelectFromModel(XGBClassifier(n_estimators=100, random_state=42), threshold='median')
xgb_selector.fit(X_train, y_train)
xgb_selected_features = X_train.columns[xgb_selector.get_support()].tolist()

print("Random Forest selected features:", rf_selected_features)
print("XGBoost selected features:", xgb_selected_features)

# Compare feature importance
plt.figure(figsize=(10, 6))
plt.scatter(range(X_train.shape[1]), rf_model.feature_importances_, label='Random Forest')
plt.scatter(range(X_train.shape[1]), xgb_model.feature_importances_, label='XGBoost')
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.legend()
plt.title('Feature Importance Comparison')
plt.show()
```

Trang trình bày 9: Ví dụ thực tế: Tỷ lệ dự kiến ​​khi bỏ hàng

Vui lòng sử dụng Random Forest và XGBoost để dự đoán tỷ lệ bỏ rơi khách hàng trong một công ty viễn thông. Chúng tôi sẽ sử dụng một tập hợp các tính năng từ dữ liệu thông thường.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a sample churn dataset
data = {
    'tenure': np.random.randint(1, 72, 1000),
    'monthly_charges': np.random.uniform(20, 100, 1000),
    'total_charges': np.random.uniform(100, 5000, 1000),
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
    'online_security': np.random.choice(['Yes', 'No'], 1000),
    'tech_support': np.random.choice(['Yes', 'No'], 1000),
    'churn': np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # 30% churn rate
}

df = pd.DataFrame(data)

# Preprocess the data
le = LabelEncoder()
df['contract'] = le.fit_transform(df['contract'])
df['online_security'] = le.fit_transform(df['online_security'])
df['tech_support'] = le.fit_transform(df['tech_support'])

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Random Forest
rf_churn = RandomForestClassifier(n_estimators=100, random_state=42)
rf_churn.fit(X_train, y_train)
rf_churn_score = rf_churn.score(X_test, y_test)

# Train and evaluate XGBoost
xgb_churn = XGBClassifier(n_estimators=100, random_state=42)
xgb_churn.fit(X_train, y_train)
xgb_churn_score = xgb_churn.score(X_test, y_test)

print(f"Random Forest Churn Prediction Accuracy: {rf_churn_score:.4f}")
print(f"XGBoost Churn Prediction Accuracy: {xgb_churn_score:.4f}")
```

Slide 10: Ví dụ thực tế: Phân loại hình ảnh

Mặc dù Random Forest và XGBoost chủ yếu được sử dụng cho dữ liệu có cấu trúc nhưng chúng cũng có thể được áp dụng cho các loại hình ảnh phân tích bằng cách trích xuất các đặc điểm từ hình ảnh. Vui lòng sử dụng một ví dụ đơn giản cho tệp dữ liệu MNIST.

```python
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Random Forest
rf_digits = RandomForestClassifier(n_estimators=100, random_state=42)
rf_digits.fit(X_train, y_train)
rf_digits_pred = rf_digits.predict(X_test)

# Train and evaluate XGBoost
xgb_digits = XGBClassifier(n_estimators=100, random_state=42)
xgb_digits.fit(X_train, y_train)
xgb_digits_pred = xgb_digits.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_digits_pred))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_digits_pred))

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"True: {y_test[i]}, RF: {rf_digits_pred[i]}, XGB: {xgb_digits_pred[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 11: So sánh hiệu suất

Vui lòng so sánh hiệu suất của Rừng ngẫu nhiên và XGBoost trên các dữ liệu tệp khác có kích thước và độ phức tạp.

```python
from sklearn.datasets import make_classification
from time import time

def compare_models(n_samples, n_features):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(n_estimators=100, random_state=42)

    rf_start = time()
    rf.fit(X_train, y_train)
    rf_time = time() - rf_start
    rf_score = rf.score(X_test, y_test)

    xgb_start = time()
    xgb.fit(X_train, y_train)
    xgb_time = time() - xgb_start
    xgb_score = xgb.score(X_test, y_test)

    return rf_time, rf_score, xgb_time, xgb_score

datasets = [(1000, 10), (10000, 20), (100000, 50)]
results = []

for samples, features in datasets:
    rf_time, rf_score, xgb_time, xgb_score = compare_models(samples, features)
    results.append((samples, features, rf_time, rf_score, xgb_time, xgb_score))

# Print results
for r in results:
    print(f"Dataset: {r[0]} samples, {r[1]} features")
    print(f"Random Forest - Time: {r[2]:.2f}s, Score: {r[3]:.4f}")
    print(f"XGBoost - Time: {r[4]:.2f}s, Score: {r[5]:.4f}")
    print()

# Plotting code omitted for brevity
```

Slide 12: Điểm mạnh và điểm yếu

Rừng ngẫu nhiên: Điểm mạnh: Xử lý tốt các mối quan hệ phi tuyến tính, ít thiết bị trang bị quá khả năng, có thể xử lý dữ liệu nhiều chiều, cung cấp tầm quan trọng của tính năng. Điểm yếu: Có thể trả giá về mặt tính toán cho các tập dữ liệu lớn, có thể gặp khó khăn với da thưa thớt.

XGBoost: Điểm mạnh: Thường đạt được hiệu suất tốt hơn, xử lý tốt các tập dữ liệu mất cân bằng, tích hợp chính quy, đào tạo và dự đoán nhanh hơn. Điểm yếu: Trang thiết bị được đo quá mức nếu không được điều chỉnh đúng, có thể nhận được ngoại lệ giá trị.

```python
# Pseudocode for choosing between Random Forest and XGBoost

def choose_algorithm(dataset, task):
    if dataset.is_high_dimensional() and not dataset.is_sparse():
        return RandomForest()
    elif dataset.is_imbalanced() or task.requires_high_performance():
        return XGBoost()
    elif dataset.has_outliers():
        return RandomForest()
    else:
        return "Try both and compare performance"

# Example usage
result = choose_algorithm(my_dataset, my_task)
print(f"Recommended algorithm: {result}")
```

Slide 13: Ensembles của Ensembles

Chúng tôi có thể tạo ra một meta hợp lý nhất bằng cách kết hợp các dự đoán ngẫu nhiên và XGBoost. Cách tiếp cận này đôi khi có thể dẫn đến hiệu suất tốt hơn bằng cách sử dụng một thuật toán chỉ.

```python
from sklearn.ensemble import VotingClassifier

# Create base models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(n_estimators=100, random_state=42)

# Create voting classifier
voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'
)

# Train the ensemble
voting_model.fit(X_train, y_train)

# Evaluate the ensemble
ensemble_score = voting_model.score(X_test, y_test)
print(f"Ensemble Model Score: {ensemble_score:.4f}")

# Compare with individual model scores
rf_score = rf_model.fit(X_train, y_train).score(X_test, y_test)
xgb_score = xgb_model.fit(X_train, y_train).score(X_test, y_test)

print(f"Random Forest Score: {rf_score:.4f}")
print(f"XGBoost Score: {xgb_score:.4f}")
```

Slide 14: Khả năng diễn giải và giải thích

Mặc dù cả Random Forest và XGBoost đều cung cấp tầm quan trọng của các tính năng, XGBoost cung cấp các công cụ bổ sung để diễn giải mô hình, bên cạnh các giá trị SHAP (SHapley Additive exPlanations).

```python
import shap

# Train XGBoost model
X, y = shap.datasets.adult()
model = XGBClassifier().fit(X, y)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize feature importance
shap.summary_plot(shap_values, X, plot_type="bar")

# Visualize SHAP values for a single prediction
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
```

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin chuyên sâu về Rừng ngẫu nhiên và XGBoost, hãy xem xét khám phá các tài nguyên sau:

1. "Những khu rừng ngẫu nhiên" của Leo Breiman (2001): [https://arxiv.org/abs/stat.ML/0312039](https://arxiv.org/abs/stat.ML/0312039)
2. "XGBoost: Hệ thống tăng cường cây có thể mở rộng" của Chen và Guestrin (2016): [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
3. Tài liệu Scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
4. Tài liệu XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

Các tài nguyên này cung cấp giải pháp chi tiết về các thuật toán, cách phát triển khai báo và các phương pháp hay nhất để sử dụng chúng trong các nhiệm vụ học máy khác nhau.
