## Hồi quy XGBoost với Python
Trang trình bày 1: Giới thiệu về XGBoost phục hồi

XGBoost (Tăng cường độ dốc cực cao) là một máy tính toán mạnh mẽ cho các nhiệm vụ phục hồi quy quy. Đó là một cách phát triển tối ưu hóa công việc tăng cường độ dốc mang lại hiệu quả và độ chính xác cao.

```python
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
```

Trang trình bày 2: Tính năng của XGBoost

XGBoost cung cấp nhiều lợi thế, bao gồm chính hóa hóa, xử lý các giá trị còn thiếu và xử lý bài hát. Nó sử dụng tập hợp cây quyết định và tăng cường độ dốc để tạo ra một mô hình mạnh mẽ.

```python
# XGBoost with custom parameters
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train the model
xgb_model.fit(X_train, y_train)
```

Slide 3: Chuẩn bị dữ liệu

Trước khi đào tạo mô hình XGBoost, điều quan trọng là bạn phải chuẩn bị dữ liệu đúng cách. Điều này bao gồm việc xử lý các giá trị bị thiếu, mã hóa các biến thể phân loại loại và tính toán tỷ lệ chia nếu cần.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load sample data
data = pd.read_csv('sample_data.csv')

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
data['category'] = le.fit_transform(data['category'])

# Scale numerical features
scaler = StandardScaler()
data[['feature1', 'feature2']] = scaler.fit_transform(data[['feature1', 'feature2']])
```

Trang trình bày 4: Đào tạo bộ phục hồi XGBoost

Huấn luyện bộ phục hồi XGBoost liên kết với công việc điều chỉnh mô hình phù hợp với huấn luyện dữ liệu của bạn. Bạn có thể sử dụng nhiều tham số khác nhau để kiểm soát quá trình luyện tập và tốc độ phức tạp của mô hình.

```python
# Prepare features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
```

Slide 5: Đưa ra dự đoán

Sau khi đào tạo mô hình XGBoost, bạn có thể sử dụng mô hình đó để đưa ra dự đoán về dữ liệu mới. Điều này rất hữu ích cho việc đánh giá mô hình và áp dụng nó vào các vấn đề trong thế giới thực.

```python
# Make predictions on test data
y_pred = xgb_model.predict(X_test)

# Calculate Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Make a single prediction
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example features
prediction = xgb_model.predict(new_data)
print(f"Prediction for new data: {prediction[0]}")
```

Trang trình bày 6: Tầm quan trọng của tính năng

XGBoost cho phép bạn đánh giá mức độ quan trọng của từng tính năng trong hình ảnh của mình. Điều này có thể giúp bạn hiểu được những biến thể nào có tác dụng tốt nhất mà bạn mong đợi.

```python
import matplotlib.pyplot as plt

# Get feature importance
importance = xgb_model.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance)
plt.xticks(range(len(importance)), feature_names, rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

Slide 7: Điều chỉnh siêu thông số

Siêu tham số tối ưu của XGBoost có thể cải thiện hiệu suất đáng kể của mô hình. Tìm kiếm mạng và tìm kiếm ngẫu nhiên là các phương pháp phổ biến để tìm kiếm các tham số tốt nhất.

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0]
}

# Perform grid search
grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)
```

Trang trình bày 8: Xác thực chéo

Xác thực tính năng chéo của mô hình XGBoost đối với dữ liệu không được tìm thấy. Nó đặc biệt hữu ích khi bạn có dữ liệu giới hạn chế độ.

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert scores to positive values
cv_scores = -cv_scores

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())
```

Trang trình bày 9: Dừng sớm

Việc dừng sớm có thể ngăn chặn trạng thái trạng thái bằng cách dừng quá trình huấn luyện khi hiệu suất của mô hình trên cơ sở xác thực cải tiến.

```python
from sklearn.model_selection import train_test_split

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train with early stopping
xgb_model = xgb.XGBRegressor(n_estimators=1000)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=10,
              eval_metric='rmse',
              verbose=False)

print("Best iteration:", xgb_model.best_iteration)
```

Slide 10: Xử lý mất cân bằng dữ liệu

Khi xử lý khôi phục dữ liệu không cân bằng, bạn có thể sử dụng số lượng mẫu quan trọng để chú ý hơn đến việc thiếu các mẫu trình bày.

```python
import numpy as np

# Generate sample weights (inverse of target frequency)
target_counts = np.bincount(np.digitize(y_train, bins=10))
sample_weights = 1 / target_counts[np.digitize(y_train, bins=10)]

# Normalize weights
sample_weights /= np.sum(sample_weights)

# Train with sample weights
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
```

Slide 11: Ví dụ thực tế: Dự đoán giá nhà

Vui lòng sử dụng XGBoost để dự đoán giá dựa trên nhiều đặc điểm khác nhau như diện tích, số phòng ngủ và vị trí.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load house price data
house_data = pd.read_csv('house_prices.csv')

# Prepare features and target
X = house_data.drop('price', axis=1)
y = house_data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: ${mae:.2f}")
```

Trang trình chiếu 12: Ví dụ thực tế: Dự đoán giá cổ phiếu

Trong ví dụ này, chúng tôi sẽ sử dụng XGBoost để dự đoán giá cổ phiếu dựa trên dữ liệu lịch sử và chỉ báo kỹ thuật.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load stock data
stock_data = pd.read_csv('stock_data.csv')

# Calculate technical indicators (e.g., Moving Average)
stock_data['MA_7'] = stock_data['Close'].rolling(window=7).mean()
stock_data['MA_21'] = stock_data['Close'].rolling(window=21).mean()

# Prepare features and target
X = stock_data[['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_21']].dropna()
y = stock_data['Close'].dropna()

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: ${rmse:.2f}")
```

Trang trình bày 13: Lưu và tải mô hình XGBoost

Sau khi đào tạo mô hình XGBoost, bạn có thể mô phỏng hình đó để sử dụng trong tương lai mà không cần đào tạo lại. Điều này đặc biệt hữu ích cho việc phát triển các mô hình trong môi trường sản xuất.

```python
import joblib

# Save the model
joblib.dump(xgb_model, 'xgboost_model.joblib')

# Load the model
loaded_model = joblib.load('xgboost_model.joblib')

# Use the loaded model for predictions
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example features
prediction = loaded_model.predict(new_data)
print(f"Prediction using loaded model: {prediction[0]}")
```

Slide 14: Quyết định trực tiếp về hóa chất

Các mô hình XGBoost bao gồm nhiều cây quyết định. Những cây này có thể cung cấp cái nhìn sâu sắc về cách đưa ra mô hình được mong đợi.

```python
from xgboost import plot_tree
import matplotlib.pyplot as plt

# Plot the first tree
plt.figure(figsize=(20, 10))
plot_tree(xgb_model, num_trees=0)
plt.title('First Decision Tree in XGBoost Model')
plt.show()

# Plot feature importance
xgb.plot_importance(xgb_model)
plt.title('Feature Importance in XGBoost Model')
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để tìm hiểu thêm về quá trình khôi phục XGBoost:

1. Tài liệu XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
2. "XGBoost: Hệ thống tăng cường cây có thể mở rộng" của Chen và Guestrin (2016): [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
3. " Hướng dẫn toàn diện để hiểu toán học đằng sau XGBoost" của Aniruddha Bhandari: [https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-know-the-math-behind-xgboost/](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-know-the-math-behind-xgboost/)
