##Đánh giá các số liệu của quy trình khôi phục mô hình trong Python
Trang trình bày 1: Đánh giá hiệu suất của mô hình phục hồi

Mô hình hồi quy là công cụ thiết yếu trong phân tích dự đoán. Để đảm bảo hiệu quả của chúng, chúng tôi cần những số liệu đáng tin cậy để đánh giá hiệu suất của chúng. Phần trình bày này sẽ khám phá các số liệu đánh giá chính xác cho các mô hình hồi phục, bao gồm các lỗi bình phương trung bình (MSE), lỗi bình phương trung bình gốc (RMSE), R bình phương (R²) và R bình phương đã điều chỉnh. Chúng tôi sẽ trình bày cách phát triển các số liệu này bằng Python, đồng thời cung cấp các ví dụ thực tế trong quá trình thực hiện.

```python
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Fit a linear regression model
model = LinearRegression().fit(X, y)

# Make predictions
y_pred = model.predict(X)

# We'll use this data to calculate our metrics
```

Trang trình bày 2: Lỗi bình bình trung bình (MSE)

Sai số phương pháp trung bình là số liệu cơ bản đo độ chênh lệch phương pháp trung bình giữa giá trị dự đoán và giá trị thực tế. Nó phạt nặng hơn các lỗi lớn hơn là hoạt động bình phương. MSE thấp hơn cho thấy hiệu quả hoạt động tốt hơn.

```python
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

mse = calculate_mse(y, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Using sklearn
mse_sklearn = mean_squared_error(y, y_pred)
print(f"MSE (sklearn): {mse_sklearn:.4f}")
```

Trang trình bày 3: Lỗi bình luận gốc trung bình (RMSE)

RMSE là cấp hai của MSE. Nó cung cấp lỗi số trong cùng một đơn vị cho mục tiêu biến đổi, làm cho nó dễ hiểu hơn. Giống như MSE, RMSE thấp hơn cho thấy hiệu suất hoạt động tốt hơn.

```python
def calculate_rmse(y_true, y_pred):
    return np.sqrt(calculate_mse(y_true, y_pred))

rmse = calculate_rmse(y, y_pred)
print(f"Root Mean Squared Error: {rmse:.4f}")

# Using sklearn
rmse_sklearn = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE (sklearn): {rmse_sklearn:.4f}")
```

Trang trình bày 4: R bình phương (R²)

R-squared, còn được gọi là hệ số xác định, phương pháp đo tỷ lệ sai trong các biến phụ thuộc có thể dự đoán được từ (các) biến độc lập. Nó nằm trong khoảng từ 0 đến 1, với 1 biểu hiện được mong đợi hoàn hảo và 0 biểu thị rằng hoạt động không tốt hơn một đường ngang.

```python
def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

r2 = calculate_r2(y, y_pred)
print(f"R-squared: {r2:.4f}")

# Using sklearn
r2_sklearn = r2_score(y, y_pred)
print(f"R-squared (sklearn): {r2_sklearn:.4f}")
```

Slide 5: Bình phương R đã điều chỉnh

Bình phương R đã điều chỉnh sẽ sửa đổi bình phương R bằng cách xử lý phạt bổ sung các yếu tố dự đoán không liên quan vào mô hình. Nó đặc biệt hữu ích khi so sánh các mô hình với số lượng yếu tố dự đoán khác nhau.

```python
def calculate_adjusted_r2(y_true, y_pred, n_features):
    r2 = calculate_r2(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

adj_r2 = calculate_adjusted_r2(y, y_pred, X.shape[1])
print(f"Adjusted R-squared: {adj_r2:.4f}")
```

Trang trình bày 6: Ví dụ thực tế: Dự đoán giá nhà ở

Vui lòng áp dụng các dữ liệu này vào kịch bản thực tế để dự đoán giá nhà dựa trên các đặc điểm khác nhau như diện tích, số phòng ngủ, v.v.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression().fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = calculate_adjusted_r2(y_test, y_pred, X.shape[1])

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
```

Slide 7: kết quả giải thích

Các số liệu mà chúng tôi đã tính toán cung cấp thông tin chi tiết về hiệu suất của mô hình của chúng tôi. MSE và RMSE low cho thấy sự mong đợi của chúng tôi gần với giá trị thực tế. Giá trị bình phương R cho biết mức độ chênh lệch trong giá đất mà chúng tôi giải thích. Bình phương R được điều chỉnh giúp chúng tôi biết liệu chúng tôi có trang bị quá mạnh hay không bằng cách bổ sung quá nhiều tính năng.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Housing Prices")
plt.tight_layout()
plt.show()
```

Trang trình bày 8: Error tuyệt đối trung bình (MAE)

Sai số tuyệt đối trung bình là một thước đo hữu ích khác được sử dụng để đo tốc độ sai trung bình trong một tập hợp các kỳ vọng mà không cần xem xét hướng dẫn của chúng. Nó ít nhạy cảm hơn các ngoại lệ so với MSE và RMSE.

```python
from sklearn.metrics import mean_absolute_error

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

mae = calculate_mae(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# Using sklearn
mae_sklearn = mean_absolute_error(y_test, y_pred)
print(f"MAE (sklearn): {mae_sklearn:.4f}")
```

Slide 9: Ví dụ thực tế: Dự đoán giá cổ phiếu

Vui lòng áp dụng số liệu của chúng tôi vào một vấn đề thực tế khác: dự đoán giá cổ phiếu dựa trên dữ liệu lịch sử và các số liệu chính khác nhau.

```python
import pandas as pd
import yfinance as yf

# Download stock data (using Apple Inc. as an example)
stock_data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

# Prepare features and target
stock_data['Returns'] = stock_data['Close'].pct_change()
stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data = stock_data.dropna()

X = stock_data[['Returns', 'MA_5', 'MA_20']]
y = stock_data['Close'].shift(-1).dropna()

# Align X and y
X = X.iloc[:-1]
y = y.iloc[:-1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model and make predictions
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")
```

Trang trình bày 10: Xác thực chéo để đánh giá mô hình

Xác thực chéo là một kỹ thuật mạnh mẽ để đánh giá các thống kê phân tích kết quả sẽ độc đáo như thế nào đối với một tập dữ liệu độc lập. Nó đặc biệt hữu ích khi bạn có một chế độ hạn chế dữ liệu.

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(LinearRegression(), X, y, cv=5,
                            scoring='neg_mean_squared_error')

# Convert MSE to RMSE
rmse_scores = np.sqrt(-cv_scores)

print("Cross-validated RMSE scores:", rmse_scores)
print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
print(f"Standard deviation of RMSE: {np.std(rmse_scores):.4f}")
```

Slide 11: Phân tích dư lượng

Phân tích phần dư là rất quan trọng để xác định các giả định của tính năng khôi phục tuyến tính. Nó liên kết đến các công việc kiểm tra khác nhau giữa các giá trị được khảo sát và dự kiến.

```python
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# Q-Q plot for normality check
from scipy import stats

fig, ax = plt.subplots(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

Trang trình bày 12: Tầm quan trọng của tính năng

Việc hiểu những tính năng nào đóng góp nhiều nhất vào dự đoán của mô hình của bạn có thể cung cấp những tính năng hiểu biết sâu sắc có giá trị. Đối với việc khôi phục tính năng tuyến tính, chúng tôi có thể kiểm tra các hệ thống.

```python
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

Trình bày 13: Trang bị quá trình độ và trang bị thiếu trang

So sánh các huấn luyện lỗi và kiểm tra có thể giúp phát hiện trang quá đủ hoặc thiếu trang. Nếu huấn luyện có nhiều lỗi hơn nên kiểm tra lỗi thì mô hình có thể quá khớp.

```python
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")

# Learning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation error')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve')
plt.legend()
plt.tight_layout()
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về đánh giá mô hình phục hồi và các chủ đề liên quan, đây là một số tài nguyên có giá trị:

1. "Khảo sát các quy trình mô xác thực chéo để lựa chọn hình ảnh" của Sylvain Arlot và Alain Celisse (2010). Có tại: [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
2. “Thành co rút hồi quy và đơn vị thông qua Lasso” của Robert Tibshirani (1996). Có tại: [https://arxiv.org/abs/math/9508054](https://arxiv.org/abs/math/9508054)
3. "Giới thiệu về học thống kê" của Gareth James, Daniela Witten, Trevor Hastie và Robert Tibshirani. Cuốn sách này cung cấp một cái nhìn tổng thể dễ dàng tiếp cận về các phương pháp học thống kê với các ứng dụng trong R.

Tài nguyên này cung cấp các thảo luận chuyên sâu về kỹ thuật đánh giá mô hình, phương pháp phục hồi nâng cao và nguyên tắc thống kê có thể nâng cao hiểu biết của bạn về phân tích phục hồi và đánh giá hiệu suất mô hình.
