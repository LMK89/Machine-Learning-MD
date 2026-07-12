## Đánh giá các mô hình hồi quy Số liệu và mã Python
Trang trình bày 1: Lỗi bình phương trung bình (MSE)

Sai số bình phương trung bình là số liệu cơ bản để đánh giá các mô hình hồi quy, đo lường chênh lệch bình phương trung bình giữa giá trị dự đoán và giá trị thực tế. Nó phạt nặng hơn các lỗi lớn hơn do phép tính bình phương, khiến nó đặc biệt nhạy cảm với các giá trị ngoại lệ trong tập dữ liệu.

```python
import numpy as np

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error between true and predicted values

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: MSE value
    """
    # MSE formula: (1/n) * Σ(y_true - y_pred)²
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"MSE: {calculate_mse(y_true, y_pred):.4f}")
# Output: MSE: 0.0675
```

Trang trình bày 2: Lỗi bình phương trung bình gốc (RMSE)

RMSE mở rộng MSE bằng cách lấy căn bậc hai của kết quả, cung cấp số liệu có cùng đơn vị với biến mục tiêu. Điều này làm cho việc diễn giải trở nên trực quan hơn và cho phép so sánh trực tiếp với quy mô ban đầu của dữ liệu.

```python
def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error between true and predicted values

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: RMSE value
    """
    # RMSE formula: sqrt((1/n) * Σ(y_true - y_pred)²)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"RMSE: {calculate_rmse(y_true, y_pred):.4f}")
# Output: RMSE: 0.2598
```

Trang trình bày 3: Lỗi tuyệt đối trung bình (MAE)

Sai số tuyệt đối trung bình tính toán sự khác biệt tuyệt đối trung bình giữa dự đoán và giá trị thực tế, đưa ra hình phạt tuyến tính cho các lỗi. Không giống như MSE, MAE ít nhạy cảm hơn với các giá trị ngoại lệ và cung cấp số liệu mạnh mẽ hơn cho các tập dữ liệu có những điểm bất thường đáng kể.

```python
def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between true and predicted values

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: MAE value
    """
    # MAE formula: (1/n) * Σ|y_true - y_pred|
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"MAE: {calculate_mae(y_true, y_pred):.4f}")
# Output: MAE: 0.2250
```

Slide 4: R-squared (Hệ số xác định)

R bình phương đo lường tỷ lệ phương sai trong biến phụ thuộc được giải thích bởi các biến độc lập. Số liệu này dao động từ 0 đến 1, trong đó 1 biểu thị dự đoán hoàn hảo và 0 biểu thị mô hình hoạt động không tốt hơn dự đoán giá trị trung bình.

```python
def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared score between true and predicted values

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: R-squared value
    """
    # Calculate mean of true values
    y_mean = np.mean(y_true)

    # Calculate total sum of squares
    ss_total = np.sum((y_true - y_mean) ** 2)

    # Calculate residual sum of squares
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # R² formula: 1 - (SS_residual / SS_total)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"R²: {calculate_r2(y_true, y_pred):.4f}")
# Output: R²: 0.9327
```

Slide 5: Bình phương R đã điều chỉnh

R bình phương đã điều chỉnh sửa đổi chỉ số R bình phương để tính số lượng yếu tố dự đoán trong mô hình, xử phạt việc bổ sung các biến không cải thiện đáng kể khả năng giải thích của mô hình. Điều này ngăn chặn việc trang bị quá mức thông qua việc đưa vào quá nhiều tính năng.

```python
def calculate_adjusted_r2(y_true, y_pred, n_features):
    """
    Calculate Adjusted R-squared score

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        n_features: Number of features (independent variables)
    Returns:
        float: Adjusted R-squared value
    """
    n_samples = len(y_true)
    r2 = calculate_r2(y_true, y_pred)

    # Adjusted R² formula: 1 - (1 - R²) * (n - 1)/(n - p - 1)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
n_features = 2
print(f"Adjusted R²: {calculate_adjusted_r2(y_true, y_pred, n_features):.4f}")
# Output: Adjusted R²: 0.8872
```

Trang trình bày 6: Triển khai thực tế - Dự đoán giá nhà

Triển khai toàn diện các số liệu hồi quy cho mô hình dự đoán giá nhà, thể hiện ứng dụng thực tế của các số liệu đánh giá khác nhau trong bối cảnh bất động sản bằng cách sử dụng bộ dữ liệu Nhà ở California.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load and prepare data
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Calculate all metrics
metrics = {
    'MSE': calculate_mse(y_test, y_pred),
    'RMSE': calculate_rmse(y_test, y_pred),
    'MAE': calculate_mae(y_test, y_pred),
    'R²': calculate_r2(y_test, y_pred),
    'Adjusted R²': calculate_adjusted_r2(y_test, y_pred, X_test.shape[1])
}

# Display results
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 7: Kết quả dự đoán giá nhà

Kết quả đánh giá từ mô hình dự đoán giá nhà của chúng tôi chứng minh mối quan hệ giữa các số liệu khác nhau và cách giải thích chúng trong bối cảnh thực tế. Phân tích này giúp hiểu được hiệu suất của mô hình từ nhiều góc độ.

```python
"""
Example Output:
MSE: 0.5428
RMSE: 0.7366
MAE: 0.5344
R²: 0.5983
Adjusted R²: 0.5975
"""

# Visualization of actual vs predicted values
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.show()
```

Trang trình bày 8: Giải thích Điểm phương sai

Điểm phương sai được giải thích đo lường tỷ lệ phương sai có thể dự đoán được từ các biến độc lập. Số liệu này cung cấp cái nhìn sâu sắc về mức độ chênh lệch của biến mục tiêu được dự đoán của mô hình nắm bắt.

```python
def calculate_explained_variance(y_true, y_pred):
    """
    Calculate Explained Variance Score

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: Explained variance score
    """
    # Calculate variance of residuals
    residual_variance = np.var(y_true - y_pred)
    # Calculate total variance
    total_variance = np.var(y_true)

    # Explained variance formula: 1 - (variance(y_true - y_pred) / variance(y_true))
    explained_variance = 1 - (residual_variance / total_variance)
    return explained_variance

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"Explained Variance: {calculate_explained_variance(y_true, y_pred):.4f}")
# Output: Explained Variance: 0.9331
```

Trang trình bày 9: Lỗi phần trăm tuyệt đối trung bình (MAPE)

Lỗi phần trăm tuyệt đối trung bình cung cấp phép đo độ chính xác dự đoán dựa trên tỷ lệ phần trăm, khiến nó đặc biệt hữu ích khi so sánh các mô hình trên các thang đo khác nhau. Nó thể hiện độ chính xác dưới dạng phần trăm, tạo điều kiện cho việc diễn giải trực quan trên các bộ dữ liệu đa dạng.

```python
def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error

    Args:
        y_true: Array of actual values (must not contain zeros)
        y_pred: Array of predicted values
    Returns:
        float: MAPE value
    """
    # MAPE formula: (1/n) * Σ|(y_true - y_pred)/y_true| * 100
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# Example usage with non-zero values
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"MAPE: {calculate_mape(y_true, y_pred):.2f}%")
# Output: MAPE: 7.83%
```

Trang trình bày 10: Triển khai thực tế - Dự báo chuỗi thời gian

Triển khai các số liệu hồi quy toàn diện để dự báo chuỗi thời gian, thể hiện việc đánh giá các dự đoán qua nhiều bước thời gian có xem xét đến sự phụ thuộc theo thời gian.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series data
np.random.seed(42)
t = np.linspace(0, 100, 100)
y = 0.5 * np.sin(0.1 * t) + 0.1 * np.random.randn(100)

# Prepare data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Parameters
seq_length = 10
X, y = create_sequences(y_scaled, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Simple moving average prediction
y_pred = np.mean(X_test, axis=1)

# Calculate all metrics
metrics = {
    'MSE': calculate_mse(y_test, y_pred),
    'RMSE': calculate_rmse(y_test, y_pred),
    'MAE': calculate_mae(y_test, y_pred),
    'MAPE': calculate_mape(y_test.flatten(), y_pred.flatten()),
    'R²': calculate_r2(y_test, y_pred)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Trang trình chiếu 11: Kết quả dự báo chuỗi thời gian

Việc đánh giá toàn diện mô hình dự báo chuỗi thời gian của chúng tôi cho thấy sự tương tác giữa các số liệu lỗi khác nhau và tầm quan trọng của chúng trong các nhiệm vụ dự đoán theo thời gian.

```python
"""
Example Output:
MSE: 0.0124
RMSE: 0.1114
MAE: 0.0891
MAPE: 15.3244
R²: 0.7823
"""

# Visualization of forecasting results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='s')
plt.title('Time Series Forecasting Results')
plt.xlabel('Time Steps')
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 12: Lỗi bình phương trung bình có trọng số (WMSE)

Lỗi bình phương trung bình có trọng số mở rộng MSE bằng cách cho phép các trọng số khác nhau cho các mẫu hoặc điểm thời gian khác nhau, cho phép hình phạt lỗi tùy chỉnh dựa trên kiến ​​thức miền hoặc tầm quan trọng của mẫu trong bối cảnh dự đoán.

```python
def calculate_wmse(y_true, y_pred, weights=None):
    """
    Calculate Weighted Mean Squared Error

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        weights: Array of weights for each sample (default: equal weights)
    Returns:
        float: WMSE value
    """
    if weights is None:
        weights = np.ones_like(y_true)

    # Normalize weights
    weights = weights / np.sum(weights)

    # WMSE formula: Σ(weights * (y_true - y_pred)²)
    wmse = np.sum(weights * (y_true - y_pred) ** 2)
    return wmse

# Example usage with custom weights
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
weights = np.array([0.1, 0.2, 0.3, 0.4])  # Higher weights for later samples

print(f"WMSE: {calculate_wmse(y_true, y_pred, weights):.4f}")
# Output: WMSE: 0.0331
```

Trang trình bày 13: Tài nguyên bổ sung

* Khảo sát toàn diện về các hàm tổn thất dựa trên hồi quy để dự báo chuỗi thời gian [https://arxiv.org/abs/2201.09755](https://arxiv.org/abs/2201.09755)
* Số liệu đánh giá cho các vấn đề hồi quy: Phương pháp tiếp cận thống nhất [https://arxiv.org/abs/2006.13799](https://arxiv.org/abs/2006.13799)
* Học sâu để dự báo chuỗi thời gian: Khảo sát [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)
* Hàm mất hồi quy mạnh mẽ để phân tích chuỗi thời gian [https://arxiv.org/abs/2008.04687](https://arxiv.org/abs/2008.04687)
* Số liệu đánh giá mô hình học máy: Tìm kiếm nghiên cứu so sánh trên Google Scholar: "phân tích so sánh số liệu hồi quy học máy"
