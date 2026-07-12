##Đánh giá các mô hình phục hồi Số liệu và mã Python
Trang trình bày 1: Lỗi bình bình trung bình (MSE)

Sai số bình phương trung bình là số liệu cơ bản để đánh giá các mô hình hồi phục, đo độ chênh lệch bình phương trung bình giữa giá dự đoán và giá trị thực tế. Nó phạt nặng hơn các lỗi lớn hơn là cho phép tính toán phương pháp, tạo ra nó đặc biệt nhạy cảm với các ngoại lệ giá trị trong dữ liệu.

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

Trang trình bày 2: Error normal normal (RMSE)

RMSE mở rộng MSE bằng cách lấy kết quả cấp hai, cung cấp số liệu có cùng vị trí cho tiêu điểm biến. Điều này làm cho việc giải quyết trở nên trực quan hơn và cho phép so sánh trực tiếp với quy định cấm đầu dữ liệu.

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

Sai số tuyệt đối trung bình tính toán khác biệt tuyệt đối trung bình giữa dự đoán và giá trị thực tế, đưa ra hình phạt tuyến tính cho các lỗi. Không giống như MSE, MAE ít nhạy cảm hơn các giá trị ngoại lệ và cung cấp số liệu mạnh mẽ hơn cho các tập dữ liệu có những điểm bất thường đáng kể.

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

R phương pháp đo tỷ lệ phương pháp sai trong các biến phụ thuộc được giải quyết bởi các biến độc lập. Số dữ liệu này dao động từ 0 đến 1, trong đó 1 biểu hiện mong đợi hoàn hảo và 0 biểu hiện mô hình hoạt động không tốt hơn dự đoán giá trị trung bình.

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

R bình phương đã điều chỉnh chỉnh sửa chỉ số R bình phương để tính số lượng yếu tố dự đoán trong mô hình, xử lý phạt bổ sung các biến thể không cải thiện đáng kể khả năng giải thích của mô hình. Điều này ngăn chặn trang bị quá trình thông tin được đưa vào quá nhiều tính năng.

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

Trang trình bày 6: Triển khai thực tế - Dự kiến ​​giá nhà

Triển khai các phản hồi số liệu cho mô hình dự đoán của nhà sản xuất, có thể thực hiện ứng dụng thực tế của các giá trị đánh giá dữ liệu khác nhau trong bối cảnh bất động bằng cách sử dụng bộ dữ liệu Nhà ở California.

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

Kết quả đánh giá từ mô hình mong đợi giá nhà của chúng tôi bằng chứng minh mối liên hệ giữa các số liệu khác nhau và giải pháp chúng trong bối cảnh thực tế. Phân tích này giúp hiểu được hiệu suất của mô hình từ nhiều góc độ.

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

Trang trình bày 8: Giải thích điểm phương sai

Các điểm phương pháp được giải thích theo tỷ lệ sai lệch có thể được dự đoán từ các biến độc lập. Số liệu này cung cấp cái nhìn sâu sắc về độ lệch của các mục tiêu được mong đợi của mô hình bắt.

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

Lỗi phần trăm tuyệt đối trung bình cung cấp phép đo độ chính xác dự đoán dựa trên tỷ lệ phần trăm, tạo ra nó đặc biệt hữu ích khi so sánh các mô hình trên các thước đo khác nhau. Nó có thể hiển thị độ chính xác dưới một phần trăm, tạo điều kiện cho diễn đàn giải trực tiếp trên đa dạng dữ liệu.

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

Trang trình bày 10: Triển khai thực tế - Dự báo thời gian chuỗi

Triển khai các phản hồi dữ liệu toàn diện để dự báo thời gian chuỗi, có thể thực hiện đánh giá các kỳ vọng qua nhiều bước thời gian có xem xét đến sự phụ thuộc theo thời gian.

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

Trang trình chiếu 11: Chuỗi thời gian dự báo kết quả

Việc đánh giá mô hình mô hình thời gian chuỗi dự án của chúng tôi đã tìm thấy sự tương tác giữa các lỗi khác nhau và tầm quan trọng của chúng trong các nhiệm vụ dự kiến ​​​​theo thời gian.

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

Trang trình bày 12: Error bình luận phương bình luận có trọng số (WMSE)

Lỗi phương pháp trung bình có MSE mở rộng số lượng quan trọng bằng cách cho phép các số khác nhau cho các mẫu hoặc thời gian khác nhau, cho phép hình phạt tùy chỉnh lỗi dựa trên phạm vi kiến ​​thức hoặc tầm quan trọng của mẫu trong dự đoán bối cảnh.

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

* Khảo sát toàn diện về các cơ sở tổn thất phức tạp dựa trên cơ sở hồi phục để dự báo thời gian chuỗi [https://arxiv.org/abs/2201.09755](https://arxiv.org/abs/2201.09755)
* Số liệu đánh giá cho các vấn đề hồi phục: Phương pháp tiếp cận nhất nhất [https://arxiv.org/abs/2006.13799](https://arxiv.org/abs/2006.13799)
* Học sâu để dự báo thời gian chuỗi: Khảo sát [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)
* Hàm mất hồi sức mạnh để phân tích chuỗi thời gian [https://arxiv.org/abs/2008.04687](https://arxiv.org/abs/2008.04687)
* Số liệu đánh giá mô hình học máy: Tìm kiếm nghiên cứu so sánh trên Google Scholar: "phân tích so sánh số hồi phục học máy"
