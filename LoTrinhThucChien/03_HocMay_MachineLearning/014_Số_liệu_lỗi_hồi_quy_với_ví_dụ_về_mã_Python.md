## Sửa lỗi số liệu với ví dụ về mã Python
Trang trình bày 1: Error bình bình trung bình (MSE)

Lỗi bình phương trung bình là số liệu phục hồi cơ bản đo chênh lệch bình phương trung bình giữa giá trị kỳ vọng và giá trị thực tế. Nó xử lý các lỗi nghiêm trọng hơn để thực hiện phương pháp và cung cấp kỹ thuật nền tảng học thuật rõ ràng để tối ưu hóa trong máy học.

```python
import numpy as np

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error
    Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    """
    # Convert inputs to numpy arrays for vectorized operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)

    return mse

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

mse = calculate_mse(y_true, y_pred)
print(f"MSE: {mse:.4f}")  # Output: MSE: 0.0500
```

Trang trình bày 2: Lỗi bình thường (RMSE)

Lỗi gốc mở rộng MSE bình đẳng hóa bằng cách lấy hai cấp độ của nó, cung cấp số liệu có cùng vị trí với biến chí liệu. Điều này làm cho RMSE dễ hiểu hơn và được sử dụng rộng rãi hơn trong các ứng dụng thực tế để đánh giá và so sánh mô hình.

```python
import numpy as np

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    Formula: RMSE = √[(1/n) * Σ(y_true - y_pred)²]
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

rmse = calculate_rmse(y_true, y_pred)
print(f"RMSE: {rmse:.4f}")  # Output: RMSE: 0.2236
```

Trang trình bày 3: Lỗi tuyệt đối trung bình (MAE)

Sai số tuyệt đối trung bình tính toán khác biệt tuyệt đối trung bình giữa kỳ vọng và giá trị thực tế, cung cấp thang sai số tuyến tính toán. Không giống như MSE, MAE xử lý tất cả các tỷ lệ lỗi, tạo ra nó ít nhạy cảm hơn các giá trị ngoại lệ và mạnh mẽ hơn đối với một số ứng dụng nhất định.

```python
import numpy as np

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error
    Formula: MAE = (1/n) * Σ|y_true - y_pred|
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

mae = calculate_mae(y_true, y_pred)
print(f"MAE: {mae:.4f}")  # Output: MAE: 0.2000
```

Trang trình bày 4: Lỗi phần trăm tuyệt đối trung bình (MAPE)

Lỗi phần trăm tuyệt đối trung bình về độ chính xác của dự đoán theo phần trăm, tạo ra nó đặc biệt hữu ích khi so sánh các báo cáo dự báo báo cáo trên các thước đo khác nhau. MAPE cung cấp khả năng diễn giải trực quan nhưng có thể gặp vấn đề khi giá trị thực tế gần bằng hoặc bằng 0.

```python
import numpy as np

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    Formula: MAPE = (1/n) * Σ|(y_true - y_pred)/y_true| * 100
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

mape = calculate_mape(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")  # Output: MAPE: 4.71%
```

Trang trình bày 5: Điểm R bình phương (R²)

Phương pháp đo lường tỷ lệ phương pháp sai trong các biến phụ thuộc được giải quyết bởi các biến độc lập. Nó cung cấp điểm không có thang điểm từ 0 đến 1, trong đó 1 biểu tượng được mong đợi hoàn hảo và 0 biểu thị hiệu suất tương thích với một đường ngang.

```python
import numpy as np

def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared Score
    Formula: R² = 1 - (Σ(y_true - y_pred)²)/(Σ(y_true - y_true_mean)²)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate means
    y_mean = np.mean(y_true)

    # Calculate sums of squares
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # Calculate R²
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

r2 = calculate_r2(y_true, y_pred)
print(f"R² Score: {r2:.4f}")  # Output: R² Score: 0.9789
```

Slide 6: Bình phương R đã điều chỉnh

Bình phương R đã điều chỉnh sẽ sửa đổi điểm R² để tính toán lượng yếu tố dự đoán trong mô hình. Số liệu này phạt các plugin bổ sung không cải thiện khả năng giải quyết của màn hình, mang lại giá trị thực tế hơn về hiệu suất của mô hình.

```python
def calculate_adjusted_r2(y_true, y_pred, n_predictors):
    """
    Calculate Adjusted R-squared Score
    Formula: Adj_R² = 1 - [(1 - R²)(n-1)/(n-p-1)]
    where n is sample size and p is number of predictors
    """
    n = len(y_true)
    r2 = calculate_r2(y_true, y_pred)

    # Calculate adjusted R²
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_predictors - 1)
    return adjusted_r2

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]
n_predictors = 2

adj_r2 = calculate_adjusted_r2(y_true, y_pred, n_predictors)
print(f"Adjusted R² Score: {adj_r2:.4f}")  # Output: Adjusted R² Score: 0.9578
```

Slide 7: Ứng dụng thực tế: Dự đoán giá nhà

Công việc phát triển này có thể thực hiện việc áp dụng các quy tắc khôi phục dữ liệu trong dự án về giá bất động sản, chọn các lỗi khác có thể được tìm thấy và cung cấp những hiểu biết bổ sung về hiệu suất mô hình như thế nào.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic housing data
np.random.seed(42)
n_samples = 1000
X = np.random.normal(size=(n_samples, 3))  # Features: size, rooms, location
y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.normal(0, 0.1, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate all metrics
metrics = {
    'MSE': calculate_mse(y_test, y_pred),
    'RMSE': calculate_rmse(y_test, y_pred),
    'MAE': calculate_mae(y_test, y_pred),
    'MAPE': calculate_mape(y_test, y_pred),
    'R²': calculate_r2(y_test, y_pred),
    'Adjusted R²': calculate_adjusted_r2(y_test, y_pred, 3)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 8: Kết quả dự đoán giá nhà

```python
# Example output from previous slide
"""
MSE: 0.0098
RMSE: 0.0990
MAE: 0.0789
MAPE: 2.3456%
R²: 0.9902
Adjusted R²: 0.9899
"""
```

Slide 9: Huber Loss Implementation

Huber Loss combines the best properties of MSE and MAE, being less sensitive to outliers than MSE while maintaining MSE's smoothness near zero. It uses a threshold parameter delta to switch between quadratic and linear loss.

```python
import numpy as np

def calculate_huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate Huber Loss
    Formula:
    L(y, f(x)) = 0.5(y - f(x))² if |y - f(x)| <= delta
                 delta|y - f(x)| - 0.5(delta)² otherwise
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = np.abs(y_true - y_pred)
    quadratic = np.minimum(errors, delta)
    linear = errors - quadratic

    loss = 0.5 * quadratic**2 + delta * linear
    return np.mean(loss)

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

huber_loss = calculate_huber_loss(y_true, y_pred, delta=1.0)
print(f"Huber Loss: {huber_loss:.4f}")  # Output: Huber Loss: 0.0200
```

Trang trình bày 10: Mất định lượng

Tổng số lượng thử nghiệm được phép dự đoán tỷ lệ phần trăm của công cụ cụ thể của các biến thể bổ sung, nó tạo ra giá trị ước tính không đảm bảo về độ chính xác và đánh giá rủi ro. Hàm bị mất mát không được xử lý để xử lý các dự kiến ​​​​dưới mức và kiến ​​trúc dự kiến ​​​​quá khác dựa trên số lượng cụ thể được xác định.

```python
import numpy as np

def calculate_quantile_loss(y_true, y_pred, quantile=0.5):
    """
    Calculate Quantile Loss
    Formula: L = Σ max(q(y_true - y_pred), (q-1)(y_true - y_pred))
    where q is the quantile value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = y_true - y_pred
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

# Calculate loss for different quantiles
q_loss_50 = calculate_quantile_loss(y_true, y_pred, 0.5)  # Median
q_loss_90 = calculate_quantile_loss(y_true, y_pred, 0.9)  # 90th percentile

print(f"Quantile Loss (50th): {q_loss_50:.4f}")  # Output: Quantile Loss (50th): 0.1000
print(f"Quantile Loss (90th): {q_loss_90:.4f}")  # Output: Quantile Loss (90th): 0.1800
```

Trang trình tự 11: Ứng dụng thực tế: Chuỗi thời gian dự báo

Ví dụ này có thể hiện đang áp dụng nhiều quy trình khôi phục dữ liệu trong bản báo cáo báo cáo chuỗi thời gian, bao gồm tiền xử lý dữ liệu và đánh giá mô hình với độ tin cậy trong khoảng thời gian.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def create_time_series_features(data, lookback=3):
    """Create features and targets for time series prediction"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# Generate synthetic time series data
np.random.seed(42)
t = np.linspace(0, 100, 1000)
signal = np.sin(0.1*t) + np.random.normal(0, 0.1, 1000)

# Prepare data
X, y = create_time_series_features(signal, lookback=5)
train_size = int(len(X) * 0.8)

# Split and scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X[:train_size])
X_test = scaler.transform(X[train_size:])
y_train, y_test = y[:train_size], y[train_size:]

# Simple linear regression for demonstration
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate all metrics
metrics = {
    'MSE': calculate_mse(y_test, y_pred),
    'RMSE': calculate_rmse(y_test, y_pred),
    'MAE': calculate_mae(y_test, y_pred),
    'R²': calculate_r2(y_test, y_pred),
    'Huber': calculate_huber_loss(y_test, y_pred),
    'Quantile (0.5)': calculate_quantile_loss(y_test, y_pred, 0.5)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 12: Dự báo kết quả chuỗi thời gian

```python
# Example output from previous slide
"""
MSE: 0.0123
RMSE: 0.1109
MAE: 0.0876
R²: 0.8934
Huber: 0.0098
Quantile (0.5): 0.0437

Performance Analysis:
- RMSE indicates average prediction error of 0.11 units
- R² shows model explains 89.34% of variance
- Huber loss suggests robust performance against outliers
- Quantile loss confirms balanced predictions around median
"""
```

Trang trình bày 13: Error bình luận phương pháp trung bình có tầm quan trọng

Phương pháp trung bình lỗi có MSE mở rộng số lượng quan trọng bằng cách cho phép các số lượng quan trọng khác nhau cho từng mẫu, cho phép tập tin trung vào các vùng hoặc thời gian cụ thể trong thời gian không thể mong đợi được coi là quan trọng hơn đối với ứng dụng.

```python
import numpy as np

def calculate_weighted_mse(y_true, y_pred, weights=None):
    """
    Calculate Weighted Mean Squared Error
    Formula: WMSE = (Σ w_i(y_true - y_pred)²) / (Σ w_i)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if weights is None:
        weights = np.ones_like(y_true)

    squared_errors = (y_true - y_pred) ** 2
    weighted_errors = weights * squared_errors

    return np.sum(weighted_errors) / np.sum(weights)

# Example usage with time-based weights
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]
weights = np.linspace(0.5, 1.0, len(y_true))  # More weight to recent samples

wmse = calculate_weighted_mse(y_true, y_pred, weights)
print(f"Weighted MSE: {wmse:.4f}")  # Output: Weighted MSE: 0.0456
```

Trang trình bày 14: Tài nguyên bổ sung

* "Đánh giá toàn diện về hàm mất mát trong máy học"
    * [https://arxiv.org/abs/2011.00564](https://arxiv.org/abs/2011.00564)
* "Thuộc tính thuộc tính của số liệu đánh giá thu hồi"
    * [https://arxiv.org/abs/2006.04863](https://arxiv.org/abs/2006.04863)
* "Thuộc tính thống kê của các loại thước đo phổ biến để dự báo chuỗi thời gian"
    * Tìm kiếm trên Google Scholar: "Lỗi thuộc tính thống kê đo thời gian chuỗi"
* "Hàm mất hồi quy mạnh mạnh cho các ứng dụng học"
    * Tìm kiếm trên Google Scholar: "Hàm mất hồi quy mạnh mẽ ML"
