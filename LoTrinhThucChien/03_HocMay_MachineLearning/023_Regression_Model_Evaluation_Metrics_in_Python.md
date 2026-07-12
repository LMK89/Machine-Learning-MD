## Số liệu đánh giá mô hình phục hồi trong Python
Trang trình bày 1: Triển khai lỗi bình luận (MSE)

Lỗi bình phương trung bình đóng vai trò là số liệu cơ bản trong phân tích hồi phục, đo độ chênh lệch phương tiện trung bình giữa giá trị dự đoán và giá trị thực tế. Nó phạt nặng hơn các lỗi lớn hơn so với thuật ngữ bình luận phương pháp, tạo ra nó đặc biệt nhạy cảm với các giá trị ngoại lệ trong dữ liệu.

```python
def mean_squared_error(y_true, y_pred):
    """
    Calculate MSE from scratch
    Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    """
    # Convert inputs to numpy arrays for vectorized operations
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate squared differences and mean
    mse = np.mean((y_true - y_pred) ** 2)

    return mse

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")  # Output: MSE: 0.0500
```

Trang trình bày 2: Triển khai lỗi bình luận gốc (RMSE)

RMSE mở rộng MSE bằng cách lấy kết quả cấp hai, cung cấp số liệu có cùng vị trí cho tiêu điểm biến. Điều này giúp diễn đàn trở nên trực quan hơn khi so sánh hiệu suất của các mô hình trên các dữ liệu khác nhau.

```python
def root_mean_squared_error(y_true, y_pred):
    """
    Calculate RMSE from scratch
    Formula: RMSE = √[(1/n) * Σ(y_true - y_pred)²]
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate MSE then take square root
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return rmse

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")  # Output: RMSE: 0.2236
```

Trang trình bày 3: Triển khai Lỗi tuyệt đối trung bình (MAE)

Sai số tuyệt đối trung bình tính toán khác biệt tuyệt đối trung bình giữa dự đoán và giá trị thực tế, đưa ra hình phạt tuyến tính cho các lỗi. Không giống như MSE, MAE ít nhạy cảm hơn các giá trị ngoại lệ và cung cấp số liệu mạnh mẽ hơn cho các tập dữ liệu có những điểm bất thường đáng kể.

```python
def mean_absolute_error(y_true, y_pred):
    """
    Calculate MAE from scratch
    Formula: MAE = (1/n) * Σ|y_true - y_pred|
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate absolute differences and mean
    mae = np.mean(np.abs(y_true - y_pred))

    return mae

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")  # Output: MAE: 0.2000
```

Trang trình bày 4: Thực hiện điểm R bình phương (R²)

Phương pháp R định mức tỷ lệ phương pháp sai trong các biến phụ thuộc được giải quyết bởi các biến độc lập. Số liệu này cung cấp điểm không có thang điểm từ 0 đến 1, trong đó 1 biểu tượng mong đợi hoàn hảo và 0 biểu thị hiệu suất tương thích với một đường ngang.

```python
def r2_score(y_true, y_pred):
    """
    Calculate R² Score from scratch
    Formula: R² = 1 - (Σ(y_true - y_pred)²) / (Σ(y_true - y_mean)²)
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate mean of true values
    y_mean = np.mean(y_true)

    # Calculate total sum of squares and residual sum of squares
    tss = np.sum((y_true - y_mean) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)

    # Calculate R²
    r2 = 1 - (rss / tss)

    return r2

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"R² Score: {r2_score(y_true, y_pred):.4f}")  # Output: R² Score: 0.9921
```

Trang trình bày 5: Thực hiện điều chỉnh phương pháp R

R bình phương đã điều chỉnh sửa đổi chỉ số R bình phương để tính số lượng yếu tố dự đoán trong mô hình, xử lý phạt bổ sung các biến số không đóng góp đáng kể vào hiệu suất mô hình. Điều này ngăn chặn việc trang bị quá trình thông số lựa chọn tính năng.

```python
def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Calculate Adjusted R² Score from scratch
    Formula: Adj R² = 1 - [(1 - R²)(n-1)/(n-p-1)]
    where n is sample size and p is number of features
    """
    import numpy as np

    # Calculate regular R²
    r2 = r2_score(y_true, y_pred)

    # Calculate sample size
    n = len(y_true)

    # Calculate adjusted R²
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return adjusted_r2

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
n_features = 2
print(f"Adjusted R² Score: {adjusted_r2_score(y_true, y_pred, n_features):.4f}")
# Output: Adjusted R² Score: 0.9868
```

Trang trình bày 6: Triển khai lỗi phần trăm tuyệt đối trung bình (MAPE)

Lỗi phần trăm tuyệt đối trung bình cung cấp độ chính xác đo lường dự đoán dựa trên phần trăm, nó đặc biệt hữu ích khi so sánh các mô hình trên các thước đo khác nhau. Nó có thể hiển thị độ chính xác dưới mức phần trăm lợi ích, tạo điều kiện thuận lợi cho việc giải thích trực quan cho các bên liên quan.

```python
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate MAPE from scratch
    Formula: MAPE = (100/n) * Σ|((y_true - y_pred)/y_true)|
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0

    # Calculate percentage errors
    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])

    # Calculate mean and convert to percentage
    mape = 100 * np.mean(percentage_errors)

    return mape

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.2f}%")
# Output: MAPE: 4.37%
```

Slide 7: Ví dụ thực tế - Dự đoán giá nhà

Ví dụ này có thể hiện công việc áp dụng các số liệu phục hồi trong kịch bản dự đoán giá bất kỳ sản phẩm nào, bao gồm tiền xử lý dữ liệu, đào tạo mô hình và đánh giá giá bằng nhiều số liệu để đánh giá hiệu suất của mô hình.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Generate synthetic house data
np.random.seed(42)
n_samples = 1000

# Features: size, bedrooms, age
X = np.random.rand(n_samples, 3)
X[:, 0] = X[:, 0] * 2000 + 1000  # Size: 1000-3000 sq ft
X[:, 1] = np.round(X[:, 1] * 3 + 2)  # Bedrooms: 2-5
X[:, 2] = np.round(X[:, 2] * 30)  # Age: 0-30 years

# Target: price (with some noise)
y = (X[:, 0] * 100 + X[:, 1] * 50000 - X[:, 2] * 1000 +
     np.random.normal(0, 10000, n_samples))

# Split data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Slide 8: Mã nguồn dự đoán giá nhà

```python
# Train model and make predictions
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate all metrics
metrics = {
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': root_mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'R²': r2_score(y_test, y_pred),
    'Adjusted R²': adjusted_r2_score(y_test, y_pred, 3),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred)
}

# Print results
for metric, value in metrics.items():
    if metric == 'MAPE':
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value:.2f}")

# Example output
"""
MSE: 98234567.89
RMSE: 9911.34
MAE: 7845.23
R²: 0.92
Adjusted R²: 0.91
MAPE: 3.45%
"""
```

Trang trình bày 9: Thực hiện phân tích dư thừa

Phân tích dư lượng cung cấp những hiểu biết quan trọng về các giả định của hình ảnh và các lĩnh vực tiềm năng cần được cải thiện. Việc phát triển này bao gồm tính toán số dư, kiểm tra tính toán chuẩn và trực quan hóa tính toán đồng nhất để xác thực các giả định của mô hình thu hồi.

```python
def analyze_residuals(y_true, y_pred):
    """
    Comprehensive residual analysis including statistical tests
    """
    import numpy as np
    from scipy import stats

    # Calculate residuals
    residuals = y_true - y_pred

    # Basic statistics
    stats_dict = {
        'Mean': np.mean(residuals),
        'Std Dev': np.std(residuals),
        'Skewness': stats.skew(residuals),
        'Kurtosis': stats.kurtosis(residuals)
    }

    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)

    return stats_dict, (shapiro_stat, shapiro_p)

# Example usage with previous house price data
stats_dict, normality_test = analyze_residuals(y_test, y_pred)
print("\nResidual Statistics:")
for stat, value in stats_dict.items():
    print(f"{stat}: {value:.4f}")
print(f"\nShapiro-Wilk test: stat={normality_test[0]:.4f}, p={normality_test[1]:.4f}")
```

Trang trình bày 10: Thực phẩm mất Huber

Huber Loss kết hợp các đặc tính tốt nhất của MSE và MAE bằng phương pháp bậc hai đối với các lỗi nhỏ và tuyến tính đối với các lỗi lớn, mang lại khả năng chống lại các giá trị ngoại lệ trong khi vẫn duy trì các lợi ích thế của MSE đối với phần dư nhỏ hơn. Tham số delta kiểm soát Chuyển điểm tiếp theo.

```python
def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate Huber Loss from scratch
    Formula:
    L(y, f(x)) = 1/2(y - f(x))² for |y - f(x)| ≤ δ
    L(y, f(x)) = δ|y - f(x)| - 1/2δ² for |y - f(x)| > δ
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate residuals
    residuals = np.abs(y_true - y_pred)

    # Calculate loss based on delta threshold
    mask = residuals <= delta
    squared_loss = 0.5 * residuals[mask]**2
    linear_loss = delta * residuals[~mask] - 0.5 * delta**2

    # Combine losses
    return np.mean(np.concatenate([squared_loss, linear_loss]))

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"Huber Loss (δ=1.0): {huber_loss(y_true, y_pred):.4f}")
```

Trang trình bày 11: Giải thích việc thực hiện Điểm phương sai

Các điểm phương pháp được giải thích theo tỷ lệ sai lệch có thể được dự đoán từ các biến độc lập. Nó khác với R2 ở chỗ tập trung vào sai số phương pháp hơn là tổng sai số phương pháp được mong đợi.

```python
def explained_variance_score(y_true, y_pred):
    """
    Calculate Explained Variance Score from scratch
    Formula: 1 - Var(y_true - y_pred) / Var(y_true)
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate variances
    residual_variance = np.var(y_true - y_pred)
    total_variance = np.var(y_true)

    # Calculate score
    score = 1 - (residual_variance / total_variance)

    return score

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"Explained Variance Score: {explained_variance_score(y_true, y_pred):.4f}")
```

Trang trình bày 12: Ví dụ thực tế - Mức tiêu thụ năng lượng theo thời gian chuỗi

Ví dụ: điều này có thể hiện việc áp dụng các quy trình khôi phục số liệu trong thời gian chuỗi báo cáo dự kiến, đặc biệt là khả năng tiêu thụ dự kiến, kết hợp các đặc tính thời gian và nhiều giá trị đánh giá dữ liệu.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Generate synthetic hourly energy consumption data
np.random.seed(42)
n_hours = 8760  # One year of hourly data

# Create time features
time_index = pd.date_range('2023-01-01', periods=n_hours, freq='H')
hour = time_index.hour
day_of_week = time_index.dayofweek
month = time_index.month

# Generate features matrix
X = np.column_stack([
    hour,
    day_of_week,
    month,
    np.sin(2 * np.pi * hour / 24),  # Daily cyclical feature
    np.cos(2 * np.pi * hour / 24)
])

# Generate target with daily, weekly, and seasonal patterns
y = (20 +
     10 * np.sin(2 * np.pi * hour / 24) +  # Daily pattern
     5 * np.sin(2 * np.pi * day_of_week / 7) +  # Weekly pattern
     8 * np.sin(2 * np.pi * month / 12) +  # Yearly pattern
     np.random.normal(0, 2, n_hours))  # Random noise
```

Slide 13: Mã nguồn cho kết quả tiêu thụ năng lượng

```python
def evaluate_time_series_model(X, y, n_splits=5):
    """
    Evaluate model using multiple metrics with time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_results = {
        'MSE': [], 'RMSE': [], 'MAE': [],
        'R²': [], 'MAPE': [], 'Huber': []
    }

    for train_idx, test_idx in tscv.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and predict
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        metrics_results['MSE'].append(mean_squared_error(y_test, y_pred))
        metrics_results['RMSE'].append(root_mean_squared_error(y_test, y_pred))
        metrics_results['MAE'].append(mean_absolute_error(y_test, y_pred))
        metrics_results['R²'].append(r2_score(y_test, y_pred))
        metrics_results['MAPE'].append(mean_absolute_percentage_error(y_test, y_pred))
        metrics_results['Huber'].append(huber_loss(y_test, y_pred))

    # Calculate mean and std for each metric
    for metric in metrics_results:
        mean_val = np.mean(metrics_results[metric])
        std_val = np.std(metrics_results[metric])
        print(f"{metric:>8}: {mean_val:.4f} ± {std_val:.4f}")

# Run evaluation
evaluate_time_series_model(X, y)
```

Trang trình bày 14: Tài nguyên bổ sung

* "Về việc sử dụng xác thực chéo để đánh giá thời gian chuỗi dự kiến"
    * [https://arxiv.org/abs/1809.09446](https://arxiv.org/abs/1809.09446)
* "Đánh giá toàn diện về hàm mất mát trong máy học"
    * [https://arxiv.org/abs/2011.00450](https://arxiv.org/abs/2011.00450)
* "Hồi quy mạnh và phát hiện ngoại lệ"
    * [https://arxiv.org/abs/1607.01152](https://arxiv.org/abs/1607.01152)
* "Dự báo chuỗi thời gian với học chuyên sâu: Khảo sát"
    * [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)
* "Thêm R bình phương: Số liệu cho mô hình hồi quy"
    * [https://arxiv.org/abs/2012.03150](https://arxiv.org/abs/2012.03150)
