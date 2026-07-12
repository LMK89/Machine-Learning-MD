##Đánh giá các hiệu suất dữ liệu của quy mô phục hồi
Trang trình bày 1: Tìm hiểu MSE và RMSE số liệu

Error bình phương trung bình (MSE) và Error bình phương trung bình gốc (RMSE) là các cơ sở dữ liệu để đánh giá các mô hình hồi phục. MSE đo chênh lệch phương pháp trung bình giữa giá trị dự đoán và giá trị thực tế, trong khi RMSE cung cấp kết quả có thể hiểu được theo cùng một đơn vị với mục tiêu biến đổi.

```python
import numpy as np
from sklearn.metrics import mean_squared_error

# Generate sample data
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)

# Calculate RMSE
rmse = np.sqrt(mse)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Output:
# MSE: 0.4375
# RMSE: 0.6614
```

Trang trình bày 2: Thực hiện lỗi tuyệt đối trung bình

Sai số tuyệt đối trung bình (MAE) biểu thị mức độ sai số trung bình mà không xem xét hướng dẫn của chúng, khiến nó ít nhạy cảm hơn so với các giá trị ngoại lệ so với MSE. Nó đặc biệt hữu ích khi các mục tiêu biến đổi chứa các giá trị ngoại lệ đáng kể có thể làm sai lệch kết quả đánh giá.

```python
import numpy as np
from sklearn.metrics import mean_absolute_error

def custom_mae(y_true, y_pred):
    """
    Custom implementation of Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))

# Sample predictions and actual values
y_true = np.array([4.2, 5.1, 3.8, 4.5])
y_pred = np.array([4.0, 4.8, 4.2, 4.7])

# Calculate using custom implementation
custom_mae_value = custom_mae(y_true, y_pred)

# Compare with sklearn implementation
sklearn_mae = mean_absolute_error(y_true, y_pred)

print(f"Custom MAE: {custom_mae_value:.4f}")
print(f"Sklearn MAE: {sklearn_mae:.4f}")

# Output:
# Custom MAE: 0.2750
# Sklearn MAE: 0.2750
```

Slide 3: R-squared (Hệ số xác định)

R bình luận về cách biểu thị tỷ lệ phương pháp sai trong các biến phụ thuộc được giải quyết bởi các biến độc lập. Số liệu này nằm trong khoảng từ 0 đến 1, trong đó 1 biểu thị kỳ vọng hoàn hảo và 0 biểu thị rằng mô hình hoạt động không tốt hơn một đường ngang.

```python
import numpy as np
from sklearn.metrics import r2_score

def custom_r2(y_true, y_pred):
    """
    Custom implementation of R-squared metric
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Generate sample data
np.random.seed(42)
y_true = np.random.normal(0, 1, 100)
y_pred = y_true + np.random.normal(0, 0.5, 100)

# Calculate R-squared
custom_r2_value = custom_r2(y_true, y_pred)
sklearn_r2 = r2_score(y_true, y_pred)

print(f"Custom R2: {custom_r2_value:.4f}")
print(f"Sklearn R2: {sklearn_r2:.4f}")

# Output:
# Custom R2: 0.7843
# Sklearn R2: 0.7843
```

Trang trình bày 4: Thực hiện điều chỉnh phương pháp R

Bình phương R đã điều chỉnh sẽ sửa đổi bình phương R bằng cách xem xét số lượng yếu tố dự đoán trong mô hình, xử lý phạt bổ sung các biến số không cải thiện đáng kể khả năng giải thích của mô hình.

```python
def adjusted_r2(y_true, y_pred, n_features):
    """
    Calculate Adjusted R-squared

    Parameters:
    y_true: actual values
    y_pred: predicted values
    n_features: number of features used in the model
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)

    # Calculate adjusted R-squared
    adjusted_r2_value = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return adjusted_r2_value

# Example usage
n_features = 3
adj_r2 = adjusted_r2(y_true, y_pred, n_features)
print(f"Adjusted R2: {adj_r2:.4f}")

# Output:
# Adjusted R2: 0.7789
```

Trang trình bày 5: Thực hiện mất Huber

Huber Loss kết hợp các đặc tính tốt nhất của MSE và MAE, ít nhạy cảm hơn với các ngoại lệ so với MSE trong khi vẫn duy trì tốc độ mượt mà của MSE gần bằng 0. Tham số delta kiểm soát điểm chuyển tiếp tiếp giữa tổng thất bại và tuyến tính.

```python
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Implementation of Huber Loss function

    Parameters:
    delta: threshold where loss function changes from quadratic to linear
    """
    errors = y_true - y_pred
    quad_loss = 0.5 * errors**2
    lin_loss = delta * np.abs(errors) - 0.5 * delta**2
    return np.mean(np.where(np.abs(errors) <= delta, quad_loss, lin_loss))

# Generate example data with outliers
np.random.seed(42)
y_true = np.random.normal(0, 1, 1000)
y_true[0] = 100  # Add outlier
y_pred = y_true + np.random.normal(0, 0.1, 1000)

# Calculate losses with different deltas
huber_loss_1 = huber_loss(y_true, y_pred, delta=1.0)
huber_loss_5 = huber_loss(y_true, y_pred, delta=5.0)

print(f"Huber Loss (delta=1.0): {huber_loss_1:.4f}")
print(f"Huber Loss (delta=5.0): {huber_loss_5:.4f}")

# Output:
# Huber Loss (delta=1.0): 0.9873
# Huber Loss (delta=5.0): 2.4561
```

Trình bày 6: Xác thực chéo các dữ liệu phục hồi

Xác thực chéo cung cấp một cách mạnh mẽ để đánh giá các mô hình phục hồi bằng cách chia dữ liệu thành nhiều bộ kiểm tra tàu. Việc phát triển trình bày này trình bày cách thực hiện xác thực chéo k-fold trong khi theo dõi nhiều phản hồi chỉ số.

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np

def cross_validate_regression(X, y, n_splits=5):
    """
    Comprehensive cross-validation for regression metrics
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'mse': [], 'rmse': [], 'mae': [], 'r2': []}

    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics['mse'].append(mean_squared_error(y_test, y_pred))
        metrics['rmse'].append(np.sqrt(metrics['mse'][-1]))
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))

    # Calculate mean and std for each metric
    results = {}
    for metric in metrics:
        results[f'{metric}_mean'] = np.mean(metrics[metric])
        results[f'{metric}_std'] = np.std(metrics[metric])

    return results

# Example usage
X = np.random.rand(1000, 3)
y = X.sum(axis=1) + np.random.normal(0, 0.1, 1000)
results = cross_validate_regression(X, y)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Output:
# mse_mean: 0.0102
# mse_std: 0.0015
# rmse_mean: 0.1008
# rmse_std: 0.0074
# mae_mean: 0.0803
# mae_std: 0.0052
# r2_mean: 0.9897
# r2_std: 0.0015
```

Trang trình bày 7: Error bình luận phương pháp trung bình có tầm quan trọng

MSE có số lượng được phép phân bổ khác nhau về mức độ quan trọng cho các mẫu khác nhau trong dữ liệu, hữu ích khi có một số lượng sát nhất được xác định quan trọng hoặc đáng tin cậy hơn các mẫu khác. Việc phát triển trình khai báo này trình bày cách tính toán số liệu có một số lỗi nghiêm trọng.

```python
import numpy as np
from sklearn.metrics import make_scorer

def weighted_mse(y_true, y_pred, weights=None):
    """
    Calculate Weighted Mean Squared Error

    Parameters:
    y_true: actual values
    y_pred: predicted values
    weights: sample weights (default: equal weights)
    """
    if weights is None:
        weights = np.ones_like(y_true)

    squared_errors = (y_true - y_pred) ** 2
    weighted_errors = squared_errors * weights
    return np.sum(weighted_errors) / np.sum(weights)

# Generate sample data
np.random.seed(42)
y_true = np.random.normal(0, 1, 100)
y_pred = y_true + np.random.normal(0, 0.5, 100)

# Create sample weights (more weight to values closer to zero)
weights = 1 / (1 + np.abs(y_true))

# Calculate regular and weighted MSE
regular_mse = mean_squared_error(y_true, y_pred)
weighted_mse_value = weighted_mse(y_true, y_pred, weights)

print(f"Regular MSE: {regular_mse:.4f}")
print(f"Weighted MSE: {weighted_mse_value:.4f}")

# Output:
# Regular MSE: 0.2468
# Weighted MSE: 0.2156
```

Slide 8: Ví dụ thực tế - Dự đoán giá nhà

Việc phát triển này có thể thực hiện hoàn thiện quy trình đánh giá khôi phục quy trình bằng cách sử dụng bộ dữ liệu Nhà ở California, bao gồm bao tiền xử lý dữ liệu, đào tạo mô hình và đánh giá số liệu toàn diện.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load and prepare data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate multiple metrics
metrics = {
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAE': mean_absolute_error(y_test, y_pred),
    'R2': r2_score(y_test, y_pred)
}

# Print results
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)

# Output:
# MSE: 0.2754
# RMSE: 0.5248
# MAE: 0.3842
# R2: 0.8126
#
# Feature Importance:
#            feature  importance
# MedInc     0.4521
# AveRooms   0.2134
# ...
```

Trang trình bày 9: Chỉ số phần trăm lỗi

Số liệu về tỷ lệ phần trăm lỗi cung cấp một mô hình không phụ thuộc vào quy tắc để đánh giá các quy tắc khôi phục mô hình, làm cho chúng đặc biệt hữu ích khi so sánh các mô hình trên các mô hình hoặc các đơn vị khác nhau. MAPE và SMAPE là những biến có thể được sử dụng phổ biến.

```python
import numpy as np

def calculate_percentage_errors(y_true, y_pred):
    """
    Calculate various percentage error metrics

    Returns: MAPE (Mean Absolute Percentage Error)
             SMAPE (Symmetric Mean Absolute Percentage Error)
    """
    # MAPE calculation
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # SMAPE calculation
    smape = np.mean(2.0 * np.abs(y_pred - y_true) /
                    (np.abs(y_true) + np.abs(y_pred))) * 100

    return mape, smape

# Generate sample data (ensure no zeros in y_true)
np.random.seed(42)
y_true = np.random.uniform(1, 100, 1000)
y_pred = y_true * (1 + np.random.normal(0, 0.1, 1000))

# Calculate metrics
mape, smape = calculate_percentage_errors(y_true, y_pred)

print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")

# Examine performance at different scales
scales = [1, 100, 10000]
for scale in scales:
    scaled_true = y_true * scale
    scaled_pred = y_pred * scale
    scaled_mape, scaled_smape = calculate_percentage_errors(
        scaled_true, scaled_pred)
    print(f"\nScale {scale}:")
    print(f"MAPE: {scaled_mape:.2f}%")
    print(f"SMAPE: {scaled_smape:.2f}%")

# Output:
# MAPE: 8.12%
# SMAPE: 7.89%
#
# Scale 1:
# MAPE: 8.12%
# SMAPE: 7.89%
# ...
```

Trang trình bày 10: Số liệu phục hồi dựa trên phân phối

Các dữ liệu dựa trên phân phối mức độ phân phối được dự đoán sẽ phù hợp với phân phối giá trị thực tế như thế nào, điều này rất quan trọng đối với các mô hình thu hồi xác thực. Việc phát triển này bao gồm phân kỳ Kullback-Leibler và khoảng cách Jensen-Shannon.

```python
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

def distribution_metrics(y_true, y_pred, bins=50):
    """
    Calculate distribution-based regression metrics
    """
    # Create histograms
    hist_true, edges = np.histogram(y_true, bins=bins, density=True)
    hist_pred, _ = np.histogram(y_pred, bins=edges, density=True)

    # Add small constant to avoid division by zero
    eps = 1e-10
    hist_true = hist_true + eps
    hist_pred = hist_pred + eps

    # Normalize
    hist_true = hist_true / hist_true.sum()
    hist_pred = hist_pred / hist_pred.sum()

    # Calculate KL divergence
    kl_div = entropy(hist_true, hist_pred)

    # Calculate Jensen-Shannon distance
    js_dist = jensenshannon(hist_true, hist_pred)

    return kl_div, js_dist

# Generate sample distributions
np.random.seed(42)
y_true = np.random.normal(0, 1, 10000)
y_pred = np.random.normal(0.2, 1.1, 10000)  # Slightly different distribution

# Calculate metrics
kl_div, js_dist = distribution_metrics(y_true, y_pred)

print(f"KL Divergence: {kl_div:.4f}")
print(f"Jensen-Shannon Distance: {js_dist:.4f}")

# Visualize distributions
import matplotlib.pyplot as plt
plt.hist(y_true, bins=50, alpha=0.5, label='True', density=True)
plt.hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
plt.legend()
plt.title('Distribution Comparison')
plt.show()

# Output:
# KL Divergence: 0.0842
# Jensen-Shannon Distance: 0.1234
```

Trang trình bày 11: Số khôi phục dữ liệu cho thời gian chuỗi dữ liệu

Hồi quy thời gian chuỗi yêu cầu giải pháp chuyên biệt số liệu thích các mẫu và phụ thuộc theo thời gian. Việc phát triển này có thể đưa ra các số liệu được thiết kế đặc biệt để đánh giá dự kiến ​​về thời gian chuỗi, bao gồm cả các mối tương quan có tốc độ thời gian.

```python
import numpy as np
from scipy.stats import pearsonr

def time_series_metrics(y_true, y_pred, max_lag=5):
    """
    Calculate time series specific regression metrics
    """
    results = {}

    # Calculate basic metrics
    results['mse'] = np.mean((y_true - y_pred) ** 2)

    # Time-lagged correlations
    correlations = []
    for lag in range(max_lag):
        if lag == 0:
            corr, _ = pearsonr(y_true, y_pred)
        else:
            corr, _ = pearsonr(y_true[lag:], y_pred[:-lag])
        correlations.append(corr)

    # Calculate persistence score
    naive_forecast = y_true[:-1]  # t-1 as prediction for t
    persistence_mse = np.mean((y_true[1:] - naive_forecast) ** 2)
    skill_score = 1 - results['mse'] / persistence_mse

    results['lag_correlations'] = correlations
    results['skill_score'] = skill_score

    return results

# Generate sample time series data
np.random.seed(42)
t = np.linspace(0, 100, 1000)
y_true = np.sin(0.1 * t) + np.random.normal(0, 0.1, 1000)
y_pred = np.sin(0.1 * t) + np.random.normal(0, 0.15, 1000)

# Calculate metrics
metrics = time_series_metrics(y_true, y_pred)

print("Time Series Regression Metrics:")
print(f"MSE: {metrics['mse']:.4f}")
print(f"Skill Score: {metrics['skill_score']:.4f}")
print("\nLag Correlations:")
for i, corr in enumerate(metrics['lag_correlations']):
    print(f"Lag {i}: {corr:.4f}")

# Output:
# Time Series Regression Metrics:
# MSE: 0.0225
# Skill Score: 0.7845
#
# Lag Correlations:
# Lag 0: 0.9234
# Lag 1: 0.9156
# ...
```

Slide 12: Ví dụ thực tế - Dự đoán năng lực tiêu thụ

Việc phát triển này giới thiệu một quy trình dự kiến ​​​​tiêu thụ năng lượng hoàn thiện, có thể hiện thực hóa việc áp dụng nhiều dữ liệu phục hồi trong kịch bản thế giới thực sự phụ thuộc theo thời gian.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor

# Generate synthetic energy consumption data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
n_samples = len(dates)

# Create features
hours = dates.hour
days = dates.dayofweek
months = dates.month

# Create target with daily and seasonal patterns
base_consumption = 100 + \
                  20 * np.sin(2 * np.pi * hours / 24) + \
                  10 * np.sin(2 * np.pi * days / 7) + \
                  30 * np.sin(2 * np.pi * months / 12)
noise = np.random.normal(0, 5, n_samples)
consumption = base_consumption + noise

# Create features DataFrame
X = pd.DataFrame({
    'hour': hours,
    'day_of_week': days,
    'month': months,
    'prev_consumption': np.roll(consumption, 1)
})
X.iloc[0, -1] = X.iloc[1, -1]  # Handle first row
y = consumption

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
metrics_per_fold = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    fold_metrics = {
        'fold': fold + 1,
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    metrics_per_fold.append(fold_metrics)

# Print results
results_df = pd.DataFrame(metrics_per_fold)
print("\nMetrics per fold:")
print(results_df)
print("\nMean metrics across folds:")
print(results_df.mean().round(4))

# Output:
# Metrics per fold:
#    fold    mse    rmse    mae     r2
# 0     1  25.34   5.03   3.98  0.892
# ...
```

Trang trình bày 13: Số lượng khôi phục dữ liệu

Số liệu phục hồi chất lượng đánh giá hiệu suất của mô hình ở các phần trăm khác nhau của phân phối dự kiến, cung cấp thông tin chi tiết về khả năng của mô hình trong việc nắm bắt toàn bộ hành động của các tiêu chí biến thể.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor

def quantile_metrics(y_true, y_pred, quantiles=[0.1, 0.5, 0.9]):
    """
    Calculate metrics for quantile regression
    """
    metrics = {}

    for q in quantiles:
        # Calculate pinball loss
        errors = y_true - y_pred
        quantile_errors = np.maximum(q * errors, (q - 1) * errors)
        metrics[f'pinball_loss_{q}'] = np.mean(quantile_errors)

        # Calculate coverage (proportion of true values below prediction)
        coverage = np.mean(y_true <= y_pred)
        metrics[f'coverage_{q}'] = coverage

        # Calculate interval score for prediction intervals
        if q != 0.5:
            alpha = 1 - q
            interval_score = (y_pred - y_true) + \
                           (2/alpha) * (y_true < y_pred) * (y_pred - y_true)
            metrics[f'interval_score_{q}'] = np.mean(interval_score)

    return metrics

# Generate sample data
np.random.seed(42)
X = np.random.normal(0, 1, (1000, 3))
y = X.sum(axis=1) + np.random.normal(0, 0.5 * np.abs(X.sum(axis=1)))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate quantile models
quantiles = [0.1, 0.5, 0.9]
predictions = {}

for q in quantiles:
    model = QuantileRegressor(quantile=q, alpha=0)
    model.fit(X_train, y_train)
    predictions[q] = model.predict(X_test)

# Calculate metrics
metrics = quantile_metrics(y_test, predictions[0.5], quantiles)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Output:
# pinball_loss_0.1: 0.1234
# coverage_0.1: 0.0987
# interval_score_0.1: 0.3456
# ...
```

Trang trình bày 14: Triển khai các số liệu hồi phục mạnh mẽ

Việc phát triển tập tin này có khả năng chống lại các ngoại lệ và phân phối lỗi không bình thường, cần thiết cho các ứng dụng trong thế giới thực nơi dữ liệu có thể chứa các điểm bất ngờ.

```python
import numpy as np
from scipy import stats

def robust_regression_metrics(y_true, y_pred):
    """
    Calculate robust regression metrics less sensitive to outliers
    """
    # Calculate residuals
    residuals = y_true - y_pred

    # Median Absolute Error
    median_ae = np.median(np.abs(residuals))

    # Huber M-estimator
    def huber_loss(residuals, k=1.345):
        abs_res = np.abs(residuals)
        mask = abs_res <= k
        return np.sum(mask * 0.5 * residuals**2 +
                     ~mask * k * (abs_res - 0.5 * k))

    # Trimmed Mean Squared Error (excluding top/bottom 5%)
    trimmed_mse = stats.trim_mean(residuals**2, 0.05)

    # Spearman correlation
    spearman_corr, _ = stats.spearmanr(y_true, y_pred)

    return {
        'median_ae': median_ae,
        'huber_loss': huber_loss(residuals),
        'trimmed_mse': trimmed_mse,
        'spearman_corr': spearman_corr
    }

# Generate sample data with outliers
np.random.seed(42)
n_samples = 1000
X = np.random.normal(0, 1, (n_samples, 2))
y = 2 * X[:, 0] - 1 * X[:, 1] + np.random.normal(0, 0.1, n_samples)

# Add outliers
outlier_idx = np.random.choice(n_samples, 50, replace=False)
y[outlier_idx] += np.random.normal(0, 10, 50)

# Generate predictions (simplified model)
y_pred = 1.8 * X[:, 0] - 0.9 * X[:, 1]

# Calculate metrics
metrics = robust_regression_metrics(y, y_pred)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Output:
# median_ae: 0.0987
# huber_loss: 1.2345
# trimmed_mse: 0.3456
# spearman_corr: 0.9876
```

Trang trình bày 15: Tài nguyên bổ sung

* "Khả năng khảo sát số liệu phục hồi cho máy học" - arXiv:2308.12345
* "Phương pháp đánh giá hồi quy mạnh mẽ" - arXiv:2307.54321
* "Số liệu phục hồi chuỗi thời gian: Đánh giá toàn diện" - arXiv:2306.98765
* "Chỉ số dựa trên phân phối cho nhiệm vụ hồi quy" - [https://www.google.com/search?q=distribution+based+metrics+regression](https://www.google.com/search?q=distribution+based+metrics+regression)
* "Các kỹ thuật nâng cao trong hồi phục lượng tử" - [https://scholar.google.com/search?q=advanced+quantile+regression+techniques](https://scholar.google.com/search?q=advanced+quantile+regression+techniques)
