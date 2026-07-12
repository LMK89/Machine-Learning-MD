## Xử lý các ngoại lệ trong dữ liệu từ điểm Z đến trực quan hóa
Trang trình bày 1: Tìm hiểu về Điểm Z để phát hiện ngoại lệ

Điểm Z biểu thị số độ lệch chuẩn mà một điểm dữ liệu nằm so với giá trị trung bình. Biện pháp thống kê này giúp xác định các giá trị ngoại lệ tiềm năng bằng cách định lượng mức độ cực đoan của mỗi giá trị so với phân bổ tổng thể, với các giá trị vượt quá ±3 thường được coi là các giá trị ngoại lệ.

```python
import numpy as np
import pandas as pd

def calculate_zscores(data):
    # Calculate z-scores for each data point
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std

    # Create DataFrame for better visualization
    df = pd.DataFrame({'original_data': data, 'z_scores': z_scores})

    # Identify outliers using |z| > 3 threshold
    outliers = df[abs(df['z_scores']) > 3]

    return df, outliers

# Example usage
data = np.array([1, 2, 2.5, 2.7, 3, 15, 2.8, 2.9, 3.1, 2.6])
results, outliers = calculate_zscores(data)
print("Full Dataset with Z-scores:")
print(results)
print("\nOutliers (|z| > 3):")
print(outliers)
```

Trang trình bày 2: Phân tích điểm Z nâng cao với điểm Z được sửa đổi

Phương pháp điểm Z được sửa đổi sử dụng độ lệch tuyệt đối trung vị và trung vị thay vì độ lệch trung bình và độ lệch chuẩn, làm cho phương pháp này trở nên chắc chắn hơn trước các giá trị cực trị và nhiều giá trị ngoại lệ trong tập dữ liệu.

```python
def modified_zscore(data):
    # Calculate median and MAD
    median = np.median(data)
    mad = np.median(np.abs(data - median)) * 1.4826

    # Calculate modified z-scores
    modified_zscores = 0.6745 * (data - median) / mad

    # Create results DataFrame
    results = pd.DataFrame({
        'data': data,
        'modified_zscore': modified_zscores,
        'is_outlier': abs(modified_zscores) > 3.5
    })

    return results

# Example with skewed data
data = np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 10, 15, 20])
results = modified_zscore(data)
print("Modified Z-score Analysis:")
print(results)
```

Trang trình bày 3: Thực hiện phương pháp IQR

Phương pháp Phạm vi liên tứ phân vị (IQR) xác định các giá trị ngoại lệ là các giá trị nằm ngoài 1,5 lần IQR dưới tứ phân vị thứ nhất hoặc cao hơn tứ phân vị thứ ba, cung cấp một cách tiếp cận mạnh mẽ ít nhạy cảm hơn với các giá trị cực đoan.

```python
def iqr_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Define bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Create mask for outliers
    outlier_mask = (data < lower_bound) | (data > upper_bound)

    results = pd.DataFrame({
        'value': data,
        'is_outlier': outlier_mask,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })

    return results

# Example usage
data = np.array([1, 2, 2.5, 2.7, 3, 25, 2.8, 2.9, 3.1, 2.6, 30, 1.8])
results = iqr_outliers(data)
print("IQR-based Outlier Detection:")
print(results)
```

Trang trình bày 4: Trực quan hóa dữ liệu để phát hiện ngoại lệ

Hiểu được sự phân phối dữ liệu thông qua trực quan hóa là rất quan trọng để phát hiện ngoại lệ. Việc triển khai này kết hợp các ô hình hộp và các ô phân tán để cung cấp cái nhìn toàn diện về các giá trị ngoại lai tiềm năng.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_outliers(data, title="Outlier Visualization"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Box plot
    sns.boxplot(x=data, ax=ax1)
    ax1.set_title("Box Plot with Outliers")

    # Scatter plot with z-scores
    z_scores = (data - np.mean(data)) / np.std(data)
    ax2.scatter(range(len(data)), z_scores)
    ax2.axhline(y=3, color='r', linestyle='--', label='Upper Threshold (z=3)')
    ax2.axhline(y=-3, color='r', linestyle='--', label='Lower Threshold (z=-3)')
    ax2.set_title("Z-Score Distribution")
    ax2.legend()

    plt.tight_layout()
    return fig

# Example usage
np.random.seed(42)
data = np.concatenate([
    np.random.normal(10, 2, 100),
    np.random.normal(30, 1, 5)  # Outliers
])
fig = visualize_outliers(data)
plt.show()
```

Trang trình bày 5: Kỹ thuật chuyển đổi ngoại lệ

Kỹ thuật chuyển đổi dữ liệu có thể giúp giảm thiểu tác động của các ngoại lệ trong khi vẫn duy trì vị trí tương đối của chúng trong tập dữ liệu. Các phương pháp phổ biến bao gồm phép biến đổi logarit, căn bậc hai và Box-Cox.

```python
import scipy.stats as stats

def transform_outliers(data):
    # Create different transformations
    log_transform = np.log1p(data - min(data) + 1)
    sqrt_transform = np.sqrt(data - min(data))
    boxcox_transform, lambda_param = stats.boxcox(data - min(data) + 1)

    results = pd.DataFrame({
        'original': data,
        'log_transform': log_transform,
        'sqrt_transform': sqrt_transform,
        'boxcox_transform': boxcox_transform
    })

    return results, lambda_param

# Example usage
data = np.array([2, 3, 4, 5, 6, 100, 4, 5, 6, 7, 200, 5])
transformed_data, lambda_param = transform_outliers(data)
print("Transformed Data:")
print(transformed_data)
print(f"\nBox-Cox transformation lambda: {lambda_param:.3f}")
```

Trang trình bày 6: Các phương pháp thống kê mạnh mẽ để xử lý ngoại lệ

Các phương pháp thống kê mạnh mẽ cung cấp các ước tính đáng tin cậy về xu hướng trung tâm và độ phân tán ngay cả khi có sự xuất hiện của các giá trị ngoại lệ. Việc triển khai này thể hiện việc sử dụng các công cụ ước tính mạnh mẽ cho vị trí và quy mô.

```python
from scipy.stats import trim_mean, iqr
from sklearn.covariance import MinCovDet

def robust_statistics(data):
    # Calculate robust estimates
    trimmed_mean = trim_mean(data, 0.1)  # 10% trimming
    winsorized_mean = stats.mstats.winsorize(data, limits=[0.05, 0.05]).mean()
    huber_location = stats.huber(data).mu

    # Create robustness comparison
    results = pd.DataFrame({
        'statistic': ['mean', 'median', 'trimmed_mean', 'winsorized_mean', 'huber_location'],
        'value': [
            np.mean(data),
            np.median(data),
            trimmed_mean,
            winsorized_mean,
            huber_location
        ]
    })

    return results

# Example with contaminated data
np.random.seed(42)
normal_data = np.random.normal(10, 2, 100)
outliers = np.array([50, 60, 70, -20, -30])
data = np.concatenate([normal_data, outliers])

results = robust_statistics(data)
print("Robust Statistics Comparison:")
print(results)
```

Trang trình bày 7: Tự động phát hiện ngoại lệ với Rừng cách ly

Thuật toán Rừng cách ly tách biệt các ngoại lệ bằng cách chọn ngẫu nhiên một tính năng và phân tách giá trị, làm cho thuật toán này đặc biệt hiệu quả đối với các tập dữ liệu nhiều chiều và yêu cầu các giả định tối thiểu về phân phối dữ liệu.

```python
from sklearn.ensemble import IsolationForest
import numpy as np

def isolation_forest_detector(data, contamination=0.1):
    # Reshape data for sklearn
    X = data.reshape(-1, 1)

    # Initialize and fit the Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )

    # Fit and predict
    predictions = iso_forest.fit_predict(X)
    scores = iso_forest.score_samples(X)

    # Create results DataFrame
    results = pd.DataFrame({
        'value': data,
        'is_outlier': predictions == -1,
        'anomaly_score': -scores  # Higher score = more likely to be outlier
    }).sort_values('anomaly_score', ascending=False)

    return results

# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, 100)
outliers = np.array([5, 7, -6, 8, -7])
data = np.concatenate([normal_data, outliers])

results = isolation_forest_detector(data)
print("Isolation Forest Results (Top 10 potential outliers):")
print(results.head(10))
```

Trang trình bày 8: Triển khai hệ số ngoại lệ cục bộ (LOF)

LOF xác định các ngoại lệ bằng cách đo độ lệch cục bộ của một điểm so với các điểm lân cận, giúp phát hiện các ngoại lệ trong các tập dữ liệu có mật độ khác nhau một cách hiệu quả.

```python
from sklearn.neighbors import LocalOutlierFactor

def lof_detector(data, n_neighbors=20):
    # Reshape data for sklearn
    X = data.reshape(-1, 1)

    # Initialize and fit LOF
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination='auto',
        novelty=False
    )

    # Predict and get negative outlier scores
    predictions = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_

    # Create results DataFrame
    results = pd.DataFrame({
        'value': data,
        'is_outlier': predictions == -1,
        'lof_score': -scores  # Convert to positive scores for consistency
    }).sort_values('lof_score', ascending=False)

    return results

# Example with clustered data and outliers
np.random.seed(42)
cluster1 = np.random.normal(0, 0.5, 50)
cluster2 = np.random.normal(5, 0.5, 50)
outliers = np.array([-2, 7, 2.5])
data = np.concatenate([cluster1, cluster2, outliers])

results = lof_detector(data)
print("LOF Detection Results (Top 10 potential outliers):")
print(results.head(10))
```

Trang trình bày 9: DBSCAN để phát hiện ngoại lệ dựa trên mật độ

DBSCAN (Phân cụm ứng dụng không gian dựa trên mật độ có nhiễu) xác định hiệu quả các điểm ngoại lệ là các điểm không thuộc bất kỳ cụm nào, đặc biệt hữu ích cho các tập dữ liệu có các cụm có hình dạng và mật độ khác nhau.

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def dbscan_outlier_detector(data, eps=0.5, min_samples=5):
    # Standardize and reshape data
    X = StandardScaler().fit_transform(data.reshape(-1, 1))

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)

    # Create results DataFrame
    results = pd.DataFrame({
        'value': data,
        'cluster': clusters,
        'is_outlier': clusters == -1
    })

    # Calculate additional statistics
    cluster_stats = results.groupby('cluster').agg({
        'value': ['count', 'mean', 'std']
    }).round(3)

    return results, cluster_stats

# Example usage
np.random.seed(42)
cluster1 = np.random.normal(0, 0.5, 50)
cluster2 = np.random.normal(5, 0.5, 50)
outliers = np.array([-2, 7, 2.5, 8, -3])
data = np.concatenate([cluster1, cluster2, outliers])

results, stats = dbscan_outlier_detector(data)
print("DBSCAN Outlier Detection Results:")
print(results[results['is_outlier']].sort_values('value'))
print("\nCluster Statistics:")
print(stats)
```

Trang trình bày 10: Ứng dụng trong thế giới thực - Các ngoại lệ của chuỗi thời gian tài chính

Dữ liệu tài chính thường chứa đựng những bất thường do các sự kiện thị trường hoặc lỗi ghi chép. Việc triển khai này thể hiện một cách tiếp cận toàn diện để phát hiện và xử lý các giá trị ngoại lệ trong dữ liệu giá cổ phiếu.

```python
import pandas as pd
import numpy as np
from scipy import stats

def analyze_financial_outliers(prices, window=20):
    # Calculate returns
    returns = np.log(prices / prices.shift(1))

    # Rolling statistics
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    # Calculate rolling z-scores
    z_scores = (returns - rolling_mean) / rolling_std

    # Multiple detection methods
    results = pd.DataFrame({
        'price': prices,
        'returns': returns,
        'z_score': z_scores,
        'is_zscore_outlier': abs(z_scores) > 3,
        'is_mad_outlier': abs(returns - returns.median()) > 3 * stats.median_abs_deviation(returns.dropna())
    })

    # Add volatility regime detection
    results['volatility'] = rolling_std
    results['high_volatility'] = results['volatility'] > results['volatility'].quantile(0.95)

    return results

# Example with simulated stock data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
prices = 100 * (1 + np.random.normal(0.0002, 0.01, 252)).cumprod()
# Add some artificial outliers
prices[50] *= 1.15  # Sudden jump
prices[150] *= 0.85  # Sudden drop

results = analyze_financial_outliers(prices)
print("Financial Outlier Analysis Results:")
print(results[results['is_zscore_outlier']].head())
```

Trang trình bày 11: Mã nguồn để trực quan hóa chuỗi thời gian tài chính

```python
def visualize_financial_outliers(results):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Price plot with outliers highlighted
    ax1.plot(results.index, results['price'], label='Price')
    outliers = results[results['is_zscore_outlier']]
    ax1.scatter(outliers.index, outliers['price'],
                color='red', label='Outliers', zorder=5)
    ax1.set_title('Price Series with Outliers')
    ax1.legend()

    # Returns distribution
    sns.histplot(data=results['returns'].dropna(), ax=ax2, bins=50)
    ax2.axvline(results['returns'].mean(), color='r', linestyle='--',
                label='Mean')
    ax2.axvline(results['returns'].median(), color='g', linestyle='--',
                label='Median')
    ax2.set_title('Returns Distribution')
    ax2.legend()

    # Volatility regime
    ax3.plot(results.index, results['volatility'], label='Volatility')
    ax3.axhline(results['volatility'].quantile(0.95), color='r',
                linestyle='--', label='95th Percentile')
    ax3.set_title('Volatility Regime')
    ax3.legend()

    plt.tight_layout()
    return fig

# Visualize results
fig = visualize_financial_outliers(results)
plt.show()
```

Slide 12: Ứng dụng trong thế giới thực - Phát hiện bất thường dữ liệu cảm biến

Mạng cảm biến thường tạo ra dữ liệu với nhiều loại dị thường khác nhau. Việc triển khai này cho thấy cách phát hiện và phân loại các loại ngoại lệ dữ liệu cảm biến khác nhau.

```python
def analyze_sensor_data(timestamps, values, window_size=12):
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })

    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # Calculate rolling statistics
    df['rolling_mean'] = df['value'].rolling(window=window_size).mean()
    df['rolling_std'] = df['value'].rolling(window=window_size).std()

    # Different types of anomalies
    df['spike'] = abs(df['value'] - df['rolling_mean']) > 3 * df['rolling_std']
    df['level_shift'] = abs(df['rolling_mean'].diff()) > 2 * df['rolling_std']
    df['variance_change'] = df['rolling_std'] > 2 * df['rolling_std'].mean()

    # Seasonal adjustment
    seasonal_means = df.groupby('hour')['value'].transform('mean')
    seasonal_std = df.groupby('hour')['value'].transform('std')
    df['seasonal_residual'] = (df['value'] - seasonal_means) / seasonal_std

    return df

# Generate example sensor data
np.random.seed(42)
timestamps = pd.date_range('2024-01-01', periods=720, freq='H')
base_signal = 100 + 10 * np.sin(np.pi * np.arange(720) / 24)  # Daily cycle
noise = np.random.normal(0, 1, 720)
anomalies = np.zeros(720)
anomalies[100:105] = 30  # Spike
anomalies[300:400] += np.linspace(0, 20, 100)  # Level shift
values = base_signal + noise + anomalies

results = analyze_sensor_data(timestamps, values)
print("Sensor Data Analysis Results:")
print(results[results[['spike', 'level_shift', 'variance_change']].any(axis=1)].head())
```

Trang trình bày 13: Mã nguồn để trực quan hóa dữ liệu cảm biến

```python
def visualize_sensor_anomalies(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Raw data with detected anomalies
    ax1.plot(results['timestamp'], results['value'], label='Raw Signal')
    spikes = results[results['spike']]
    level_shifts = results[results['level_shift']]
    ax1.scatter(spikes['timestamp'], spikes['value'],
                color='red', label='Spikes', zorder=5)
    ax1.scatter(level_shifts['timestamp'], level_shifts['value'],
                color='orange', label='Level Shifts', zorder=5)
    ax1.set_title('Sensor Data with Detected Anomalies')
    ax1.legend()

    # Rolling statistics
    ax2.plot(results['timestamp'], results['rolling_mean'], label='Rolling Mean')
    ax2.fill_between(results['timestamp'],
                     results['rolling_mean'] - 2*results['rolling_std'],
                     results['rolling_mean'] + 2*results['rolling_std'],
                     alpha=0.2, label='±2σ Band')
    ax2.set_title('Rolling Statistics')
    ax2.legend()

    # Seasonal pattern
    hourly_mean = results.groupby('hour')['value'].mean()
    hourly_std = results.groupby('hour')['value'].std()
    ax3.plot(hourly_mean.index, hourly_mean.values, label='Hourly Mean')
    ax3.fill_between(hourly_mean.index,
                     hourly_mean - hourly_std,
                     hourly_mean + hourly_std,
                     alpha=0.2, label='±1σ Band')
    ax3.set_title('Daily Pattern')
    ax3.legend()

    # Seasonal residuals distribution
    sns.histplot(data=results['seasonal_residual'].dropna(), ax=ax4, bins=50)
    ax4.axvline(0, color='r', linestyle='--', label='Mean')
    ax4.axvline(-3, color='g', linestyle='--', label='-3σ')
    ax4.axvline(3, color='g', linestyle='--', label='+3σ')
    ax4.set_title('Seasonal Residuals Distribution')
    ax4.legend()

    plt.tight_layout()
    return fig

# Visualize the results
fig = visualize_sensor_anomalies(results)
plt.show()
```

Trang trình bày 14: Phương pháp tập hợp để phát hiện ngoại lệ mạnh mẽ

Việc triển khai này kết hợp nhiều phương pháp phát hiện ngoại lệ để tạo ra một hệ thống phát hiện mạnh mẽ và đáng tin cậy hơn, sử dụng cơ chế bỏ phiếu để giảm kết quả dương tính giả.

```python
class EnsembleOutlierDetector:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.detectors = {
            'isolation_forest': IsolationForest(contamination=contamination),
            'lof': LocalOutlierFactor(contamination=contamination, novelty=True),
            'robust_covariance': MinCovDet(contamination=contamination)
        }

    def fit(self, X):
        # Ensure 2D array
        X = np.atleast_2d(X)
        if X.shape[1] == 1:
            X = np.hstack([X, np.zeros_like(X)])

        # Fit all detectors
        for name, detector in self.detectors.items():
            try:
                detector.fit(X)
            except Exception as e:
                print(f"Warning: {name} fitting failed: {e}")
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] == 1:
            X = np.hstack([X, np.zeros_like(X)])

        # Collect predictions from all detectors
        predictions = {}
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'predict'):
                    predictions[name] = detector.predict(X)
                else:
                    predictions[name] = detector.fit_predict(X)
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")

        # Combine predictions using majority voting
        votes = np.zeros(X.shape[0])
        for pred in predictions.values():
            votes += (pred == -1)

        # Final decision: point is outlier if majority says so
        return (votes > len(predictions) / 2).astype(int) * -2 + 1

# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
outliers = np.random.normal(4, 0.5, 50)
data = np.concatenate([normal_data, outliers])

detector = EnsembleOutlierDetector()
predictions = detector.fit_predict(data.reshape(-1, 1))

results = pd.DataFrame({
    'value': data,
    'is_outlier': predictions == -1
})
print("Ensemble Detector Results:")
print(f"Total outliers detected: {sum(predictions == -1)}")
print(results[results['is_outlier']].describe())
```

Trang trình bày 15: Tài nguyên bổ sung

* "Nghiên cứu về các phương pháp phát hiện ngoại lệ và ứng dụng của chúng" - [https://arxiv.org/abs/2202.01048](https://arxiv.org/abs/2202.01048)
* "Thuật toán rừng cô lập và các ứng dụng của nó" - [https://arxiv.org/abs/1811.02141](https://arxiv.org/abs/1811.02141)
* "Khảo sát về các phương pháp học sâu để phát hiện sự bất thường" - [https://arxiv.org/abs/2009.14017](https://arxiv.org/abs/2009.14017)
* "Thống kê mạnh mẽ để phát hiện ngoại lệ: Một nghiên cứu so sánh" - [https://arxiv.org/abs/1904.02181](https://arxiv.org/abs/1904.02181)
* "Yếu tố ngoại lệ cục bộ: Phương pháp tiếp cận dựa trên mật độ để phát hiện ngoại lệ" - [https://arxiv.org/abs/1906.03509](https://arxiv.org/abs/1906.03509)
