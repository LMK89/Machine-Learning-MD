## Tìm hiểu về độ lệch trong trực quan hóa dữ liệu
Trang trình bày 1: Tìm hiểu về độ lệch trong phân tích dữ liệu

Độ lệch đo lường tính bất đối xứng của phân bố xác suất, cho biết dữ liệu nghiêng về bên trái hay bên phải. Trong phân tích thống kê, việc hiểu độ lệch giúp xác định các giá trị ngoại lệ, đánh giá tính chuẩn của dữ liệu và đưa ra quyết định sáng suốt về chuyển đổi dữ liệu và phương pháp lập mô hình.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate sample data with different skewness
np.random.seed(42)
normal_dist = np.random.normal(0, 1, 1000)
right_skewed = np.exp(normal_dist)
left_skewed = -np.exp(normal_dist)

# Calculate skewness
print(f"Normal Distribution Skewness: {skew(normal_dist):.3f}")
print(f"Right-skewed Distribution Skewness: {skew(right_skewed):.3f}")
print(f"Left-skewed Distribution Skewness: {skew(left_skewed):.3f}")
```

Slide 2: Công thức tính độ lệch

Định nghĩa toán học về độ lệch liên quan đến thời điểm chuẩn hóa thứ ba của một phân bố. Công thức này định lượng mức độ và hướng bất đối xứng trong tập dữ liệu so với giá trị trung bình của nó.

```python
# Mathematical formula for skewness using LaTeX notation
$$\text{Skewness} = \frac{\mathbb{E}[(X-\mu)^3]}{\sigma^3} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^3}{(\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2)^{3/2}}$$

# Implementation from scratch
def calculate_skewness(data):
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5

    third_moment = sum((x - mean) ** 3 for x in data) / n
    skewness = third_moment / (std_dev ** 3)

    return skewness

# Example usage
data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
print(f"Calculated Skewness: {calculate_skewness(data):.3f}")
```

Slide 3: Trực quan hóa các mẫu độ lệch

Hiểu cách phân phối dữ liệu khác nhau xuất hiện bằng đồ họa là rất quan trọng để phân tích dữ liệu. Việc triển khai này tạo ra biểu đồ và biểu đồ mật độ để trực quan hóa các mẫu độ lệch khác nhau trong bộ dữ liệu trong thế giới thực.

```python
import seaborn as sns
import pandas as pd

def plot_skewness_patterns(data, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(f"{title} (Skewness: {skew(data):.3f})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

# Generate example distributions
gamma_dist = np.random.gamma(2, 2, 1000)  # Right-skewed
beta_dist = np.random.beta(2, 5, 1000)    # Left-skewed

plot_skewness_patterns(gamma_dist, "Right-Skewed Distribution")
plot_skewness_patterns(beta_dist, "Left-Skewed Distribution")
plt.tight_layout()
plt.show()
```

Trang trình bày 4: Ứng dụng thực tế - Phân tích lợi nhuận chứng khoán

Phân tích dữ liệu tài chính thường xuyên gặp phải sự phân phối sai lệch, đặc biệt là lợi nhuận chứng khoán. Việc triển khai này phân tích lợi nhuận hàng ngày của danh mục đầu tư chứng khoán để hiểu các đặc điểm rủi ro của nó thông qua độ lệch.

```python
import yfinance as yf
from datetime import datetime, timedelta

def analyze_stock_returns(symbol, period='1y'):
    # Download stock data
    stock = yf.download(symbol, period=period)

    # Calculate daily returns
    returns = stock['Adj Close'].pct_change().dropna()

    # Calculate statistics
    skewness = skew(returns)

    print(f"Stock: {symbol}")
    print(f"Returns Skewness: {skewness:.3f}")

    return returns

# Analyze multiple stocks
symbols = ['AAPL', 'MSFT', 'GOOGL']
returns_data = {sym: analyze_stock_returns(sym) for sym in symbols}

# Visualize distributions
plt.figure(figsize=(12, 6))
for sym, returns in returns_data.items():
    sns.kdeplot(returns, label=sym)
plt.title("Distribution of Daily Returns")
plt.xlabel("Return")
plt.ylabel("Density")
plt.legend()
plt.show()
```

Slide 5: Phát hiện và xử lý các đặc điểm sai lệch

Khi làm việc với các mô hình học máy, các tính năng bị sai lệch có thể ảnh hưởng đáng kể đến hiệu suất của mô hình. Việc triển khai này thể hiện các kỹ thuật phát hiện và chuyển đổi các tính năng bị lệch để cải thiện độ chính xác của mô hình.

```python
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.preprocessing import PowerTransformer

def analyze_and_transform_skewness(data):
    # Calculate initial skewness
    initial_skewness = stats.skew(data)

    # Apply different transformations
    log_transform = np.log1p(data - min(data) + 1)
    box_cox = PowerTransformer(method='box-cox').fit_transform(
        data.reshape(-1, 1)).flatten()

    # Calculate transformed skewness
    log_skewness = stats.skew(log_transform)
    box_cox_skewness = stats.skew(box_cox)

    print(f"Original Skewness: {initial_skewness:.3f}")
    print(f"Log Transform Skewness: {log_skewness:.3f}")
    print(f"Box-Cox Transform Skewness: {box_cox_skewness:.3f}")

    return log_transform, box_cox

# Generate sample skewed data
np.random.seed(42)
skewed_data = np.random.lognormal(0, 1, 1000)

# Analyze and transform
log_data, box_cox_data = analyze_and_transform_skewness(skewed_data)
```

Trang trình bày 6: Sự sai lệch trong kiểm soát chất lượng

Trong quy trình sản xuất, phân tích độ lệch giúp xác định những sai lệch mang tính hệ thống về chất lượng sản phẩm. Việc triển khai này phân tích các số liệu sản xuất và thiết lập các giới hạn kiểm soát dựa trên các mẫu độ lệch.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def quality_control_analysis(measurements, spec_limits):
    # Calculate basic statistics
    mean_val = np.mean(measurements)
    std_val = np.std(measurements)
    skewness = stats.skew(measurements)

    # Calculate control limits
    ucl = mean_val + 3 * std_val
    lcl = mean_val - 3 * std_val

    # Analysis results
    out_of_spec = np.sum((measurements < spec_limits[0]) |
                        (measurements > spec_limits[1]))

    print(f"Process Skewness: {skewness:.3f}")
    print(f"Out of Spec Items: {out_of_spec}")
    print(f"Control Limits: [{lcl:.2f}, {ucl:.2f}]")

    return ucl, lcl

# Simulate manufacturing data
np.random.seed(42)
measurements = np.random.gamma(shape=2, scale=2, size=1000)
spec_limits = (2, 8)

# Perform quality control analysis
ucl, lcl = quality_control_analysis(measurements, spec_limits)

# Visualize distribution with control limits
plt.figure(figsize=(10, 6))
plt.hist(measurements, bins=30, density=True, alpha=0.7)
plt.axvline(ucl, color='r', linestyle='--', label='UCL')
plt.axvline(lcl, color='r', linestyle='--', label='LCL')
plt.title("Quality Control Distribution")
plt.legend()
plt.show()
```

Trang trình bày 7: Tác động sai lệch đến các thước đo rủi ro tài chính

Độ lệch đóng một vai trò quan trọng trong đánh giá rủi ro tài chính, đặc biệt là trong việc tính toán Giá trị rủi ro (VaR) và Thiếu hụt dự kiến. Việc triển khai này cho thấy độ lệch ảnh hưởng như thế nào đến việc tính toán số liệu rủi ro.

```python
import numpy as np
from scipy import stats
import pandas as pd

def calculate_risk_metrics(returns, confidence_level=0.95):
    # Calculate statistical moments
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    skewness = stats.skew(returns)

    # Calculate VaR and ES
    var = np.percentile(returns, (1 - confidence_level) * 100)
    es = returns[returns <= var].mean()

    # Adjust for skewness using Cornish-Fisher expansion
    z_score = stats.norm.ppf(confidence_level)
    cf_var = mean_return + std_return * (z_score +
             (z_score**2 - 1) * skewness / 6)

    results = {
        'Standard VaR': var,
        'Expected Shortfall': es,
        'Skewness-adjusted VaR': cf_var,
        'Distribution Skewness': skewness
    }

    return pd.Series(results)

# Simulate financial returns
np.random.seed(42)
returns = np.random.standard_t(df=3, size=1000) * 0.01

# Calculate risk metrics
risk_metrics = calculate_risk_metrics(returns)
print(risk_metrics)
```

Trang trình bày 8: Phát hiện độ lệch nâng cao bằng Machine Learning

Học máy có thể được sử dụng để tự động phát hiện và phân loại các loại mẫu sai lệch khác nhau trong các tập dữ liệu lớn. Việc triển khai này sử dụng cách tiếp cận mạng thần kinh để nhận dạng mẫu độ lệch.

```python
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_skewness_classifier():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def generate_skewed_samples(n_samples):
    # Generate different types of skewed distributions
    normal = np.random.normal(0, 1, (n_samples, 10))
    right_skewed = np.random.lognormal(0, 1, (n_samples, 10))
    left_skewed = -np.random.lognormal(0, 1, (n_samples, 10))

    X = np.vstack([normal, right_skewed, left_skewed])
    y = np.repeat([0, 1, 2], n_samples)

    return X, y

# Generate and prepare data
X, y = generate_skewed_samples(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = create_skewness_classifier()
history = model.fit(X_train_scaled, y_train,
                   epochs=10,
                   validation_split=0.2,
                   verbose=0)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

Trang trình bày 9: Phân tích độ lệch chuỗi thời gian

Dữ liệu chuỗi thời gian thường thể hiện các dạng sai lệch khác nhau trong các khoảng thời gian khác nhau. Việc triển khai này phân tích độ lệch phát triển như thế nào theo thời gian và thực hiện phép tính độ lệch cuộn để phát hiện mẫu thời gian.

```python
import pandas as pd
import numpy as np
from scipy import stats

def analyze_rolling_skewness(data, window_size=30):
    # Calculate rolling statistics
    rolling_skew = data.rolling(window=window_size).apply(stats.skew)
    rolling_mean = data.rolling(window=window_size).mean()

    # Create time-based features
    result = pd.DataFrame({
        'Original': data,
        'Rolling_Skewness': rolling_skew,
        'Rolling_Mean': rolling_mean
    })

    # Detect significant skewness changes
    threshold = np.std(rolling_skew.dropna()) * 2
    significant_changes = rolling_skew.abs() > threshold

    print(f"Periods with Significant Skewness: {significant_changes.sum()}")
    return result

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
data = pd.Series(np.random.gamma(2, 2, 365) +
                np.sin(np.linspace(0, 4*np.pi, 365)), index=dates)

# Analyze rolling skewness
results = analyze_rolling_skewness(data)
print("\nSkewness Statistics:")
print(results.describe())

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['Rolling_Skewness'], label='Rolling Skewness')
plt.axhline(y=0, color='r', linestyle='--', label='No Skewness')
plt.title('Rolling Skewness Over Time')
plt.legend()
plt.show()
```

Slide 10: Đánh giá độ lệch đa biến

Độ lệch đa biến mở rộng khái niệm này sang nhiều chiều, rất quan trọng đối với các bộ dữ liệu phức tạp. Việc triển khai này tính toán và trực quan hóa độ lệch đa biến bằng cách sử dụng các hệ số của Mardia.

```python
import numpy as np
from scipy.stats import chi2
import pandas as pd

def mardia_skewness(X):
    n, p = X.shape
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Calculate covariance matrix inverse
    S_inv = np.linalg.inv(np.cov(X.T))

    # Calculate Mardia's skewness
    b1p = 0
    for i in range(n):
        for j in range(n):
            mult = np.dot(X_centered[i], np.dot(S_inv, X_centered[j]))
            b1p += mult**3

    b1p = b1p / (n**2)

    # Calculate test statistic
    test_stat = (n * b1p) / 6
    p_value = 1 - chi2.cdf(test_stat, p * (p + 1) * (p + 2) / 6)

    return {
        'Mardia_Skewness': b1p,
        'Test_Statistic': test_stat,
        'P_Value': p_value
    }

# Generate multivariate data
np.random.seed(42)
n_samples = 1000
n_features = 3

# Create correlated features with different skewness
X = np.random.multivariate_normal(
    mean=[0, 0, 0],
    cov=[[1, 0.5, 0.2],
         [0.5, 1, 0.3],
         [0.2, 0.3, 1]],
    size=n_samples
)

# Transform one feature to be skewed
X[:, 0] = np.exp(X[:, 0])

# Calculate multivariate skewness
results = mardia_skewness(X)
print("\nMultivariate Skewness Analysis:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
```

Trang trình bày 11: Ước tính độ lệch chắc chắn

Các thước đo độ lệch truyền thống có thể nhạy cảm với các giá trị ngoại lệ. Việc triển khai này thể hiện các kỹ thuật ước tính độ lệch mạnh mẽ bằng cách sử dụng các phương pháp dựa trên tứ phân vị và lấy mẫu lại bootstrap.

```python
import numpy as np
from scipy import stats
from sklearn.utils import resample

def robust_skewness_estimation(data, n_bootstrap=1000):
    # Quartile skewness coefficient (Bowley skewness)
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    bowley_skewness = ((q3 + q1 - 2*q2) / (q3 - q1))

    # Bootstrap confidence intervals
    bootstrap_skewness = []
    for _ in range(n_bootstrap):
        boot_sample = resample(data)
        bootstrap_skewness.append(stats.skew(boot_sample))

    ci_lower, ci_upper = np.percentile(bootstrap_skewness, [2.5, 97.5])

    results = {
        'Bowley_Skewness': bowley_skewness,
        'Traditional_Skewness': stats.skew(data),
        'Bootstrap_CI_Lower': ci_lower,
        'Bootstrap_CI_Upper': ci_upper,
        'Bootstrap_SE': np.std(bootstrap_skewness)
    }

    return results

# Generate data with outliers
np.random.seed(42)
clean_data = np.random.gamma(2, 2, 1000)
outliers = np.random.uniform(20, 30, 50)
contaminated_data = np.concatenate([clean_data, outliers])

# Compare skewness estimates
clean_results = robust_skewness_estimation(clean_data)
contaminated_results = robust_skewness_estimation(contaminated_data)

print("Clean Data Results:")
for k, v in clean_results.items():
    print(f"{k}: {v:.4f}")

print("\nContaminated Data Results:")
for k, v in contaminated_results.items():
    print(f"{k}: {v:.4f}")
```

Trang trình bày 12: Kỹ thuật tính năng nhận biết độ lệch

Khi chuẩn bị dữ liệu cho các mô hình học máy, việc tính toán độ lệch trong kỹ thuật tính năng có thể cải thiện đáng kể hiệu suất của mô hình. Việc triển khai này thể hiện các kỹ thuật nâng cao để xử lý các tính năng bị sai lệch.

```python
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline

def advanced_skewness_transformation(X, method='auto'):
    def calculate_transformation_scores(X_transformed):
        skewness = np.abs(stats.skew(X_transformed))
        normality = stats.normaltest(X_transformed)[1]
        return skewness, normality

    transformers = {
        'box-cox': PowerTransformer(method='box-cox'),
        'yeo-johnson': PowerTransformer(method='yeo-johnson'),
        'quantile_normal': QuantileTransformer(output_distribution='normal'),
        'quantile_uniform': QuantileTransformer(output_distribution='uniform')
    }

    results = {}
    for name, transformer in transformers.items():
        try:
            X_transformed = transformer.fit_transform(X.reshape(-1, 1)).ravel()
            skewness, normality = calculate_transformation_scores(X_transformed)
            results[name] = {
                'transformed_data': X_transformed,
                'skewness': skewness,
                'normality_p_value': normality
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    if method == 'auto':
        best_method = min(results.items(),
                         key=lambda x: x[1].get('skewness', float('inf'))
                         if isinstance(x[1], dict) and 'skewness' in x[1]
                         else float('inf'))[0]
        return results[best_method]['transformed_data'], results

    return results[method]['transformed_data'], results

# Generate highly skewed data
np.random.seed(42)
skewed_feature = np.exp(np.random.normal(0, 1, 1000))

# Apply transformations
transformed_data, transformation_results = advanced_skewness_transformation(
    skewed_feature)

# Print results
print("Original Skewness:", stats.skew(skewed_feature))
for method, results in transformation_results.items():
    if 'skewness' in results:
        print(f"\n{method} transformation:")
        print(f"Skewness: {results['skewness']:.4f}")
        print(f"Normality p-value: {results['normality_p_value']:.4f}")
```

Slide 13: Dự báo độ lệch theo thời gian

Dự đoán các mô hình sai lệch trong tương lai có thể có giá trị cho việc quản lý rủi ro và ra quyết định. Việc triển khai này tạo ra một mô hình để dự báo độ lệch trong dữ liệu chuỗi thời gian.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

class SkewnessForecaster:
    def __init__(self, window_size=30, forecast_horizon=5):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def create_features(self, data):
        df = pd.DataFrame()

        # Rolling statistics
        for w in [5, 10, self.window_size]:
            df[f'skew_{w}'] = data.rolling(w).apply(stats.skew)
            df[f'std_{w}'] = data.rolling(w).std()
            df[f'kurt_{w}'] = data.rolling(w).apply(stats.kurtosis)

        return df

    def prepare_data(self, data):
        features = self.create_features(data)

        X, y = [], []
        for i in range(len(data) - self.window_size - self.forecast_horizon):
            X.append(features.iloc[i:i+self.window_size].values.flatten())
            future_skew = stats.skew(
                data.iloc[i+self.window_size:i+self.window_size+self.forecast_horizon]
            )
            y.append(future_skew)

        return np.array(X), np.array(y)

    def fit(self, data):
        X, y = self.prepare_data(data)
        self.model.fit(X, y)
        return self

    def predict(self, data):
        features = self.create_features(data)
        X = features.iloc[-self.window_size:].values.reshape(1, -1)
        return self.model.predict(X)[0]

# Generate sample time series
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
data = pd.Series(
    np.random.gamma(2, 2, 1000) + np.sin(np.linspace(0, 8*np.pi, 1000)),
    index=dates
)

# Train and evaluate forecaster
forecaster = SkewnessForecaster()
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

forecaster.fit(train_data)
predictions = []
actual = []

for i in range(len(test_data) - forecaster.forecast_horizon):
    pred = forecaster.predict(
        test_data.iloc[i:i+forecaster.window_size]
    )
    actual_skew = stats.skew(
        test_data.iloc[i+forecaster.window_size:
                      i+forecaster.window_size+forecaster.forecast_horizon]
    )
    predictions.append(pred)
    actual.append(actual_skew)

mse = mean_squared_error(actual, predictions)
print(f"Mean Squared Error: {mse:.4f}")
```

Trang trình bày 14: Tài nguyên bổ sung

Danh sách các giấy tờ liên quan từ ArXiv:

* [https://arxiv.org/abs/2103.02323](https://arxiv.org/abs/2103.02323) "Ước tính mạnh mẽ về độ lệch và độ Kurtosis trong phân phối với vô số khoảnh khắc cao hơn"
* [https://arxiv.org/abs/1908.05953](https://arxiv.org/abs/1908.05953) "Về tác động của độ lệch và độ nhọn đối với phân tích chuỗi thời gian"
* [https://arxiv.org/abs/2006.16942](https://arxiv.org/abs/2006.16942) "Học sâu để dự báo chuỗi thời gian: Trường hợp phụ tải điện"
* [https://arxiv.org/abs/1910.07920](https://arxiv.org/abs/1910.07920) "Kỹ thuật tính năng nhận biết độ lệch để dự báo chuỗi thời gian thần kinh"
* [https://arxiv.org/abs/2012.09445](https://arxiv.org/abs/2012.09445) "Khảo sát về kiểm tra phân phối: Dữ liệu của bạn không bình thường"
