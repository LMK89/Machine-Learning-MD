## Phát hiện bất ngờ trong dữ liệu chuỗi thời gian bằng Python
Trang trình bày 1: Giới thiệu về hiện tượng bất thường trong dữ liệu chuỗi

Phát hiện sự bất thường trong chuỗi dữ liệu thời gian là một nhiệm vụ quan trọng trong nhiều lĩnh vực khác nhau, từ giám sát các quy trình công nghiệp đến phân tích những thay đổi của môi trường. Kỹ thuật này giúp xác định các mô hình hoặc sự kiện bất thường có sai lệch đáng kể đối với hành vi dự kiến. Trong bài trình bày này, chúng tôi sẽ khám phá cách thực hiện những điều bất ngờ bằng Python, tập trung vào các ví dụ thực tế và thông tin chi tiết hữu ích.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample time series with an anomaly
np.random.seed(42)
time = np.arange(0, 100, 0.1)
signal = np.sin(time) + np.random.normal(0, 0.1, len(time))
signal[800:830] += 2  # Introduce an anomaly

plt.figure(figsize=(12, 6))
plt.plot(time, signal)
plt.title('Time Series with Anomaly')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()
```

Trang trình bày 2: Phương pháp thống kê: Đường trung bình động

Một trong những phương pháp đơn giản nhất để phát hiện sự bất thường là sử dụng đường trung bình động. Kỹ thuật này tính toán mức độ trung bình của một cửa sổ dữ liệu có kích thước cố định và so sánh từng điểm với mức độ trung bình này. Nếu có một điểm đáng kể đối với đường trung bình, điểm đó sẽ được gắn cờ là điểm bất thường.

```python
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Generate sample data
np.random.seed(42)
time = np.arange(1000)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.5, 1000)
signal[700:720] += 5  # Introduce an anomaly

# Calculate moving average
window_size = 50
ma = moving_average(signal, window_size)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Original Signal')
plt.plot(time[window_size-1:], ma, label='Moving Average', color='red')
plt.title('Time Series with Moving Average')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Trang trình bày 3: Phát hiện bất ngờ bằng đường trung bình

Để phát hiện sự bất thường bằng phương pháp trung bình động, chúng tôi có thể đặt ngưỡng dựa trên độ lệch của tín hiệu. Nhiều điểm lệch khỏi đường trung bình hơn mức độ lệch chuẩn nhất được xác định là điểm bất thường.

```python
import numpy as np
import matplotlib.pyplot as plt

def detect_anomalies(signal, ma, threshold=2):
    std = np.std(signal)
    anomalies = np.abs(signal - ma) > threshold * std
    return anomalies

# Using the same data from the previous slide
window_size = 50
ma = moving_average(signal, window_size)

# Detect anomalies
anomalies = detect_anomalies(signal[window_size-1:], ma)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Original Signal')
plt.plot(time[window_size-1:], ma, label='Moving Average', color='red')
plt.scatter(time[window_size-1:][anomalies], signal[window_size-1:][anomalies],
            color='green', label='Anomalies')
plt.title('Anomaly Detection using Moving Average')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Trang trình bày 4: Phương pháp Z-Score

Phương pháp điểm Z là một phương pháp khác để phát hiện những điều không mong đợi. Nó đo một điểm dữ liệu có độ lệch bao nhiêu để có giá trị trung bình. Phương pháp này đặc biệt hữu ích khi dữ liệu đậm đặc theo chuẩn phân phối.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def z_score_anomalies(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

# Generate sample data
np.random.seed(42)
time = np.arange(1000)
signal = np.random.normal(0, 1, 1000)
signal[800:820] = 5  # Introduce anomalies

# Detect anomalies using Z-score
anomalies = z_score_anomalies(signal)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Signal')
plt.scatter(time[anomalies], signal[anomalies], color='red', label='Anomalies')
plt.title('Anomaly Detection using Z-Score Method')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Slide 5: Phân vùng theo mùa

Nhiều chuỗi thời gian có thể hiện các mô hình theo mùa. Việc phân tích theo mùa giúp phân chia thời gian chuỗi thành các phần xu hướng, mùa vụ và phần dư. Sự cố bất ngờ có thể được phát hiện trong phần còn lại của thành phần.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample seasonal data
np.random.seed(42)
time = np.arange(1000)
trend = 0.01 * time
seasonal = 5 * np.sin(2 * np.pi * time / 100)
residual = np.random.normal(0, 1, 1000)
signal = trend + seasonal + residual
signal[800:820] += 10  # Introduce anomalies

# Perform seasonal decomposition
result = seasonal_decompose(signal, model='additive', period=100)

# Plot results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Original Signal')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Trang trình bày 6: Phát triển bất thường ở phần dư

Sau khi phân tích theo mùa, chúng tôi có thể áp dụng các kỹ thuật phát hiện thường xuyên cho thành phần dư. Cách tiếp cận này giúp xác định những điểm bất ngờ không phụ thuộc vào mùa thông thường hoặc xu hướng chung.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Using the same seasonal data from the previous slide
result = seasonal_decompose(signal, model='additive', period=100)

# Detect anomalies in residuals using Z-score method
residual_anomalies = z_score_anomalies(result.resid, threshold=3)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, result.resid, label='Residual')
plt.scatter(time[residual_anomalies], result.resid[residual_anomalies],
            color='red', label='Anomalies')
plt.title('Anomaly Detection in Residual Component')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.legend()
plt.show()
```

Trang trình bày 7: Phương pháp học máy: Rừng cô lập

Cách ly là một máy tính học không giám sát, đặc biệt là hiệu quả trong việc phát hiện sự việc bất ngờ. Nó hoạt động bằng cách thiết lập các điểm bất ngờ trong dữ liệu thay vì cài đặt thông tin sơ đồ thông thường.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate sample data
np.random.seed(42)
time = np.arange(1000).reshape(-1, 1)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.2, (1000, 1))
signal[800:820] += 5  # Introduce anomalies

# Train Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(signal)

# Predict anomalies
anomalies = clf.predict(signal) == -1

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Signal')
plt.scatter(time[anomalies], signal[anomalies], color='red', label='Anomalies')
plt.title('Anomaly Detection using Isolation Forest')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Slide 8: Ví dụ thực tế: Giám sát nhiệt độ

Vui lòng xem xét một ví dụ thực tế về giám sát nhiệt độ trong quy trình sản xuất. Chúng tôi sẽ tạo tổng hợp dữ liệu để mô phỏng nhiệt độ theo thời gian và phát hiện những điểm bất ngờ có thể gây ra lỗi cho thiết bị hoặc quá trình cố định quy trình.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate synthetic temperature data
np.random.seed(42)
time = np.arange(1000)
temperature = 20 + 5 * np.sin(2 * np.pi * time / 100) + np.random.normal(0, 1, 1000)
temperature[700:720] += 15  # Simulate equipment malfunction

# Reshape data for Isolation Forest
X = temperature.reshape(-1, 1)

# Train Isolation Forest
clf = IsolationForest(contamination=0.02, random_state=42)
clf.fit(X)

# Predict anomalies
anomalies = clf.predict(X) == -1

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, temperature, label='Temperature')
plt.scatter(time[anomalies], temperature[anomalies], color='red', label='Anomalies')
plt.title('Temperature Monitoring with Anomaly Detection')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

Trang trình bày 9: Ví dụ thực tế: Phân tích lưu lượng mạng

Một ứng dụng thực tế khác thường được phát hiện trong phân tích lưu lượng mạng. Chúng tôi sẽ mô phỏng lượng lưu trữ dữ liệu của quyền truy cập mạng và sử dụng phương pháp Z để xác định các mối đe dọa bảo mật hoặc hoạt động của mạng một cách bất ngờ.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate synthetic network traffic data
np.random.seed(42)
time = np.arange(1000)
traffic = np.random.poisson(50, 1000)  # Normal traffic
traffic[800:820] = np.random.poisson(200, 20)  # Simulate traffic spike

# Detect anomalies using Z-score method
def z_score_anomalies(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

anomalies = z_score_anomalies(traffic, threshold=3)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, traffic, label='Network Traffic')
plt.scatter(time[anomalies], traffic[anomalies], color='red', label='Anomalies')
plt.title('Network Traffic Analysis with Anomaly Detection')
plt.xlabel('Time (minutes)')
plt.ylabel('Traffic Volume (packets/min)')
plt.legend()
plt.show()
```

Slide 10: Xử lý nhiều biến: Phát hiện đa biến bất ngờ

Trong nhiều vấn đề thực tế, chúng ta cần xem xét nhiều biến thể cùng một lúc. Kỹ thuật phát hiện các biến thường gặp có thể xác định các biến thường gặp trong dữ liệu đa chiều của vùng, phức tạp.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope

# Generate multivariate data
np.random.seed(42)
n_samples = 1000
n_outliers = 50
n_features = 2

# Generate normal data
X = np.random.randn(n_samples - n_outliers, n_features)

# Generate outliers
outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, n_features))
X = np.r_[X, outliers]

# Fit the Elliptic Envelope model
ee = EllipticEnvelope(contamination=0.05, random_state=42)
ee.fit(X)

# Predict anomalies
y_pred = ee.predict(X)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Normal')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Anomalies')
plt.title('Multivariate Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

Trang trình bày 11: Dự báo thời gian chuỗi để phát hiện sự bất thường

Kết quả báo cáo chuỗi thời gian hợp nhất nhất với việc phát hiện bất thường có thể mang lại hiệu quả cao. Chúng tôi có thể sử dụng các mô hình dự báo để dự đoán các giá trị kỳ vọng và sau đó xác định những điểm bất ngờ là những sai lệch đáng kể đối với những kỳ vọng này.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate sample data
np.random.seed(42)
time = np.arange(1000)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.2, 1000)
signal[900:920] += 2  # Introduce anomalies

# Fit ARIMA model
model = ARIMA(signal[:800], order=(1, 1, 1))
results = model.fit()

# Make predictions
predictions = results.forecast(steps=200)

# Calculate prediction intervals
pred_int = results.get_forecast(steps=200).conf_int()

# Detect anomalies
anomalies = (signal[800:] < pred_int[:, 0]) | (signal[800:] > pred_int[:, 1])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time[:800], signal[:800], label='Training Data')
plt.plot(time[800:], signal[800:], label='Actual Data')
plt.plot(time[800:], predictions, label='Forecast', color='red')
plt.fill_between(time[800:], pred_int[:, 0], pred_int[:, 1], color='pink', alpha=0.3)
plt.scatter(time[800:][anomalies], signal[800:][anomalies], color='green', label='Anomalies')
plt.title('Time Series Forecasting and Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Trang trình bày 12: Các phương pháp tập hợp để phát hiện trạng thái bất ổn

Việc kết hợp nhiều kỹ thuật phát hiện có thể mang lại kết quả chính xác và chắc chắn hơn. Chúng tôi sẽ trình bày một cách tiếp cận tổng hợp bằng các phương pháp khác nhau mà chúng tôi đã đề xuất.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest

def z_score_anomalies(data, threshold=3):
    return np.abs(stats.zscore(data)) > threshold

def moving_average_anomalies(data, window_size=20, threshold=2):
    ma = np.convolve(data, np.ones(window_size), 'valid') / window_size
    residuals = data[window_size-1:] - ma
    return np.abs(residuals) > threshold * np.std(residuals)

# Generate sample data
np.random.seed(42)
time = np.arange(1000)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.2, 1000)
signal[800:820] += 3  # Introduce anomalies

# Apply different methods
z_score_result = z_score_anomalies(signal)
ma_result = np.pad(moving_average_anomalies(signal), (19, 0), 'constant')
iso_forest = IsolationForest(contamination=0.02, random_state=42)
iso_forest_result = iso_forest.fit_predict(signal.reshape(-1, 1)) == -1

# Combine results (majority voting)
ensemble_result = ((z_score_result.astype(int) +
                    ma_result.astype(int) +
                    iso_forest_result.astype(int)) >= 2)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Signal')
plt.scatter(time[ensemble_result], signal[ensemble_result], color='red', label='Ensemble Anomalies')
plt.title('Ensemble Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()
```

Trang trình bày 13: Đánh giá hiệu suất phát hiện bất thường

Đánh giá hiệu suất của các thuật toán được phát hiện bất ngờ là rất quan trọng. Chúng tôi sẽ khám phá các số liệu phổ biến và kỹ thuật trực quan hóa để đánh giá giá các mô hình của mình.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Generate data with known anomalies
np.random.seed(42)
time = np.arange(1000)
signal = np.sin(time * 0.05) + np.random.normal(0, 0.2, 1000)
true_anomalies = np.zeros(1000, dtype=bool)
true_anomalies[800:820] = True
signal[true_anomalies] += 3

# Detect anomalies using a simple threshold
threshold = np.mean(signal) + 2 * np.std(signal)
detected_anomalies = signal > threshold

# Calculate metrics
cm = confusion_matrix(true_anomalies, detected_anomalies)
precision = precision_score(true_anomalies, detected_anomalies)
recall = recall_score(true_anomalies, detected_anomalies)
f1 = f1_score(true_anomalies, detected_anomalies)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Signal')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(time[detected_anomalies], signal[detected_anomalies], color='red', label='Detected Anomalies')
plt.scatter(time[true_anomalies], signal[true_anomalies], color='green', marker='x', s=100, label='True Anomalies')
plt.title(f'Anomaly Detection Evaluation (Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f})')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.show()

# Print confusion matrix
print("Confusion Matrix:")
print(cm)
```

Trang trình bày 14: Các công thức và cân nhắc bất ngờ trong quá trình phát hiện

Việc phát hiện sự bất thường trong chuỗi dữ liệu đi kèm theo thời gian với nhiều phương thức khác nhau. Chúng ta sẽ thảo luận về một số cân chính và các giải pháp tiềm năng.

1. Khái niệm về dạng trôi: Dữ liệu chuỗi thời gian có thể phát triển theo thời gian, tạo ra các mô hình hiệu quả tĩnh hơn. Giải pháp: Sử dụng thuật toán thích ứng hoặc thường xuyên đào tạo lại các hình.
2. Các mô hình theo mùa: Các mô hình theo mùa phức tạp có thể trộn lẫn với những điều bất thường. Giải pháp: Áp dụng phân rã theo mùa hoặc sử dụng kiến ​​trúc miền để thiết lập mô hình tính thời vụ.
3. Dữ liệu mất cân bằng: Sự việc bất thường rất lạ, dẫn đến vấn đề mất cân bằng lớp. Giải thích: Sử dụng hợp lý các số liệu và kỹ thuật đánh giá để lấy mẫu quá mức hoặc lấy mẫu dưới mức.
4. Chuỗi đa thời gian: Việc xử lý nhiều biến phụ thuộc lẫn nhau có thể phức tạp. Giải pháp: Sử dụng kỹ thuật giảm kích thước hoặc thuật toán phát hiện các biến thông thường.
5. Phát hiện theo thời gian thực: Một số ứng dụng yêu cầu phát hiện sự kiện bất ngờ ngay lập tức. Giải pháp: Triển khai thuật toán phát trực tuyến hoặc sử dụng các phương pháp học trực tuyến hiệu quả.

Trang trình bày 15: Các công thức và cân nhắc bất ngờ trong quá trình phát hiện

```python
# Pseudocode for handling concept drift
def adaptive_anomaly_detection(data_stream):
    model = initialize_model()
    for data_point in data_stream:
        prediction = model.predict(data_point)
        if is_anomaly(prediction):
            report_anomaly(data_point)
        model.update(data_point)
```

Slide 16: Additional Resources

For those interested in diving deeper into anomaly detection in time series data, here are some valuable resources:

1. "Outlier Detection for Temporal Data" by Gupta et al. (2014) ArXiv link: [https://arxiv.org/abs/1401.3665](https://arxiv.org/abs/1401.3665)
2. "A Survey of Deep Learning Techniques for Anomaly Detection in Time Series Data" by Aljohani et al. (2023) ArXiv link: [https://arxiv.org/abs/2305.18415](https://arxiv.org/abs/2305.18415)
3. "Time Series Anomaly Detection; A Survey" by Braei and Wagner (2020) ArXiv link: [https://arxiv.org/abs/2004.00433](https://arxiv.org/abs/2004.00433)

These papers provide comprehensive overviews of various techniques and recent advancements in the field of time series anomaly detection.
