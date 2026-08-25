## Tiêu chuẩn học tập không đúng thời gian để phát hiện chuỗi thời gian bất thường
Trang trình bày 1: Giới thiệu về Học tập chuẩn không thời gian (STEN)

Học tập quy chuẩn không thời gian (STEN) là một kỹ thuật mạnh mẽ để phát hiện những điều bất thường trong dữ liệu thời gian. Nó kết hợp chiều và thời gian để xác định bất kỳ mô hình hoặc hành động nào. STEN đặc hữu ích trong nhiều lĩnh vực khác nhau, coi hạn như mạng cảm biến IoT, phân tích lưu lượng mạng và giám sát môi trường.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating a normal time series
np.random.seed(42)
time = np.arange(100)
normal_series = np.sin(time * 0.1) + np.random.normal(0, 0.1, 100)

# Introducing an anomaly
anomaly_index = 70
normal_series[anomaly_index] += 2

plt.figure(figsize=(10, 5))
plt.plot(time, normal_series)
plt.axvline(x=anomaly_index, color='r', linestyle='--', label='Anomaly')
plt.title('Time Series with Anomaly')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Slide 2: Kích thước tạm thời trong STEN

Khoảng thời gian dài trong STEN tập trung vào tính chất tuần tự đặc biệt của dữ liệu chuỗi. Nó nắm bắt các mô hình và sự phụ thuộc theo các bước thời gian khác nhau, cho phép mô hình tìm hiểu hành vi thời gian bình thường và xác định các sai lệch.

```python
import pandas as pd

# Creating a time series dataset
dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='H')
values = np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 0.1, len(dates))
df = pd.DataFrame({'timestamp': dates, 'value': values})

# Extracting temporal features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

print(df.head())
```

Slide 3: Kích thước không gian trong STEN

Chiều không gian trong STEN xem xét mối quan hệ giữa các biến hoặc cảm biến khác nhau trong chuỗi thời gian đa biến. Nó giúp nắm bắt các mối tương quan và phụ thuộc trên nhiều luồng dữ liệu, cho phép phát hiện các điểm bất ngờ có thể không rõ ràng khi kiểm tra từng biến thể một cách độc lập.

```python
import seaborn as sns

# Simulating multivariate time series data
np.random.seed(42)
n_sensors = 5
n_timestamps = 100
data = np.random.randn(n_timestamps, n_sensors)

# Introducing correlations between sensors
data[:, 1] = data[:, 0] * 0.8 + np.random.randn(n_timestamps) * 0.2
data[:, 2] = data[:, 1] * 0.7 + np.random.randn(n_timestamps) * 0.3

# Visualizing correlations
corr_matrix = np.corrcoef(data.T)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Sensor Data')
plt.show()
```

Trang trình bày 4: Học tập quy tắc tạm thời dựa trên dự đoán (OTN)

OTN là một thành phần của STEN tập trung để tìm hiểu các mẫu thời gian bằng cách mong đợi thứ tự của các sự kiện hoặc giá trị trong chuỗi thời gian. Nó giúp xác định các điểm bất thường bằng cách phát hiện các trình tự không mong muốn hoặc phụ thuộc theo thời gian.

```python
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare data for OTN
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(normal_series.reshape(-1, 1))

# Create sequences for order prediction
seq_length = 10
X, y = [], []
for i in range(len(scaled_data) - seq_length):
    X.append(scaled_data[i:i+seq_length])
    y.append(scaled_data[i+seq_length])

X = np.array(X)
y = np.array(y)

# Build and train OTN model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=0)

# Predict next value
last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
predicted = model.predict(last_sequence)
print(f"Predicted next value: {scaler.inverse_transform(predicted)[0][0]:.2f}")
```

Trang trình bày 5: Học tập quy tắc không dựa trên khoảng cách dự kiến ​​​​(DSN)

DSN là một thành phần khác của tập trung STEN để tìm hiểu các mối quan hệ không gian giữa các biến hoặc cảm biến khác nhau. Nó dự đoán khoảng cách hoặc điểm tương thích giữa dữ liệu điểm trong chiều không gian, xác định những điểm bất thường khác với các mô hình không thông thường.

```python
from sklearn.metrics.pairwise import euclidean_distances

# Simulating multivariate sensor data
n_sensors = 5
n_timestamps = 100
sensor_data = np.random.randn(n_timestamps, n_sensors)

# Calculate pairwise distances between sensors
distances = euclidean_distances(sensor_data.T)

# Visualize distance matrix
plt.figure(figsize=(8, 6))
sns.heatmap(distances, annot=True, cmap='viridis')
plt.title('Pairwise Distances Between Sensors')
plt.xlabel('Sensor ID')
plt.ylabel('Sensor ID')
plt.show()

# Predict distance for a new data point
new_data_point = np.random.randn(1, n_sensors)
predicted_distances = euclidean_distances(new_data_point, sensor_data.T)
print("Predicted distances for new data point:", predicted_distances[0])
```

Slide 6: Kết hợp OTN và DSN trong STEN

STEN kết hợp các điểm mạnh của OTN và DSN để tạo ra một giao diện phát hiện khung bất ngờ. Bằng cách xem xét cả chiều không gian và thời gian, STEN có thể phát hiện ra những phức tạp thường gặp mà các phương pháp truyền thông có thể bỏ qua.

```python
import tensorflow as tf

# Simplified STEN model combining OTN and DSN
class STENModel(tf.keras.Model):
    def __init__(self, seq_length, n_sensors):
        super(STENModel, self).__init__()
        self.lstm = LSTM(50, activation='relu', input_shape=(seq_length, n_sensors))
        self.dense_temporal = Dense(n_sensors)
        self.dense_spatial = Dense(n_sensors * (n_sensors - 1) // 2)

    def call(self, inputs):
        x = self.lstm(inputs)
        temporal_output = self.dense_temporal(x)
        spatial_output = self.dense_spatial(x)
        return temporal_output, spatial_output

# Create and compile the model
seq_length = 10
n_sensors = 5
model = STENModel(seq_length, n_sensors)
model.compile(optimizer='adam', loss=['mse', 'mse'])

# Generate dummy data and train the model
X = np.random.randn(100, seq_length, n_sensors)
y_temporal = np.random.randn(100, n_sensors)
y_spatial = np.random.randn(100, n_sensors * (n_sensors - 1) // 2)
model.fit(X, [y_temporal, y_spatial], epochs=10, verbose=0)

print("STEN model trained successfully")
```

Trang trình bày 7: Tính điểm bất ngờ

Điểm bất thường trong STEN thường được tính toán bằng cách hợp lý nhất các lỗi dự đoán từ cả thành phần thời gian (OTN) và không gian (DSN). Điểm bất thường cao hơn khả năng xảy ra bất thường cao hơn.

```python
def calculate_anomaly_score(temporal_error, spatial_error, alpha=0.5):
    """
    Calculate the anomaly score using a weighted combination of temporal and spatial errors.

    :param temporal_error: Error from the OTN component
    :param spatial_error: Error from the DSN component
    :param alpha: Weight for temporal error (1 - alpha for spatial error)
    :return: Anomaly score
    """
    return alpha * temporal_error + (1 - alpha) * spatial_error

# Simulate prediction errors
temporal_errors = np.abs(np.random.randn(100))
spatial_errors = np.abs(np.random.randn(100))

# Calculate anomaly scores
anomaly_scores = [calculate_anomaly_score(te, se) for te, se in zip(temporal_errors, spatial_errors)]

# Visualize anomaly scores
plt.figure(figsize=(10, 5))
plt.plot(anomaly_scores)
plt.title('Anomaly Scores')
plt.xlabel('Time')
plt.ylabel('Anomaly Score')
plt.show()

print(f"Mean anomaly score: {np.mean(anomaly_scores):.2f}")
print(f"Max anomaly score: {np.max(anomaly_scores):.2f}")
```

Slide 8: Đánh giá chỉ số - Hình vẽ bên dưới đường cong đặc tính hoạt động của máy thu (AUC-ROC)

AUC-ROC là thước đo phổ biến để đánh giá hiệu suất của các mô hình được phát hiện một cách bất thường. Nó đo lường khả năng của mô hình trong công việc phân biệt giữa các dữ liệu bình thường và không mong đợi đối với các giá trị ngưỡng khác nhau.

```python
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

# Simulate true labels and predicted probabilities
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_pred = np.random.rand(1000)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc_roc = roc_auc_score(y_true, y_pred)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

print(f"AUC-ROC score: {auc_roc:.2f}")
```

Trang trình bày 9: Số liệu đánh giá - Khu vực dưới đường cong thu hồi chính xác (AUC-PR)

AUC-PR là một số quan trọng của dữ liệu để phát hiện sự bất ngờ, đặc biệt là khi xử lý các bộ dữ liệu mất cân bằng trong các trường hợp hiếm gặp khi có sự bất ngờ. Nó tập trung vào sự cân bằng giữa độ chính xác và thu hồi ở các giá trị ngưỡng khác nhau.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate precision-recall curve and AUC-PR
precision, recall, _ = precision_recall_curve(y_true, y_pred)
auc_pr = average_precision_score(y_true, y_pred)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR curve (AUC = {auc_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

print(f"AUC-PR score: {auc_pr:.2f}")
```

Slide 10: Chỉ số đánh giá - Điểm F1 tốt nhất

Điểm F1 là giá trị trung bình hài hòa của độ chính xác và khả năng thu hồi. Điểm F1 tốt nhất là điểm F1 cao nhất đạt được trên các ngưỡng giá trị khác nhau, cung cấp thước đo cân bằng về hiệu suất của mô hình.

```python
from sklearn.metrics import f1_score

def find_best_f1_score(y_true, y_pred_proba):
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    best_f1 = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]

    return best_f1, best_threshold

# Find best F1 score and corresponding threshold
best_f1, best_threshold = find_best_f1_score(y_true, y_pred)

print(f"Best F1 score: {best_f1:.2f}")
print(f"Best threshold: {best_threshold:.2f}")

# Plot F1 scores for different thresholds
thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(y_true, (y_pred >= t).astype(int)) for t in thresholds]

plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores)
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best threshold: {best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Threshold')
plt.legend()
plt.show()
```

Trang trình bày 11: Ví dụ thực tế - Phát hiện bất ngờ về việc lưu lượng truy cập mạng

STEN có thể được sử dụng để phân tích mạng lưu trữ nhằm phát hiện các mô hình bất ngờ hoặc các mối đe dọa bảo mật tiềm ẩn. Trong ví dụ này, chúng tôi sẽ mô phỏng lượng truy cập dữ liệu lưu trữ mạng và sử dụng STEN để xác định những điểm bất ngờ.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Simulate network traffic data
np.random.seed(42)
n_samples = 1000
timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='5T')
packet_count = np.random.poisson(lam=100, size=n_samples)
byte_count = packet_count * np.random.randint(100, 1500, size=n_samples)
unique_ips = np.random.randint(10, 100, size=n_samples)

# Introduce anomalies
anomaly_indices = [200, 500, 800]
packet_count[anomaly_indices] *= 10
byte_count[anomaly_indices] *= 15
unique_ips[anomaly_indices] *= 5

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'packet_count': packet_count,
    'byte_count': byte_count,
    'unique_ips': unique_ips
})

# Normalize features
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[['packet_count', 'byte_count', 'unique_ips']])

# Simple anomaly detection using Z-score
z_scores = np.abs(normalized_data).mean(axis=1)
threshold = 3
anomalies = z_scores > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], z_scores, label='Anomaly Score')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(df['timestamp'][anomalies], z_scores[anomalies], color='red', label='Detected Anomalies')
plt.title('Network Traffic Anomaly Detection')
plt.xlabel('Timestamp')
plt.ylabel('Anomaly Score')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Number of detected anomalies: {anomalies.sum()}")
```

Slide 12: Ví dụ thực tế - Giám sát môi trường

STEN can't be apply on the giám sát môi trường để phát hiện các mẫu bất thường trong dữ liệu biến. Ví dụ này mô phỏng dữ liệu nhiệt độ và độ ẩm từ nhiều biến thể và áp dụng phiên bản STEN đơn giản hóa để xác định điểm bất thường.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Simulate environmental sensor data
np.random.seed(42)
n_sensors = 5
n_samples = 1000
timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

# Generate normal patterns with daily and seasonal variations
time = np.arange(n_samples)
base_temp = 20 + 5 * np.sin(2 * np.pi * time / (24 * 365)) + 2 * np.sin(2 * np.pi * time / 24)
base_humidity = 60 + 10 * np.sin(2 * np.pi * time / (24 * 365)) - 5 * np.sin(2 * np.pi * time / 24)

# Create sensor data with some variations
temperatures = np.array([base_temp + np.random.normal(0, 1, n_samples) for _ in range(n_sensors)]).T
humidities = np.array([base_humidity + np.random.normal(0, 2, n_samples) for _ in range(n_sensors)]).T

# Introduce anomalies
anomaly_indices = [200, 500, 800]
temperatures[anomaly_indices] += np.random.uniform(5, 10, size=(len(anomaly_indices), n_sensors))
humidities[anomaly_indices] += np.random.uniform(-20, 20, size=(len(anomaly_indices), n_sensors))

# Combine data
data = np.concatenate([temperatures, humidities], axis=1)

# Normalize data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Simple anomaly detection using Mahalanobis distance
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    return np.sqrt(diff.dot(np.linalg.inv(cov)).dot(diff))

mean = np.mean(normalized_data, axis=0)
cov = np.cov(normalized_data.T)
anomaly_scores = np.array([mahalanobis_distance(x, mean, cov) for x in normalized_data])

# Detect anomalies
threshold = np.percentile(anomaly_scores, 99)
anomalies = anomaly_scores > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(timestamps, anomaly_scores, label='Anomaly Score')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(timestamps[anomalies], anomaly_scores[anomalies], color='red', label='Detected Anomalies')
plt.title('Environmental Monitoring Anomaly Detection')
plt.xlabel('Timestamp')
plt.ylabel('Anomaly Score')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Number of detected anomalies: {anomalies.sum()}")
```

Trang trình bày 13: Định thức và giới hạn của STEN

Mặc dù STEN là một kỹ thuật mạnh mẽ để phát hiện sự bất thường theo thời gian chuỗi nhưng nó phải đối mặt với một số công thức và giới hạn:

1. Phức tạp tính toán cao đối với các bộ dữ liệu lớn
2. Độ nhạy của điều chỉnh siêu thông số
3. Khó khăn trong việc xử lý khái niệm trôi dạt và phát triển các mô hình thông thường
4. Công thức giải thích nguyên nhân gốc của những điều bất ngờ được phát hiện

Để giải quyết những vấn đề này, các nhà nghiên cứu tiếp tục phát triển các biến thể cải tiến của phương pháp STEN và phương pháp kết hợp STEN với các kỹ thuật học máy khác.

```python
# Pseudocode for an adaptive STEN algorithm

class AdaptiveSTEN:
    def __init__(self, window_size, update_frequency):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.model = initialize_sten_model()
        self.data_buffer = []

    def detect_anomalies(self, new_data):
        anomaly_scores = self.model.compute_anomaly_scores(new_data)
        self.data_buffer.extend(new_data)

        if len(self.data_buffer) >= self.update_frequency:
            self.update_model()

        return anomaly_scores

    def update_model(self):
        recent_data = self.data_buffer[-self.window_size:]
        self.model.retrain(recent_data)
        self.data_buffer = []

# Usage
adaptive_sten = AdaptiveSTEN(window_size=1000, update_frequency=100)
for batch in data_stream:
    anomaly_scores = adaptive_sten.detect_anomalies(batch)
    # Process anomaly scores
```

Slide 14: Định hướng tương lai và cơ hội nghiên cứu

Lĩnh vực phát hiện bất thường thời gian bằng STEN tiếp tục phát triển. Một số hướng nghiên cứu đầy hứa hẹn bao gồm:

1. Kết hợp các ý tưởng cơ bản để cải thiện công việc học tập đặc điểm không gian và thời gian
2. Phát triển STEN biến thể không giám sát các vấn đề có dữ liệu được đóng khung theo chế độ
3. Khám phá các phương pháp học chuyển giao để điều chỉnh mô hình STEN trên các lĩnh vực khác nhau
4. Tích hợp các kỹ thuật AI có thể giải quyết để nâng cao khả năng giải quyết kết quả STEN
5. Nghiên cứu ứng dụng STEN trong môi trường điện toán biên dịch để phát hiện sự bất thường theo thời gian thực hiện

```python
# Pseudocode for a STEN model with attention mechanism

import tensorflow as tf

class STENWithAttention(tf.keras.Model):
    def __init__(self, seq_length, n_sensors):
        super(STENWithAttention, self).__init__()
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
        self.attention = tf.keras.layers.Attention()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        lstm_output = self.lstm(inputs)
        attention_output = self.attention([lstm_output, lstm_output])
        return self.dense(attention_output)

# Usage
seq_length = 100
n_sensors = 5
model = STENWithAttention(seq_length, n_sensors)
model.compile(optimizer='adam', loss='mse')

# Train the model
# model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về STEN và các kỹ thuật liên quan để phát hiện sự bất thường theo thời gian chuỗi thì đây là một số tài nguyên có giá trị:

1. "Phát hiện bất thường chuỗi thời gian mạnh mẽ với việc học tính chuẩn hóa không thời gian" của Xu et al. (2023) - ArXiv: [https://arxiv.org/abs/2303.08850](https://arxiv.org/abs/2303.08850)
2. "Khảo sát về khả năng phát hiện bất thường đồ thị bằng chiều sâu học" của Ma và cộng đồng. (2021) - ArXiv: [https://arxiv.org/abs/2106.07178](https://arxiv.org/abs/2106.07178)
3. "Phát hiện bất thường chuỗi thời gian: Khảo sát" của Braei và Wagner (2022) - ArXiv: [https://arxiv.org/abs/2101.02666](https://arxiv.org/abs/2101.02666)

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về STEN và các kỹ thuật liên quan, cũng như tổng quan rộng hơn về các phương pháp phát hiện dị thường theo chuỗi thời gian.
