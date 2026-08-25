## Phát hiện chuỗi bất thường bằng cách học tính chuẩn hóa không thời gian
Trang trình bày 1: Giới thiệu về thời gian phát triển chuỗi dự kiến

Phát hiện sự thật bất thường theo chuỗi thời gian là một nhiệm vụ quan trọng trong nhiều lĩnh vực khác nhau, bao gồm IoT, an ninh mạng và giám sát công nghiệp. Học tập quy chuẩn không thời gian (STEN) là một kỹ thuật tiên tiến kết hợp thông tin không gian và thời gian để xác định sự bất thường trong dữ liệu thời gian chuỗi. Bài trình bày này sẽ khám phá STEN và cách phát triển nó bằng Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample time series with an anomaly
np.random.seed(42)
time = np.arange(100)
normal_data = np.sin(time * 0.1) + np.random.normal(0, 0.1, 100)
anomaly = np.zeros(100)
anomaly[60:70] = 2  # Introduce an anomaly

time_series = normal_data + anomaly

plt.figure(figsize=(12, 6))
plt.plot(time, time_series)
plt.title("Time Series with Anomaly")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
```

Trang trình bày 2: Tìm hiểu tính chất phạm tội không thời gian

STEN xem xét cả khía cạnh không gian và thời gian của dữ liệu để phát hiện sự bất thường. Tính phạm vi không đề cập đến mối quan hệ giữa các biến số hoặc đặc điểm khác nhau tại một thời điểm nhất định, trong khi tính toán phạm vi thời gian tập trung vào các mô hình và xu hướng theo thời gian của từng biến.

```python
import pandas as pd

# Create a sample multivariate time series
dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.sin(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100),
    'humidity': np.cos(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100),
    'pressure': np.tan(np.arange(100) * 0.05) + np.random.normal(0, 0.1, 100)
})

print(df.head())

# Visualize spatial relationships
plt.figure(figsize=(10, 6))
plt.scatter(df['temperature'], df['humidity'], c=df['pressure'], cmap='viridis')
plt.colorbar(label='Pressure')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Spatial Relationship between Variables')
plt.show()
```

Slide 3: Kiến trúc STEN

STEN thường sử dụng kiến ​​trúc deep learning, thường dựa trên bộ mã hóa tự động hoặc mạng thần kinh tái phát (RNN). Mô hình học cách tái tạo các mô hình bình thường theo chiều không gian và thời gian, cho phép mô hình xác định các sai lệch là điểm bất thường.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed

# Define STEN model architecture
def create_sten_model(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, activation='relu')(inputs)
    repeated = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Example usage
input_shape = (10, 3)  # 10 time steps, 3 features
model = create_sten_model(input_shape)
model.summary()
```

Slide 4: Xử lý dữ liệu tiền tệ cho STEN

Trước khi áp dụng STEN, điều quan trọng phải được xử lý trước dữ liệu chuỗi thời gian. Điều này liên quan đến công việc chuẩn hóa, xử lý việc thiếu các giá trị và tạo cảnh cửa sổ tạm thời.

```python
from sklearn.preprocessing import MinMaxScaler

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[['temperature', 'humidity', 'pressure']])

# Create sliding windows
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 10
X = create_sequences(normalized_data, seq_length)

print("Shape of input sequences:", X.shape)
```

Slide 5: Huấn luyện mô hình STEN

Huấn luyện mô hình STEN bao gồm việc sử dụng dữ liệu đã được xử lý trước đó để tìm hiểu các thông số mẫu. Mô hình được huấn luyện để tái sử dụng cấu trúc chuỗi, giảm thiểu lỗi tái sử dụng cấu trúc đối với dữ liệu thông thường.

```python
# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]

# Train the model
model = create_sten_model((seq_length, 3))
history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 6: Phát hiện bất ngờ bằng STEN

Sau khi được đào tạo, mô hình STEN có thể phát hiện các điểm bất thường bằng cách so sánh lỗi tái tạo của dữ liệu mới với một ngưỡng. Lỗi thiết bị tái sinh được gắn cờ là nơi ẩn náu bất ngờ.

```python
# Predict on test data
X_pred = model.predict(X_test)

# Calculate reconstruction error
mse = np.mean(np.power(X_test - X_pred, 2), axis=(1, 2))

# Set threshold for anomaly detection
threshold = np.mean(mse) + 2 * np.std(mse)

# Identify anomalies
anomalies = mse > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(mse)
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error')
plt.xlabel('Sample')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

print(f"Number of anomalies detected: {np.sum(anomalies)}")
```

Slide 7: Ví dụ thực tế: Giám sát môi trường

STEN có thể được áp dụng cho hệ thống giám sát môi trường để phát hiện các mẫu bất thường trong dữ liệu biến. Ví dụ, trong một dự án thành phố thông minh, các phản ứng biến đổi sẽ thu thập dữ liệu về chất lượng không khí, nhiệt độ và độ ồn.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Simulated environmental data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.sin(np.arange(1000) * 0.02) + np.random.normal(20, 5, 1000),
    'humidity': np.cos(np.arange(1000) * 0.02) + np.random.normal(60, 10, 1000),
    'air_quality': np.random.normal(50, 10, 1000)
})

# Introduce anomalies
df.loc[500:520, 'air_quality'] += 100  # Sudden spike in air pollution

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['temperature', 'humidity', 'air_quality']])

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 24  # Use 24 hours of data to predict the next hour
X = create_sequences(scaled_data, seq_length)

# Train STEN model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 3), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(3)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:-1], scaled_data[seq_length:], epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Detect anomalies
predictions = model.predict(X)
mse = np.mean(np.power(X[:, -1, :] - predictions, 2), axis=1)
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'][seq_length:], df['air_quality'][seq_length:], label='Actual')
plt.scatter(df['timestamp'][seq_length:][anomalies], df['air_quality'][seq_length:][anomalies], color='red', label='Anomaly')
plt.title('Air Quality Monitoring with Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Air Quality Index')
plt.legend()
plt.show()
```

Trang trình bày 8: Ví dụ thực tế: Phân tích lưu lượng mạng

STEN có thể được sử dụng trong mạng lưới để phát hiện các mẫu bất thường trong lượng truy cập được lưu trữ trên mạng, có khả năng chỉ ra các cuộc tấn công mạng hoặc mạng cố gắng nỗ lực.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Simulated network traffic data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')
df = pd.DataFrame({
    'timestamp': dates,
    'incoming_traffic': np.random.poisson(100, 1000),
    'outgoing_traffic': np.random.poisson(80, 1000),
    'active_connections': np.random.poisson(50, 1000)
})

# Introduce anomalies
df.loc[800:820, 'incoming_traffic'] *= 5  # Sudden spike in incoming traffic

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['incoming_traffic', 'outgoing_traffic', 'active_connections']])

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 12  # Use 1 hour of data to predict the next 5 minutes
X = create_sequences(scaled_data, seq_length)

# Train STEN model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 3), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(3)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:-1], scaled_data[seq_length:], epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Detect anomalies
predictions = model.predict(X)
mse = np.mean(np.power(X[:, -1, :] - predictions, 2), axis=1)
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'][seq_length:], df['incoming_traffic'][seq_length:], label='Actual')
plt.scatter(df['timestamp'][seq_length:][anomalies], df['incoming_traffic'][seq_length:][anomalies], color='red', label='Anomaly')
plt.title('Network Traffic Monitoring with Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Incoming Traffic (packets)')
plt.legend()
plt.show()
```

Slide 9: Xử lý các thành phần theo mùa và theo hướng

Nhiều chuỗi thời gian có thể hiện các mô hình theo mùa và theo giới hạn thời gian. STEN có thể được tăng cường để xử lý các thành phần này bằng cách kết hợp các kỹ thuật như phân tích theo mùa hoặc sử dụng các kiến ​​trúc phức tạp hơn.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample data with trend and seasonality
np.random.seed(42)
time = np.arange(1000)
trend = 0.02 * time
seasonality = 10 * np.sin(2 * np.pi * time / 365.25)
noise = np.random.normal(0, 1, 1000)
data = trend + seasonality + noise

# Perform seasonal decomposition
result = seasonal_decompose(data, model='additive', period=365)

# Visualize components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()

# Use residuals for anomaly detection
residuals = result.resid.dropna().values
scaler = MinMaxScaler()
scaled_residuals = scaler.fit_transform(residuals.reshape(-1, 1))

# Create sequences and train STEN model on residuals
X = create_sequences(scaled_residuals, seq_length=24)
model = create_sten_model((24, 1))
model.fit(X[:-1], X[:-1], epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Detect anomalies in residuals
predictions = model.predict(X)
mse = np.mean(np.power(X - predictions, 2), axis=(1, 2))
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals')
plt.scatter(np.where(anomalies)[0], residuals[anomalies], color='red', label='Anomaly')
plt.title('Anomaly Detection on Residuals')
plt.xlabel('Time')
plt.ylabel('Residual Value')
plt.legend()
plt.show()
```

Slide 10: Xử lý đa thời gian của chuỗi

STEN có thể được mở rộng để xử lý nhiều chuỗi thời gian biến thể, trong đó nhiều biến thể được quan sát đồng thời. Điều này đặc biệt hữu ích trong các hệ thống phức tạp, nơi mà sự bất thường có thể biểu hiện trên nhiều chiều.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate multivariate time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.sin(np.arange(1000) * 0.02) + np.random.normal(20, 2, 1000),
    'humidity': np.cos(np.arange(1000) * 0.02) + np.random.normal(60, 5, 1000),
    'pressure': np.random.normal(1013, 5, 1000),
    'wind_speed': np.abs(np.random.normal(0, 5, 1000))
})

# Introduce correlated anomalies
df.loc[500:510, ['temperature', 'humidity']] += [10, -20]

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['temperature', 'humidity', 'pressure', 'wind_speed']])

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 24
X = create_sequences(scaled_data, seq_length)

# Train STEN model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 4), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(4)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:-1], scaled_data[seq_length:], epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Detect anomalies
predictions = model.predict(X)
mse = np.mean(np.power(X[:, -1, :] - predictions, 2), axis=1)
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'][seq_length:], df['temperature'][seq_length:], label='Temperature')
plt.plot(df['timestamp'][seq_length:], df['humidity'][seq_length:], label='Humidity')
plt.scatter(df['timestamp'][seq_length:][anomalies],
            df['temperature'][seq_length:][anomalies],
            color='red', marker='x', s=50, label='Anomaly')
plt.title('Multivariate Time Series Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Slide 11: Tầm quan trọng của tính năng trong STEN

Việc hiểu rõ những tính năng nào được đóng góp nhiều nhất vào việc phát hiện công việc thường có thể cung cấp những tính năng hiểu biết có giá trị. Các kỹ thuật như SHAP (SHapley Additive exPlanations) có thể được sử dụng để giải mã các mô hình STEN và xác định các biến có ảnh hưởng nhiều nhất.

```python
import shap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Create a simplified STEN model for interpretation
input_shape = (seq_length, 4)
inputs = Input(shape=input_shape)
lstm = LSTM(32, activation='relu')(inputs)
outputs = Dense(4)(lstm)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, scaled_data[seq_length:], epochs=50, batch_size=32, verbose=0)

# Create an explainer
explainer = shap.DeepExplainer(model, X[:100])

# Calculate SHAP values
shap_values = explainer.shap_values(X[100:110])

# Visualize feature importance
shap.summary_plot(shap_values[0], X[100:110], feature_names=['Temperature', 'Humidity', 'Pressure', 'Wind Speed'])
```

Slide 12: Các phương pháp tập hợp cho STEN

Các phương pháp tập hợp có thể cải thiện độ tin cậy và độ chính xác của mô hình STEN bằng cách kết hợp nhiều mô hình hoặc kỹ thuật. Cách tiếp cận này có thể giúp nắm bắt các khía cạnh khác của dữ liệu và giảm các kết quả dương tính.

```python
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# STEN model
sten_model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 4), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(4)
])
sten_model.compile(optimizer='adam', loss='mse')
sten_model.fit(X, scaled_data[seq_length:], epochs=50, batch_size=32, verbose=0)

# Isolation Forest model
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(scaled_data)

# Combine predictions
sten_mse = np.mean(np.power(X[:, -1, :] - sten_model.predict(X), 2), axis=1)
sten_anomalies = sten_mse > np.mean(sten_mse) + 2 * np.std(sten_mse)

iso_anomalies = iso_forest.predict(scaled_data) == -1

# Ensemble anomaly detection
ensemble_anomalies = sten_anomalies[seq_length:] & iso_anomalies[seq_length:]

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'][seq_length:], df['temperature'][seq_length:], label='Temperature')
plt.scatter(df['timestamp'][seq_length:][ensemble_anomalies],
            df['temperature'][seq_length:][ensemble_anomalies],
            color='red', marker='x', s=50, label='Ensemble Anomaly')
plt.title('Ensemble Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()
```

Slide 13: Thử nghiệm và định hướng tương lai

Mặc dù STEN là một kỹ thuật mạnh mẽ để phát hiện sự bất thường theo thời gian chuỗi nhưng nó phải đối mặt với một số công thức:

1. Xử lý khái niệm dạng trôi và mô hình phát triển trong dữ liệu chuỗi thời gian
2. Cân bằng mức độ phức tạp của mô hình với khả năng giải quyết của diễn đàn
3. Xử lý các sự kiện cực thú hoặc tình huống thiên nga đen
4. Thích ứng với các loại dị thường khác nhau (dị thường điểm, ngữ cảnh và tập tin)

Các hướng nghiên cứu trong tương lai bao gồm:

1. Kết hợp các kỹ thuật học chuyển giao để điều chỉnh mô hình STEN trên các lĩnh vực khác nhau
2. Phát triển các phương pháp học tập tự giám sát cho STEN để tận dụng dữ liệu chưa được gắn nhãn
3. Khám phá liên kết để phát hiện những điểm bất ngờ được đảm bảo quyền riêng tư trong hệ thống phân tán
4. Tích hợp các kỹ thuật AI có thể giải quyết thích hợp để cải thiện khả năng giải quyết và độ tin cậy của mô hình

```python
# Pseudocode for an adaptive STEN model

class AdaptiveSTEN:
    def __init__(self, input_shape, learning_rate):
        self.model = create_sten_model(input_shape)
        self.learning_rate = learning_rate

    def detect_anomalies(self, data):
        predictions = self.model.predict(data)
        errors = calculate_reconstruction_error(data, predictions)
        return errors > self.calculate_threshold(errors)

    def update_model(self, new_data):
        self.model.fit(new_data, new_data, epochs=1, batch_size=32)

    def calculate_threshold(self, errors):
        return np.mean(errors) + 2 * np.std(errors)

    def adapt_to_concept_drift(self, data_stream):
        for batch in data_stream:
            anomalies = self.detect_anomalies(batch)
            self.update_model(batch[~anomalies])  # Update model with non-anomalous data
            yield anomalies

# Usage
adaptive_sten = AdaptiveSTEN(input_shape=(24, 4), learning_rate=0.001)
for anomalies in adaptive_sten.adapt_to_concept_drift(data_stream):
    process_anomalies(anomalies)
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về STEN và các kỹ thuật liên quan, đây là một số tài nguyên có giá trị:

1. “Học sâu để dự báo thời gian chuỗi” của N. Laptev, J. Yosinski, L. E. Li và S. Smyl (2017). ArXiv:1701.01887 \[cs.LG\] [https://arxiv.org/abs/1701.01887](https://arxiv.org/abs/1701.01887)
2. "Phát hiện bất ngờ trong thời gian chuỗi: Đánh giá toàn diện" của G. Bontempi, S. Ben Taieb và Y.-A. Lê Borgne (2021). ArXiv:2103.16236 \[cs.LG\] [https://arxiv.org/abs/2103.16236](https://arxiv.org/abs/2103.16236)
3. "Khảo sát học sâu để dự báo thời gian chuỗi" của B. Lim và S. Zohren (2020). ArXiv:2004.13408 \[cs.LG\] [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về các khía cạnh khác nhau của phân tích chuỗi thời gian, phương pháp học sâu và kỹ thuật phát hiện bất thường, có thể bổ sung cho sự hiểu biết về STEN và các ứng dụng của nó.
