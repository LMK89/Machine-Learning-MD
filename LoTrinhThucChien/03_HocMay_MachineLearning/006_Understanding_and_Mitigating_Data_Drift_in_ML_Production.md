## Hiểu và giảm thiểu dữ liệu trôi dạt trong ML sản phẩm
Trang trình bày 1: Dữ liệu trôi dạt trong ML sản phẩm

Format dữ liệu trôi dạt đến sự thay đổi danh sách thuộc tính của các tính năng đầu vào theo thời gian. Đó là một khái niệm quan trọng trong học máy, đặc biệt đối với các hình thức đã phát triển. Công việc giám sát liên tục dữ liệu trôi trong môi trường sản xuất là điều cần thiết để duy trì hiệu suất và độ tin cậy của mô hình.

Trang trình bày 2: Mã nguồn cho dữ liệu dạng trôi trong ML sản phẩm

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def detect_data_drift(reference_data, current_data, threshold=0.05):
    _, p_value = ks_2samp(reference_data, current_data)
    return p_value < threshold

# Generate sample data
np.random.seed(42)
reference_data = np.random.normal(0, 1, 1000)
current_data_no_drift = np.random.normal(0, 1, 1000)
current_data_with_drift = np.random.normal(0.5, 1.2, 1000)

# Detect drift
drift_detected_no_drift = detect_data_drift(reference_data, current_data_no_drift)
drift_detected_with_drift = detect_data_drift(reference_data, current_data_with_drift)

print(f"Drift detected (no drift): {drift_detected_no_drift}")
print(f"Drift detected (with drift): {drift_detected_with_drift}")

# Visualize distributions
plt.figure(figsize=(10, 5))
plt.hist(reference_data, bins=30, alpha=0.5, label='Reference')
plt.hist(current_data_with_drift, bins=30, alpha=0.5, label='Current (with drift)')
plt.legend()
plt.title('Data Distribution Comparison')
plt.show()
```

Trang trình bày 3: Kết quả về dữ liệu trôi dạt trong ML sản phẩm

```
Drift detected (no drift): False
Drift detected (with drift): True
```

Slide 4: Importance of Data Drift Monitoring

Continuous data drift monitoring in production is crucial for maintaining model performance. It helps identify changes in input data distribution that may affect model accuracy. Early detection of drift allows for timely model updates and prevents degradation of predictive power.

Slide 5: Case 1: Delayed or Rare Target Information

In scenarios where true target values are infrequently available, undetected data drift can lead to prolonged periods of inaccurate predictions. A data drift detection system serves as a proxy for potential poor model performance, alerting you to issues before they significantly impact your system.

Slide 6: Source Code for Case 1: Delayed or Rare Target Information

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_delayed_target_scenario(days, drift_start, drift_magnitude):
    np.random.seed(42)
    predictions = np.random.normal(0, 1, days)
    true_values = np.random.normal(0, 1, days)

    # Introduce drift
    drift = np.linspace(0, drift_magnitude, days - drift_start)
    predictions[drift_start:] += drift

    # Simulate delayed target availability
    available_targets = np.full(days, np.nan)
    available_targets[::5] = true_values[::5]  # Every 5th day

    return predictions, true_values, available_targets

days = 100
drift_start = 50
drift_magnitude = 2

predictions, true_values, available_targets = simulate_delayed_target_scenario(days, drift_start, drift_magnitude)

plt.figure(figsize=(12, 6))
plt.plot(predictions, label='Predictions')
plt.plot(true_values, label='True Values')
plt.scatter(range(days), available_targets, color='red', label='Available Targets', alpha=0.5)
plt.axvline(x=drift_start, color='green', linestyle='--', label='Drift Start')
plt.legend()
plt.title('Delayed Target Information Scenario')
plt.xlabel('Days')
plt.ylabel('Values')
plt.show()
```

Slide 7: Trường hợp 2: Gỡ bỏ lỗi hiệu suất màn hình

Khi hiệu suất mô hình suy giảm, phân tích dữ liệu trôi dạt có thể cung cấp những hiểu biết có giá trị. Nó giúp xác định nguyên nhân gốc rễ hiệu quả và hướng dẫn các quyết định về chiến lược đào tạo lại. Hệ thống giám sát dữ liệu có thể thúc đẩy nhanh quá trình này, tiết kiệm thời gian và tài nguyên trong nỗ lực gỡ lỗi.

Slide 8: Mã nguồn cho trường hợp 2: Xóa hiệu suất lỗi

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def analyze_feature_drift(reference_data, current_data, feature_names, threshold=0.05):
    drifted_features = []
    for i, feature in enumerate(feature_names):
        _, p_value = ks_2samp(reference_data[:, i], current_data[:, i])
        if p_value < threshold:
            drifted_features.append(feature)
    return drifted_features

# Generate sample data
np.random.seed(42)
feature_names = ['Feature A', 'Feature B', 'Feature C', 'Feature D']
reference_data = np.random.normal(0, 1, (1000, 4))
current_data = np.random.normal(0, 1, (1000, 4))

# Introduce drift in Feature B and Feature D
current_data[:, 1] += 0.5
current_data[:, 3] *= 1.2

drifted_features = analyze_feature_drift(reference_data, current_data, feature_names)

print("Drifted features:", drifted_features)

# Visualize drifted features
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
for i, ax in enumerate(axs.flatten()):
    ax.hist(reference_data[:, i], bins=30, alpha=0.5, label='Reference')
    ax.hist(current_data[:, i], bins=30, alpha=0.5, label='Current')
    ax.set_title(feature_names[i])
    ax.legend()

plt.tight_layout()
plt.show()
```

Trang trình bày 9: Kết quả cho trường hợp 2: Loại bỏ lỗi hiệu suất màn hình

```
Drifted features: ['Feature B', 'Feature D']
```

Slide 10: Case 3: Identifying Errors in Target Data

Data drift monitoring can also help identify issues with newly collected target data. If features haven't drifted but the model shows high error rates, it might indicate problems with the target data. This insight allows you to exclude erroneous data from error computations and model retraining.

Slide 11: Source Code for Case 3: Identifying Errors in Target Data

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_target_data_error(days, error_start, error_magnitude):
    np.random.seed(42)
    features = np.random.normal(0, 1, (days, 3))
    true_targets = np.sum(features, axis=1) + np.random.normal(0, 0.1, days)

    # Introduce error in target data
    erroneous_targets = true_targets.copy()
    erroneous_targets[error_start:] += np.random.normal(error_magnitude, 0.5, days - error_start)

    return features, true_targets, erroneous_targets

days = 100
error_start = 70
error_magnitude = 2

features, true_targets, erroneous_targets = simulate_target_data_error(days, error_start, error_magnitude)

# Calculate errors
true_errors = np.abs(np.sum(features, axis=1) - true_targets)
erroneous_errors = np.abs(np.sum(features, axis=1) - erroneous_targets)

plt.figure(figsize=(12, 6))
plt.plot(true_errors, label='True Errors')
plt.plot(erroneous_errors, label='Erroneous Errors')
plt.axvline(x=error_start, color='red', linestyle='--', label='Error Introduction')
plt.legend()
plt.title('Impact of Target Data Errors on Model Performance')
plt.xlabel('Days')
plt.ylabel('Absolute Error')
plt.show()
```

Slide 12: Ví dụ thực tế: Dự báo thời tiết

Vui lòng xem xét một mô hình báo cáo thời tiết được phát triển ở một thành phố ven biển. Theo thời gian, các biến đổi khí hậu ảnh hưởng đến nhiệt độ mô hình, dẫn đến sai lệch dữ liệu. Mô hình được đào tạo dựa trên dữ liệu lịch sử bắt đầu đưa ra những dự đoán không chính xác. Công việc giám sát liên tục phân phối nhiệt độ giúp xác định sai lệch này, nhắc nhở cập nhật kịp thời để duy trì chính xác dự án.

Slide 13: Ví dụ thực tế: Hệ thống khuyến nghị thương mại điện tử

Nền tảng thương mại điện tử sử dụng hệ thống sản xuất dựa trên lịch sử trình duyệt web của người dùng. Trong thời kỳ đại dịch toàn cầu, hành vi của người dùng thay đổi đáng kể, khiến dữ liệu bị trôi dạt trong các tính năng như tùy chọn danh mục sản phẩm và kiểu duyệt web thời gian. Tính năng phát hiện dữ liệu trôi dạt sẽ cảnh báo cho nhóm về những thay đổi này, cho phép họ điều chỉnh thuật toán xuất ra phù hợp với điều kiện thông thường mới, duy trì các chủ đề sản xuất sản phẩm có liên quan.

Trang trình bày 14: Thực hiện phát hiện sai lệch dữ liệu

Để phát triển tính năng phát hiện sai lệch dữ liệu, bạn có thể sử dụng các thử nghiệm thống kê như thử nghiệm Kolmogorov-Smirnov hoặc các phương pháp dựa trên số dân số ổn định. So sánh thường xuyên giữa tham số tham chiếu (được sử dụng để đào tạo) và sản phẩm xuất hiện tại có thể tiết lộ những thay đổi đáng kể về phân phối.

Trang trình bày 15: Mã nguồn để phát triển tính năng phát hiện sai lệch dữ liệu

```python
import numpy as np
from scipy.stats import ks_2samp

def detect_feature_drift(reference_data, current_data, feature_names, threshold=0.05):
    drift_results = {}
    for i, feature in enumerate(feature_names):
        statistic, p_value = ks_2samp(reference_data[:, i], current_data[:, i])
        drift_detected = p_value < threshold
        drift_results[feature] = {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'statistic': statistic
        }
    return drift_results

# Example usage
np.random.seed(42)
feature_names = ['user_age', 'session_duration', 'pages_visited', 'cart_value']
reference_data = np.random.normal(0, 1, (1000, 4))
current_data = np.random.normal(0, 1, (1000, 4))

# Introduce drift in 'session_duration' and 'cart_value'
current_data[:, 1] += 0.5  # Shift in session duration
current_data[:, 3] *= 1.2  # Scale change in cart value

drift_results = detect_feature_drift(reference_data, current_data, feature_names)

for feature, result in drift_results.items():
    print(f"{feature}: Drift detected: {result['drift_detected']}, p-value: {result['p_value']:.4f}")
```

Trang trình bày 16: Kết quả thực hiện phát hiện sai lệch dữ liệu

```
user_age: Drift detected: False, p-value: 0.8295
session_duration: Drift detected: True, p-value: 0.0000
pages_visited: Drift detected: False, p-value: 0.9809
cart_value: Drift detected: True, p-value: 0.0000
```

Slide 17: Additional Resources

For more information on data drift in machine learning, consider the following resources:

1.  "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift" by Rabanser et al. (2019) - ArXiv:1810.11953
2.  "A Survey on Concept Drift Adaptation" by Gama et al. (2014) - ACM Computing Surveys, Vol. 46, No. 4, Article 44
3.  "Learning under Concept Drift: A Review" by Lu et al. (2018) - ArXiv:1810.11944

These papers provide in-depth discussions on data drift detection methods and adaptation strategies in machine learning systems.
