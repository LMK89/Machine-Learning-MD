## Đánh giá Entropy truyền cho phân phối chuẩn và phân phối Gamma trong Python
Slide 1: Giới thiệu về Entropy truyền

Entropy truyền là thước đo truyền thông tin có hướng giữa hai quá trình ngẫu nhiên. Nó định lượng mức độ không chắc chắn giảm đi trong các giá trị tương lai của một quá trình bằng cách biết các giá trị trong quá khứ của một quá trình khác, ngoài mức độ không chắc chắn đã giảm đi khi biết quá khứ của chính nó.

```python
import numpy as np
from scipy import stats

def transfer_entropy(source, target, k=1, l=1):
    """
    Calculate transfer entropy from source to target.
    k: history length for target
    l: history length for source
    """
    joint = np.array(list(zip(target[k:], target[:len(target)-k], source[:len(source)-l])))
    p_joint = stats.gaussian_kde(joint.T)(joint.T)
    p_cond_target = stats.gaussian_kde(joint[:, :2].T)(joint[:, :2].T)
    p_cond_both = stats.gaussian_kde(joint[:, [0, 1, 2]].T)(joint[:, [0, 1, 2]].T)
    return np.mean(np.log2(p_cond_both / p_cond_target))

# Example usage
np.random.seed(0)
source = np.random.normal(0, 1, 1000)
target = np.roll(source, 1) + np.random.normal(0, 0.1, 1000)

te = transfer_entropy(source, target)
print(f"Transfer entropy: {te:.4f}")
```

Trang trình bày 2: Phân phối bình thường

Phân phối chuẩn, còn được gọi là phân phối Gaussian, là phân bố xác suất liên tục được đặc trưng bởi đường cong hình chuông. Nó đối xứng về giá trị trung bình và được xác định đầy đủ bởi hai tham số: giá trị trung bình (μ) và độ lệch chuẩn (σ).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data points
x = np.linspace(-5, 5, 1000)

# Create normal distributions
mu1, sigma1 = 0, 1
mu2, sigma2 = 1, 1.5

y1 = norm.pdf(x, mu1, sigma1)
y2 = norm.pdf(x, mu2, sigma2)

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=f'μ={mu1}, σ={sigma1}')
plt.plot(x, y2, label=f'μ={mu2}, σ={sigma2}')
plt.title('Normal Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 3: Phân bố gamma

Phân bố gamma là phân bố xác suất liên tục với hai tham số: hình dạng (k) và tỷ lệ (θ). Nó thường được sử dụng để mô hình hóa thời gian chờ đợi và là sự tổng quát hóa của phân bố hàm mũ và phân bố chi bình phương.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Generate data points
x = np.linspace(0, 20, 1000)

# Create gamma distributions
k1, theta1 = 2, 2
k2, theta2 = 5, 1

y1 = gamma.pdf(x, k1, scale=theta1)
y2 = gamma.pdf(x, k2, scale=theta2)

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=f'k={k1}, θ={theta1}')
plt.plot(x, y2, label=f'k={k2}, θ={theta2}')
plt.title('Gamma Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 4: Tạo phân phối chuẩn và phân phối Gamma

Để đánh giá entropy truyền, trước tiên chúng ta cần tạo dữ liệu từ phân phối chuẩn và phân phối gamma. Đây là cách chúng tôi có thể tạo dữ liệu tổng hợp bằng NumPy:

```python
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate normal distribution
mu, sigma = 0, 1
normal_data = np.random.normal(mu, sigma, 1000)

# Generate gamma distribution
k, theta = 2, 2
gamma_data = np.random.gamma(k, theta, 1000)

print("Normal distribution statistics:")
print(f"Mean: {np.mean(normal_data):.4f}")
print(f"Standard deviation: {np.std(normal_data):.4f}")

print("\nGamma distribution statistics:")
print(f"Mean: {np.mean(gamma_data):.4f}")
print(f"Standard deviation: {np.std(gamma_data):.4f}")
```

Slide 5: Tính Entropy truyền

Bây giờ chúng ta đã có dữ liệu, hãy tính entropy truyền giữa phân bố chuẩn và phân bố gamma. Chúng ta sẽ sử dụng hàm `transfer_entropy` được xác định trước đó:

```python
def transfer_entropy(source, target, k=1, l=1):
    joint = np.array(list(zip(target[k:], target[:len(target)-k], source[:len(source)-l])))
    p_joint = stats.gaussian_kde(joint.T)(joint.T)
    p_cond_target = stats.gaussian_kde(joint[:, :2].T)(joint[:, :2].T)
    p_cond_both = stats.gaussian_kde(joint[:, [0, 1, 2]].T)(joint[:, [0, 1, 2]].T)
    return np.mean(np.log2(p_cond_both / p_cond_target))

# Calculate transfer entropy
te_normal_to_gamma = transfer_entropy(normal_data, gamma_data)
te_gamma_to_normal = transfer_entropy(gamma_data, normal_data)

print(f"Transfer entropy (Normal to Gamma): {te_normal_to_gamma:.4f}")
print(f"Transfer entropy (Gamma to Normal): {te_gamma_to_normal:.4f}")
```

Trang trình bày 6: Giải thích kết quả Entropy chuyển giao

Các giá trị entropy truyền mà chúng tôi đã tính toán cung cấp cái nhìn sâu sắc về luồng thông tin giữa phân phối chuẩn và phân phối gamma. Giá trị cao hơn cho thấy việc truyền thông tin mạnh hơn, trong khi giá trị gần bằng 0 cho thấy việc truyền thông tin tối thiểu.

```python
import matplotlib.pyplot as plt

# Visualize the transfer entropy results
plt.figure(figsize=(10, 6))
plt.bar(['Normal to Gamma', 'Gamma to Normal'], [te_normal_to_gamma, te_gamma_to_normal])
plt.title('Transfer Entropy Between Normal and Gamma Distributions')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(axis='y')
plt.show()

# Interpret the results
if te_normal_to_gamma > te_gamma_to_normal:
    print("The normal distribution provides more information about the gamma distribution than vice versa.")
elif te_normal_to_gamma < te_gamma_to_normal:
    print("The gamma distribution provides more information about the normal distribution than vice versa.")
else:
    print("The information transfer between the normal and gamma distributions is symmetric.")
```

Trang trình bày 7: Entropy chuyển giao có độ trễ thời gian

Entropy truyền cũng có thể được tính toán với độ trễ thời gian để khám phá việc truyền thông tin bị trì hoãn. Hãy triển khai một hàm để tính entropy truyền có độ trễ thời gian:

```python
def time_lagged_transfer_entropy(source, target, lag, k=1, l=1):
    source_lagged = np.roll(source, lag)
    return transfer_entropy(source_lagged, target, k, l)

# Calculate time-lagged transfer entropy for different lags
lags = range(-10, 11)
te_values = [time_lagged_transfer_entropy(normal_data, gamma_data, lag) for lag in lags]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(lags, te_values)
plt.title('Time-Lagged Transfer Entropy (Normal to Gamma)')
plt.xlabel('Time Lag')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(True)
plt.show()

# Find the lag with maximum transfer entropy
max_lag = lags[np.argmax(te_values)]
print(f"Maximum transfer entropy occurs at lag {max_lag}")
```

Trang trình bày 8: So sánh Entropy truyền với mối tương quan

Trong khi entropy truyền đo lường luồng thông tin định hướng, thì mối tương quan đo lường mối quan hệ tuyến tính giữa các biến. Hãy so sánh hai biện pháp này:

```python
from scipy.stats import pearsonr

# Calculate Pearson correlation
correlation, _ = pearsonr(normal_data, gamma_data)

# Calculate transfer entropy in both directions
te_normal_to_gamma = transfer_entropy(normal_data, gamma_data)
te_gamma_to_normal = transfer_entropy(gamma_data, normal_data)

print(f"Pearson correlation: {correlation:.4f}")
print(f"Transfer entropy (Normal to Gamma): {te_normal_to_gamma:.4f}")
print(f"Transfer entropy (Gamma to Normal): {te_gamma_to_normal:.4f}")

# Visualize the comparison
plt.figure(figsize=(10, 6))
plt.bar(['Correlation', 'TE (N to G)', 'TE (G to N)'], [abs(correlation), te_normal_to_gamma, te_gamma_to_normal])
plt.title('Comparison of Correlation and Transfer Entropy')
plt.ylabel('Magnitude')
plt.grid(axis='y')
plt.show()
```

Trang trình chiếu 9: Ví dụ thực tế: Phân tích thị trường chứng khoán

Hãy áp dụng entropy chuyển giao để phân tích luồng thông tin giữa hai giá cổ phiếu. Chúng tôi sẽ sử dụng Yahoo Finance để lấy dữ liệu chứng khoán thực:

```python
import yfinance as yf
import pandas as pd

# Fetch stock data
apple = yf.Ticker("AAPL")
google = yf.Ticker("GOOGL")

start_date = "2022-01-01"
end_date = "2023-01-01"

apple_data = apple.history(start=start_date, end=end_date)['Close']
google_data = google.history(start=start_date, end=end_date)['Close']

# Calculate daily returns
apple_returns = apple_data.pct_change().dropna()
google_returns = google_data.pct_change().dropna()

# Calculate transfer entropy
te_apple_to_google = transfer_entropy(apple_returns, google_returns)
te_google_to_apple = transfer_entropy(google_returns, apple_returns)

print(f"Transfer entropy (Apple to Google): {te_apple_to_google:.4f}")
print(f"Transfer entropy (Google to Apple): {te_google_to_apple:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(['Apple to Google', 'Google to Apple'], [te_apple_to_google, te_google_to_apple])
plt.title('Transfer Entropy Between Apple and Google Stock Returns')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(axis='y')
plt.show()
```

Trang trình bày 10: Ví dụ thực tế: Phân tích dữ liệu khí hậu

Một ứng dụng khác của entropy truyền là trong phân tích dữ liệu khí hậu. Hãy kiểm tra luồng thông tin giữa nhiệt độ và độ ẩm:

```python
import pandas as pd
import numpy as np

# Generate synthetic climate data
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
temperature = np.random.normal(20, 5, len(dates)) + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
humidity = 50 + 0.5 * temperature + np.random.normal(0, 5, len(dates))

climate_data = pd.DataFrame({"Temperature": temperature, "Humidity": humidity}, index=dates)

# Calculate transfer entropy
te_temp_to_humid = transfer_entropy(climate_data['Temperature'], climate_data['Humidity'])
te_humid_to_temp = transfer_entropy(climate_data['Humidity'], climate_data['Temperature'])

print(f"Transfer entropy (Temperature to Humidity): {te_temp_to_humid:.4f}")
print(f"Transfer entropy (Humidity to Temperature): {te_humid_to_temp:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(['Temperature to Humidity', 'Humidity to Temperature'], [te_temp_to_humid, te_humid_to_temp])
plt.title('Transfer Entropy in Climate Data')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(axis='y')
plt.show()
```

Slide 11: Entropy truyền có điều kiện

Entropy truyền có điều kiện đo luồng thông tin từ biến này sang biến khác, cho trước biến thứ ba. Điều này có thể giúp xác định những ảnh hưởng gián tiếp trong các hệ thống phức tạp:

```python
def conditional_transfer_entropy(source, target, condition, k=1, l=1, m=1):
    joint = np.array(list(zip(target[k:], target[:len(target)-k], source[:len(source)-l], condition[:len(condition)-m])))
    p_joint = stats.gaussian_kde(joint.T)(joint.T)
    p_cond_target = stats.gaussian_kde(joint[:, :3].T)(joint[:, :3].T)
    p_cond_all = stats.gaussian_kde(joint.T)(joint.T)
    return np.mean(np.log2(p_cond_all * p_cond_target[:, :2] / (p_cond_target * p_joint[:, :3])))

# Generate synthetic data
np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = 0.5 * x + np.random.normal(0, 0.5, 1000)
z = 0.3 * x + 0.7 * y + np.random.normal(0, 0.3, 1000)

# Calculate conditional transfer entropy
cte_x_to_z_given_y = conditional_transfer_entropy(x, z, y)
cte_y_to_z_given_x = conditional_transfer_entropy(y, z, x)

print(f"Conditional TE (X to Z given Y): {cte_x_to_z_given_y:.4f}")
print(f"Conditional TE (Y to Z given X): {cte_y_to_z_given_x:.4f}")
```

Trang trình bày 12: Truyền Entropy trong phân tích chuỗi thời gian

Entropy truyền đặc biệt hữu ích trong phân tích chuỗi thời gian để phát hiện mối quan hệ nhân quả. Hãy áp dụng nó cho một mô hình tự hồi quy đơn giản:

```python
def generate_ar_process(n, a):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i-1] + np.random.normal(0, 1)
    return x

# Generate two AR(1) processes
n = 1000
x = generate_ar_process(n, 0.8)
y = generate_ar_process(n, 0.6)

# Introduce causal relationship: y affects x
x[1:] += 0.3 * y[:-1]

# Calculate transfer entropy
te_x_to_y = transfer_entropy(x, y)
te_y_to_x = transfer_entropy(y, x)

print(f"Transfer entropy (X to Y): {te_x_to_y:.4f}")
print(f"Transfer entropy (Y to X): {te_y_to_x:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(['X to Y', 'Y to X'], [te_x_to_y, te_y_to_x])
plt.title('Transfer Entropy in AR Processes')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(axis='y')
plt.show()
```

Trang trình bày 13: Hạn chế và cân nhắc

Mặc dù entropy truyền là một công cụ mạnh mẽ để phân tích luồng thông tin nhưng nó có một số hạn chế và cần cân nhắc:

1. Độ phức tạp tính toán: Việc tính toán entropy truyền có thể tốn kém về mặt tính toán, đặc biệt đối với các tập dữ liệu lớn.
2. Yêu cầu về dữ liệu: Ước tính chính xác đòi hỏi phải có đủ lượng dữ liệu.
3. Tính phi tuyến: Entropy truyền có thể nắm bắt các mối quan hệ phi tuyến, nhưng việc giải thích kết quả có thể là một thách thức.
4. Lựa chọn tham số: Việc chọn độ dài lịch sử thích hợp (k và l) có thể ảnh hưởng đến kết quả.

Để giải quyết một số vấn đề này, chúng ta có thể sử dụng các kỹ thuật như khởi động để ước tính khoảng tin cậy:

```python
import numpy as np
from scipy import stats

def bootstrap_transfer_entropy(source, target, n_bootstrap=1000, k=1, l=1):
    original_te = transfer_entropy(source, target, k, l)
    bootstrap_samples = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(source), len(source), replace=True)
        boot_source = source[indices]
        boot_target = target[indices]
        bootstrap_samples.append(transfer_entropy(boot_source, boot_target, k, l))

    ci_lower, ci_upper = np.percentile(bootstrap_samples, [2.5, 97.5])
    return original_te, ci_lower, ci_upper

# Example usage
np.random.seed(42)
source = np.random.normal(0, 1, 1000)
target = 0.5 * source + np.random.normal(0, 0.5, 1000)

te, ci_lower, ci_upper = bootstrap_transfer_entropy(source, target)
print(f"Transfer Entropy: {te:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

Trang trình bày 14: Kiểm tra dữ liệu thay thế

Để xác định xem entropy truyền được quan sát có ý nghĩa thống kê hay không, chúng ta có thể sử dụng thử nghiệm dữ liệu thay thế. Điều này liên quan đến việc tạo các phiên bản ngẫu nhiên của dữ liệu gốc và so sánh entropy truyền:

```python
def surrogate_data_test(source, target, n_surrogates=1000, k=1, l=1):
    original_te = transfer_entropy(source, target, k, l)
    surrogate_te = []

    for _ in range(n_surrogates):
        surrogate_source = np.random.permutation(source)
        surrogate_te.append(transfer_entropy(surrogate_source, target, k, l))

    p_value = np.mean(np.array(surrogate_te) >= original_te)
    return original_te, p_value

# Example usage
np.random.seed(42)
source = np.random.normal(0, 1, 1000)
target = 0.5 * source + np.random.normal(0, 0.5, 1000)

te, p_value = surrogate_data_test(source, target)
print(f"Transfer Entropy: {te:.4f}")
print(f"p-value: {p_value:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.hist(surrogate_te, bins=30, edgecolor='black')
plt.axvline(original_te, color='red', linestyle='dashed', linewidth=2)
plt.title('Surrogate Data Test for Transfer Entropy')
plt.xlabel('Transfer Entropy')
plt.ylabel('Frequency')
plt.legend(['Original TE', 'Surrogate TE'])
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về entropy truyền và các ứng dụng của nó, đây là một số tài nguyên có giá trị:

1. Schreiber, T. (2000). "Đo lường chuyển giao thông tin". Thư Đánh giá Vật lý, 85(2), 461-464. ArXiv: [https://arxiv.org/abs/nlin/0001042](https://arxiv.org/abs/nlin/0001042)
2. Lizier, J. T. (2014). &quot;JIDT: Bộ công cụ lý thuyết thông tin để nghiên cứu động lực học của các hệ thống phức tạp&quot;. Biên giới trong Robotics và AI, 1, 11. ArXiv: [https://arxiv.org/abs/1408.3270](https://arxiv.org/abs/1408.3270)
3. Bossomaier, T., Barnett, L., Harré, M., & Lizier, J. T. (2016). "Giới thiệu về Entropy truyền: Luồng thông tin trong các hệ thống phức tạp". Nhà xuất bản quốc tế Springer. (Sách)
4. Vicente, R., Wibral, M., Lindner, M., & Pipa, G. (2011). &quot;Truyền entropy, một thước đo kết nối hiệu quả không có mô hình cho khoa học thần kinh&quot;. Tạp chí Khoa học thần kinh tính toán, 30(1), 45-67. ArXiv: [https://arxiv.org/abs/0902.3616](https://arxiv.org/abs/0902.3616)

Những tài nguyên này cung cấp sự kết hợp giữa nền tảng lý thuyết và ứng dụng thực tế của entropy truyền trong các lĩnh vực khác nhau, từ vật lý đến khoa học thần kinh.
