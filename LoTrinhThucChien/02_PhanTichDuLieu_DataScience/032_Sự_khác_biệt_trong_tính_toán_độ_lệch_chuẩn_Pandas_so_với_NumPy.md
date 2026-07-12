## Sự khác biệt trong tính toán độ lệch chuẩn Pandas so với NumPy
Trang trình bày 1: Tìm hiểu sự khác biệt về độ lệch chuẩn

Tính toán thống kê trong Python có thể mang lại kết quả khác nhau tùy thuộc vào thư viện được sử dụng. Sự khác biệt chính giữa NumPy và Pandas nằm ở hành vi mặc định của chúng liên quan đến tham số bậc tự do (ddof) khi tính toán độ lệch chuẩn.

```python
import numpy as np
import pandas as pd

# Create sample data
data = [2, 4, 4, 4, 5, 5, 7, 9]

# NumPy std calculation (default ddof=0)
np_std = np.std(data)

# Pandas std calculation (default ddof=1)
pd_std = pd.Series(data).std()

print(f"NumPy std (ddof=0): {np_std:.6f}")
print(f"Pandas std (ddof=1): {pd_std:.6f}")
```

Slide 2: Cơ sở toán học

Sự khác biệt cơ bản bắt nguồn từ công thức độ lệch chuẩn của dân số so với mẫu. Các biểu thức toán học cho thấy mức độ tự do tác động như thế nào đến phép tính cuối cùng.

```python
# Mathematical formulas in LaTeX notation (not rendered)
$$\text{Population SD} = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}$$

$$\text{Sample SD} = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \bar{x})^2}{N-1}}$$

# Implementation from scratch
def calculate_std(data, ddof=0):
    mean = sum(data) / len(data)
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    return (squared_diff_sum / (len(data) - ddof)) ** 0.5
```

Slide 3: Độ lệch chuẩn của dân số

Độ lệch chuẩn của dân số giả định chúng ta có dữ liệu đầy đủ về toàn bộ dân số. Việc triển khai mặc định của NumPy (ddof=0) sử dụng phương pháp này, chia cho N ở mẫu số.

```python
import numpy as np

data = [2, 4, 4, 4, 5, 5, 7, 9]

# NumPy population std
pop_std = np.std(data, ddof=0)

# Custom implementation
def population_std(data):
    mean = np.mean(data)
    squared_diff = [(x - mean) ** 2 for x in data]
    return np.sqrt(sum(squared_diff) / len(data))

print(f"NumPy population std: {pop_std:.6f}")
print(f"Custom population std: {population_std(data):.6f}")
```

Slide 4: Độ lệch chuẩn mẫu

Khi làm việc với dữ liệu mẫu, các nhà thống kê thường thích sử dụng bậc tự do N-1 (hiệu chỉnh Bessel). Pandas áp dụng quy ước này theo mặc định, điều này giải thích các kết quả khác nhau.

```python
import pandas as pd

data = [2, 4, 4, 4, 5, 5, 7, 9]

# Pandas sample std
sample_std = pd.Series(data).std()

# Custom implementation
def sample_std(data):
    mean = sum(data) / len(data)
    squared_diff = [(x - mean) ** 2 for x in data]
    return np.sqrt(sum(squared_diff) / (len(data) - 1))

print(f"Pandas sample std: {sample_std:.6f}")
print(f"Custom sample std: {sample_std(data):.6f}")
```

Trang trình bày 5: Ví dụ thực tế: Phân tích giá cổ phiếu

Các nhà phân tích tài chính thường sử dụng độ lệch chuẩn để đo lường sự biến động của thị trường. Ví dụ này cho thấy các phép tính tiêu chuẩn khác nhau ảnh hưởng như thế nào đến việc đánh giá rủi ro.

```python
import numpy as np
import pandas as pd

# Sample daily stock returns
stock_returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.03, 0.02, 0.01, -0.01]

# Calculate volatility using both methods
np_volatility = np.std(stock_returns) * np.sqrt(252)  # Annualized
pd_volatility = pd.Series(stock_returns).std() * np.sqrt(252)

print(f"NumPy Annualized Volatility: {np_volatility:.4f}")
print(f"Pandas Annualized Volatility: {pd_volatility:.4f}")
```

Trang trình bày 6: Tác động đến phân tích nghiên cứu

Sự lựa chọn giữa độ lệch chuẩn tổng thể và mẫu tác động đáng kể đến kết luận nghiên cứu, đặc biệt là trong các tập dữ liệu nhỏ. Hiểu những khác biệt này là rất quan trọng để suy luận thống kê chính xác và thiết kế thử nghiệm.

```python
import numpy as np
import pandas as pd

# Compare impact on different sample sizes
sample_sizes = [5, 10, 30, 100]

for size in sample_sizes:
    data = np.random.normal(0, 1, size)
    np_std = np.std(data)
    pd_std = pd.Series(data).std()
    diff_percent = ((pd_std - np_std) / np_std) * 100

    print(f"Sample size: {size}")
    print(f"NumPy std: {np_std:.6f}")
    print(f"Pandas std: {pd_std:.6f}")
    print(f"Difference: {diff_percent:.2f}%\n")
```

Trang trình bày 7: Ví dụ về phân tích dữ liệu chăm sóc sức khỏe

Ứng dụng trong thế giới thực chứng minh tác động của các phép tính độ lệch chuẩn khác nhau đối với việc theo dõi dấu hiệu sinh tồn của bệnh nhân và quá trình ra quyết định lâm sàng.

```python
import numpy as np
import pandas as pd

# Simulated patient blood pressure readings
bp_readings = [120, 122, 118, 125, 119, 121, 123, 120]

def analyze_vitals(data):
    np_std = np.std(data, ddof=0)
    pd_std = pd.Series(data).std()

    # Calculate reference ranges
    np_range = (np.mean(data) - 2*np_std, np.mean(data) + 2*np_std)
    pd_range = (np.mean(data) - 2*pd_std, np.mean(data) + 2*pd_std)

    return {
        'pop_std': np_std,
        'sample_std': pd_std,
        'pop_range': np_range,
        'sample_range': pd_range
    }

results = analyze_vitals(bp_readings)
for key, value in results.items():
    print(f"{key}: {value}")
```

Slide 8: Ảnh hưởng của cỡ mẫu đến độ lệch chuẩn

Phân tích toàn diện về mức độ ảnh hưởng của cỡ mẫu đến sự khác biệt giữa phép tính độ lệch chuẩn của tổng thể và mẫu, với mã trực quan.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_size_effect(min_size=5, max_size=100, steps=20):
    sizes = np.linspace(min_size, max_size, steps, dtype=int)
    differences = []

    for size in sizes:
        data = np.random.normal(0, 1, size)
        np_std = np.std(data, ddof=0)
        pd_std = pd.Series(data).std()
        diff = ((pd_std - np_std) / np_std) * 100
        differences.append(diff)

    return sizes, differences

sizes, diffs = analyze_size_effect()
print("Sample Size | Difference (%)")
print("-" * 25)
for size, diff in zip(sizes, diffs):
    print(f"{size:10d} | {diff:12.2f}")
```

Slide 9: Xử lý các giá trị bị thiếu

Các phép tính độ lệch chuẩn khác nhau xử lý các giá trị còn thiếu theo cách khác nhau, điều này có thể tác động đáng kể đến kết quả phân tích trong bộ dữ liệu trong thế giới thực.

```python
import numpy as np
import pandas as pd

# Dataset with missing values
data_with_nan = [1, 2, np.nan, 4, 5, 6, np.nan, 8]

# NumPy approach
np_clean = np.array(data_with_nan)[~np.isnan(data_with_nan)]
np_std = np.std(np_clean)

# Pandas approach
pd_std = pd.Series(data_with_nan).std()

# Custom implementation with missing value handling
def robust_std(data, ddof=1):
    clean_data = [x for x in data if not pd.isna(x)]
    mean = sum(clean_data) / len(clean_data)
    squared_diff = [(x - mean) ** 2 for x in clean_data]
    return np.sqrt(sum(squared_diff) / (len(clean_data) - ddof))

print(f"NumPy std: {np_std:.6f}")
print(f"Pandas std: {pd_std:.6f}")
print(f"Custom robust std: {robust_std(data_with_nan):.6f}")
```

Trang trình bày 10: Những cân nhắc về tính toán song song

Việc tính toán độ lệch chuẩn trong môi trường điện toán phân tán đòi hỏi sự chú ý đặc biệt để duy trì độ ổn định và độ chính xác về số trên các phương pháp tính toán khác nhau.

```python
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import math

def parallel_std(data, chunks=4, ddof=1):
    chunk_size = math.ceil(len(data) / chunks)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def chunk_stats(chunk):
        return len(chunk), np.sum(chunk), np.sum(np.square(chunk))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(chunk_stats, chunks))

    total_n = sum(r[0] for r in results)
    total_sum = sum(r[1] for r in results)
    total_sq_sum = sum(r[2] for r in results)

    mean = total_sum / total_n
    variance = (total_sq_sum - (total_sum ** 2) / total_n) / (total_n - ddof)
    return np.sqrt(variance)

# Example usage
data = np.random.normal(0, 1, 1000000)
print(f"Parallel std: {parallel_std(data.tolist()):.6f}")
print(f"Pandas std: {pd.Series(data).std():.6f}")
```

Slide 11: Độ lệch chuẩn chuỗi thời gian

Dữ liệu chuỗi thời gian cần được xem xét đặc biệt khi tính toán độ lệch chuẩn, vì sự phụ thuộc theo thời gian có thể ảnh hưởng đến việc giải thích các phép đo độ biến thiên.

```python
import numpy as np
import pandas as pd

# Create time series data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.random.normal(100, 15, 100)
ts_data = pd.Series(values, index=dates)

# Calculate rolling standard deviation
def analyze_time_series_std(data, windows=[7, 14, 30]):
    results = {}
    for window in windows:
        # NumPy approach (manual rolling)
        np_rolling = [np.std(values[max(0, i-window):i])
                     for i in range(1, len(values)+1)]

        # Pandas approach
        pd_rolling = data.rolling(window=window).std()

        results[f'window_{window}'] = {
            'numpy': np_rolling[-1],
            'pandas': pd_rolling.iloc[-1]
        }

    return results

results = analyze_time_series_std(ts_data)
for window, stats in results.items():
    print(f"\n{window}:")
    print(f"NumPy rolling std: {stats['numpy']:.6f}")
    print(f"Pandas rolling std: {stats['pandas']:.6f}")
```

Slide 12: Độ lệch chuẩn có trọng số

Khi các quan sát có mức độ quan trọng khác nhau, độ lệch chuẩn có trọng số cung cấp thước đo độ phân tán chính xác hơn khi xem xét tầm quan trọng tương đối của từng điểm dữ liệu.

```python
import numpy as np
import pandas as pd

def weighted_std(values, weights, ddof=1):
    """
    Calculate weighted standard deviation with specified degrees of freedom
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance * len(weights) / (len(weights) - ddof))

# Example with student grades and credit weights
grades = [85, 92, 78, 90, 88]
credits = [3, 4, 2, 4, 3]

# Calculate using different methods
numpy_weighted = np.sqrt(np.cov(grades, aweights=credits))
custom_weighted = weighted_std(grades, credits)
simple_std = pd.Series(grades).std()

print(f"Weighted std (custom): {custom_weighted:.6f}")
print(f"Weighted std (numpy): {numpy_weighted[0][0]:.6f}")
print(f"Unweighted std: {simple_std:.6f}")
```

Slide 13: Độ lệch chuẩn chắc chắn

Dữ liệu trong thế giới thực thường chứa các giá trị ngoại lệ có thể tác động đáng kể đến việc tính toán độ lệch chuẩn. Các phương pháp mạnh mẽ cung cấp các thước đo đáng tin cậy hơn về tính biến thiên trong những trường hợp như vậy.

```python
import numpy as np
from scipy import stats

def robust_statistics(data):
    # Regular std
    standard_std = np.std(data, ddof=1)

    # Median Absolute Deviation (MAD)
    mad = stats.median_abs_deviation(data)

    # Interquartile Range based std
    q75, q25 = np.percentile(data, [75, 25])
    iqr_std = (q75 - q25) / 1.349

    # Trimmed std (removing 10% from each end)
    trimmed_std = stats.trim_mean(np.square(data - np.mean(data)), 0.1) ** 0.5

    return {
        'standard': standard_std,
        'mad': mad,
        'iqr_based': iqr_std,
        'trimmed': trimmed_std
    }

# Example with outliers
data_with_outliers = [1, 2, 2, 3, 3, 4, 4, 100]
results = robust_statistics(data_with_outliers)

for method, value in results.items():
    print(f"{method} std: {value:.6f}")
```

Trang trình bày 14: Tài nguyên bổ sung

* [https://arxiv.org/abs/1906.07101](https://arxiv.org/abs/1906.07101) - "Một cái nhìn mới về độ lệch chuẩn: Khái quát hóa các quan sát có trọng số"
* [https://arxiv.org/abs/1811.02891](https://arxiv.org/abs/1811.02891) - "Thống kê mạnh mẽ để phát hiện ngoại lệ trong dữ liệu lớn"
* [https://arxiv.org/abs/2003.06663](https://arxiv.org/abs/2003.06663) - "Về việc lựa chọn số bậc tự do trong ước tính thống kê"
* [https://arxiv.org/abs/1712.04788](https://arxiv.org/abs/1712.04788) - "Phân tích thống kê dữ liệu chuỗi thời gian: Hướng dẫn toàn diện"
* [https://arxiv.org/abs/1902.06021](https://arxiv.org/abs/1902.06021) - "Tính toán hiệu quả độ lệch chuẩn trong hệ thống phân tán"
