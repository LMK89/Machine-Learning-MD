## Cơ bản về Xác suất và Thống kê
Slide 1: Giới thiệu về Xác suất

Xác suất là một nhánh của toán học liên quan đến khả năng xảy ra các sự kiện. Nó tạo thành nền tảng cho việc phân tích thống kê và ra quyết định trong điều kiện không chắc chắn.

```python
import random

# Simulating a coin flip
coin = ['Heads', 'Tails']
flips = 1000
results = [random.choice(coin) for _ in range(flips)]

heads_count = results.count('Heads')
probability_heads = heads_count / flips

print(f"Probability of getting Heads: {probability_height:.2f}")
```

Slide 2: Các khái niệm cơ bản về xác suất

Ba tiên đề xác suất xác định nền tảng toán học của nó: tính không âm, tính chuẩn hóa và tính cộng. Những nguyên tắc này đảm bảo rằng xác suất luôn nằm trong khoảng từ 0 đến 1 và tổng của tất cả các kết quả có thể xảy ra bằng 1.

```python
def check_probability_axioms(probabilities):
    non_negativity = all(p >= 0 for p in probabilities)
    normalization = sum(probabilities) == 1
    additivity = sum(probabilities) == sum(set(probabilities))

    return non_negativity and normalization and additivity

# Example probabilities
event_probabilities = [0.2, 0.3, 0.5]
print(f"Probabilities satisfy axioms: {check_probability_axioms(event_probabilities)}")
```

Trang trình bày 3: Mô tả xác suất chính

Hàm khối lượng xác suất (PMF) cho các biến rời rạc và hàm mật độ xác suất (PDF) cho các biến liên tục mô tả khả năng xảy ra các kết quả khác nhau. Hàm phân phối tích lũy (CDF) cho biết xác suất của một giá trị nhỏ hơn hoặc bằng một điểm nhất định.

```python
import numpy as np
import matplotlib.pyplot as plt

# PMF for a discrete uniform distribution
x = np.arange(1, 7)
pmf = np.ones_like(x) / len(x)

plt.bar(x, pmf)
plt.title("PMF of a Fair Die Roll")
plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.show()
```

Trang trình bày 4: Số liệu xu hướng trung tâm

Các thước đo về xu hướng trung tâm bao gồm giá trị trung bình, trung vị và mode. Chúng cung cấp các quan điểm khác nhau về giá trị điển hình hoặc giá trị trung tâm trong tập dữ liệu.

```python
import numpy as np

data = [2, 3, 3, 4, 5, 5, 5, 6, 7]

mean = np.mean(data)
median = np.median(data)
mode = max(set(data), key=data.count)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
```

Trang trình bày 5: Phân bố xác suất

Phân phối xác suất mô tả khả năng xảy ra các kết quả khác nhau đối với một biến ngẫu nhiên. Các phân phối phổ biến bao gồm phân phối chuẩn, phân phối nhị thức và phân phối Poisson.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 100)
y = norm.pdf(x, 0, 1)

plt.plot(x, y)
plt.title("Standard Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.show()
```

Slide 6: Các thước đo tương đồng và tương quan

Các hệ số tương quan đo lường cường độ và hướng của mối quan hệ giữa các biến. Các biện pháp phổ biến bao gồm mối tương quan của Pearson cho các mối quan hệ tuyến tính và tương quan xếp hạng của Spearman cho các mối quan hệ đơn điệu.

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

pearson_corr = np.corrcoef(x, y)[0, 1]
print(f"Pearson correlation: {pearson_corr:.2f}")
```

Slide 7: Giới thiệu về Thống kê

Thống kê bao gồm việc thu thập, phân tích, giải thích và trình bày dữ liệu. Nó cho phép chúng ta đưa ra suy luận về quần thể dựa trên dữ liệu mẫu.

```python
import numpy as np

population = np.random.normal(loc=100, scale=15, size=10000)
sample = np.random.choice(population, size=100, replace=False)

population_mean = np.mean(population)
sample_mean = np.mean(sample)

print(f"Population mean: {population_mean:.2f}")
print(f"Sample mean: {sample_mean:.2f}")
```

Slide 8: Kiểm định giả thuyết

Kiểm tra giả thuyết là một phương pháp thống kê được sử dụng để đưa ra suy luận về tham số tổng thể dựa trên dữ liệu mẫu. Nó liên quan đến việc xây dựng các giả thuyết không và giả thuyết thay thế, đồng thời sử dụng các kiểm tra thống kê để quyết định xem có bác bỏ giả thuyết không hay không.

```python
from scipy import stats

# Example: Testing if a coin is fair
flips = 100
heads = 60

# Perform binomial test
p_value = stats.binom_test(heads, n=flips, p=0.5, alternative='two-sided')

print(f"P-value: {p_value:.4f}")
print(f"{'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis")
```

Trang trình bày 9: Kiểm tra Z

Kiểm định Z được sử dụng khi đã biết độ lệch chuẩn của tổng thể và cỡ mẫu lớn. Nó so sánh giá trị trung bình mẫu với giá trị trung bình tổng thể đã biết bằng cách sử dụng phân phối chuẩn chuẩn.

```python
from scipy import stats
import numpy as np

population_mean = 100
population_std = 15
sample_size = 30

sample = np.random.normal(loc=105, scale=population_std, size=sample_size)
sample_mean = np.mean(sample)

z_statistic = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

print(f"Z-statistic: {z_statistic:.2f}")
print(f"P-value: {p_value:.4f}")
```

Trang trình bày 10: t-test

Kiểm định t được sử dụng khi chưa biết độ lệch chuẩn của tổng thể và cỡ mẫu nhỏ. Nó so sánh giá trị trung bình giữa hai nhóm hoặc giá trị trung bình mẫu với giá trị đã biết.

```python
from scipy import stats
import numpy as np

group1 = np.random.normal(loc=100, scale=15, size=20)
group2 = np.random.normal(loc=110, scale=15, size=20)

t_statistic, p_value = stats.ttest_ind(group1, group2)

print(f"T-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.4f}")
```

Slide 11: Kiểm tra Chi-Square

Kiểm tra Chi-Square được sử dụng để xác định xem liệu có mối liên hệ đáng kể giữa các biến phân loại hay để kiểm tra mức độ phù hợp của dữ liệu được quan sát với phân phối dự kiến.

```python
from scipy.stats import chi2_contingency

observed = np.array([[10, 20, 30],
                     [15, 25, 20]])

chi2, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi-square statistic: {chi2:.2f}")
print(f"P-value: {p_value:.4f}")
```

Slide 12: Phân tích phương sai (ANOVA)

ANOVA được sử dụng để so sánh các phương tiện giữa ba nhóm trở lên. Nó giúp xác định xem có sự khác biệt có ý nghĩa thống kê giữa các phương tiện nhóm hay không.

```python
import numpy as np
from scipy import stats

group1 = np.random.normal(loc=10, scale=2, size=30)
group2 = np.random.normal(loc=12, scale=2, size=30)
group3 = np.random.normal(loc=11, scale=2, size=30)

f_statistic, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_statistic:.2f}")
print(f"P-value: {p_value:.4f}")
```

Trang trình bày 13: Nhiều so sánh

Khi tiến hành nhiều thử nghiệm thống kê, khả năng xảy ra lỗi Loại I (dương tính giả) sẽ tăng lên. Nhiều quy trình so sánh, chẳng hạn như hiệu chỉnh Bonferroni hoặc Tỷ lệ phát hiện sai, điều chỉnh giá trị p để kiểm soát tỷ lệ lỗi này.

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

# Simulating p-values from multiple tests
p_values = np.random.uniform(0, 1, 10)

# Bonferroni correction
bonferroni_corrected = multipletests(p_values, method='bonferroni')

print("Original p-values:", p_values)
print("Bonferroni corrected p-values:", bonferroni_corrected[1])
```

Slide 14: Phân tích nhân tố

Phân tích nhân tố là một phương pháp thống kê được sử dụng để mô tả sự biến thiên giữa các biến được quan sát, có mối tương quan với số lượng biến không được quan sát được gọi là các yếu tố có thể thấp hơn.

```python
from factor_analyzer import FactorAnalyzer
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(0)
data = pd.DataFrame(np.random.rand(100, 5), columns=['V1', 'V2', 'V3', 'V4', 'V5'])

# Perform factor analysis
fa = FactorAnalyzer(rotation=None, n_factors=2)
fa.fit(data)

# Get factor loadings
loadings = pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2'], index=data.columns)
print(loadings)
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về xác suất và thống kê, hãy xem xét các tài nguyên sau:

1. "Giới thiệu về xác suất" của Blitzstein và Hwang (arXiv:1302.1281)
2. “Suy luận thống kê” của Casella và Berger
3. "Các yếu tố của việc học thống kê" của Hastie, Tibshirani và Friedman (arXiv:1011.0933)
4. Các khóa học trực tuyến trên các nền tảng như Coursera, edX hoặc MIT OpenCourseWare
5. Tài liệu phần mềm thống kê (ví dụ: thư viện SciPy và statsmodels của Python)
