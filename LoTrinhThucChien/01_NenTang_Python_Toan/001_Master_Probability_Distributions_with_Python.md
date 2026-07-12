## Phân phối xác suất chính với Python
Trang trình bày 1: Tìm hiểu về phân phối xác suất

Phân phối xác suất là các hàm toán học mô tả khả năng xảy ra các kết quả khác nhau trong một sự kiện ngẫu nhiên. Chúng là nền tảng của thống kê và khoa học dữ liệu, giúp chúng ta lập mô hình về sự không chắc chắn và đưa ra dự đoán. Trong phần trình bày này, chúng ta sẽ khám phá các phân bố xác suất chính và cách làm việc với chúng bằng Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate random data
data = np.random.randn(1000)

# Plot histogram
plt.hist(data, bins=30, density=True)
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Trang trình bày 2: Phân phối bình thường

Phân phối chuẩn, còn được gọi là phân phối Gaussian, là một đường cong hình chuông đối xứng. Nó được sử dụng rộng rãi trong khoa học tự nhiên và xã hội để biểu diễn các biến ngẫu nhiên có giá trị thực. Trong Python, chúng ta có thể tạo và trực quan hóa phân phối chuẩn bằng NumPy và Matplotlib.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data points
x = np.linspace(-5, 5, 100)
y = stats.norm.pdf(x, 0, 1)

# Plot the distribution
plt.plot(x, y)
plt.title('Standard Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
```

Trang trình bày 3: Phân phối đồng đều

Sự phân bố đồng đều thể hiện xác suất không đổi trên một phạm vi xác định. Nó thường được sử dụng trong mô phỏng và tạo số ngẫu nhiên. Dưới đây là cách tạo và trực quan hóa phân phối đồng đều trong Python:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate uniform random numbers
data = np.random.uniform(0, 1, 1000)

# Plot histogram
plt.hist(data, bins=30, density=True)
plt.title('Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Trang trình bày 4: Phân phối nhị thức

Phân phối nhị thức mô hình hóa số lần thành công trong một số thử nghiệm Bernoulli độc lập cố định. Nó thường được sử dụng trong các tình huống liên quan đến kết quả có/không, chẳng hạn như tung đồng xu hoặc kiểm soát chất lượng. Hãy mô phỏng việc lật đồng xu bằng cách sử dụng phân phối nhị thức:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 10  # number of trials
p = 0.5  # probability of success

# Generate binomial distribution
x = np.arange(0, n+1)
y = binom.pmf(x, n, p)

# Plot
plt.bar(x, y)
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.show()
```

Trang trình bày 5: Phân phối Poisson

Phân phối Poisson mô hình hóa số lượng sự kiện xảy ra trong một khoảng thời gian hoặc không gian cố định. Nó thường được sử dụng trong lý thuyết xếp hàng, luồng giao thông và mô hình sự kiện hiếm gặp. Dưới đây là ví dụ về việc tạo phân phối Poisson:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parameter
lambda_param = 3  # average number of events

# Generate Poisson distribution
x = np.arange(0, 15)
y = poisson.pmf(x, lambda_param)

# Plot
plt.bar(x, y)
plt.title(f'Poisson Distribution (λ={lambda_param})')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.show()
```

Trang trình bày 6: Phân phối theo cấp số nhân

Phân bố hàm mũ mô hình hóa thời gian giữa các sự kiện trong quy trình Poisson. Nó thường được sử dụng trong kỹ thuật độ tin cậy và lý thuyết xếp hàng. Hãy tạo một phân bố hàm mũ và vẽ hàm mật độ xác suất của nó:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Parameter
lambda_param = 0.5  # rate parameter

# Generate data points
x = np.linspace(0, 10, 100)
y = expon.pdf(x, scale=1/lambda_param)

# Plot
plt.plot(x, y)
plt.title(f'Exponential Distribution (λ={lambda_param})')
plt.xlabel('Time')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
```

Trang trình bày 7: Ví dụ thực tế: Khách hàng đến

Hãy lập mô hình khách hàng đến quán cà phê bằng cách sử dụng phân phối Poisson. Giả sử trung bình mỗi giờ có 20 khách hàng đến. Chúng tôi sẽ mô phỏng số lượng khách đến trong một ngày 12 giờ:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_param = 20  # average arrivals per hour
hours = 12

# Simulate customer arrivals
arrivals = np.random.poisson(lambda_param, hours)

# Plot
plt.bar(range(1, hours+1), arrivals)
plt.title('Customer Arrivals at Coffee Shop')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Arrivals')
plt.show()

print(f"Total customers: {sum(arrivals)}")
```

Slide 8: Ví dụ thực tế: Kiểm soát chất lượng sản xuất

Trong quy trình sản xuất, chúng ta có thể sử dụng phân phối nhị thức để lập mô hình số lượng mặt hàng bị lỗi trong một lô. Hãy mô phỏng việc kiểm soát chất lượng cho một dây chuyền sản xuất trong đó mỗi mặt hàng có 5% khả năng bị lỗi:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 100  # items per batch
p = 0.05  # probability of defect
num_batches = 1000

# Simulate batches
defects = np.random.binomial(n, p, num_batches)

# Plot histogram
plt.hist(defects, bins=range(0, max(defects)+2), align='left', rwidth=0.8)
plt.title('Defective Items per Batch')
plt.xlabel('Number of Defective Items')
plt.ylabel('Frequency')
plt.show()

print(f"Average defects per batch: {np.mean(defects):.2f}")
```

Slide 9: Phù hợp phân bố xác suất

Thông thường, chúng ta cần xác định phân bố xác suất nào phù hợp nhất với dữ liệu của mình. SciPy cung cấp các công cụ để phân phối phù hợp. Hãy tạo một số dữ liệu ngẫu nhiên và cố gắng phân phối phù hợp với nó:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data (let's assume it's from a gamma distribution)
true_shape, true_scale = 2.0, 2.0
data = np.random.gamma(true_shape, true_scale, 1000)

# Fit a gamma distribution to the data
fitted_params = stats.gamma.fit(data)
fitted_shape, _, fitted_scale = fitted_params

# Plot the results
x = np.linspace(0, 20, 100)
plt.hist(data, bins=50, density=True, alpha=0.7, label='Data')
plt.plot(x, stats.gamma.pdf(x, *fitted_params), 'r-', label='Fitted')
plt.title('Gamma Distribution Fitting')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"True shape: {true_shape}, Fitted shape: {fitted_shape:.2f}")
print(f"True scale: {true_scale}, Fitted scale: {fitted_scale:.2f}")
```

Trang trình bày 10: Phân phối chuẩn đa biến

Phân phối chuẩn đa biến là sự mở rộng của phân phối chuẩn một chiều đến các chiều cao hơn. Nó hữu ích cho việc mô hình hóa các biến ngẫu nhiên tương quan. Hãy tạo và trực quan hóa phân phối chuẩn đa biến 2D:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Parameters
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]

# Create grid and multivariate normal
x, y = np.mgrid[-3:3:.1, -3:3:.1]
pos = np.dstack((x, y))
rv = multivariate_normal(mean, cov)

# Plot
plt.contourf(x, y, rv.pdf(pos))
plt.title('2D Multivariate Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()
```

Trang trình bày 11: Ước tính mật độ hạt nhân

Ước tính mật độ hạt nhân (KDE) là một cách phi tham số để ước tính hàm mật độ xác suất của một biến ngẫu nhiên. Điều này hữu ích khi bạn không biết phân phối cơ bản của dữ liệu của mình. Hãy sử dụng KDE để ước tính mức phân bổ của một số dữ liệu mẫu:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
data = np.concatenate([np.random.normal(0, 1, 1000),
                       np.random.normal(4, 1.5, 500)])

# Compute KDE
kde = stats.gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 100)

# Plot
plt.hist(data, bins=50, density=True, alpha=0.7, label='Data')
plt.plot(x_range, kde(x_range), label='KDE')
plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
```

Slide 12: Hàm phân phối tích lũy (CDF)

Hàm phân phối tích lũy (CDF) cho biết xác suất một biến ngẫu nhiên nhỏ hơn hoặc bằng một giá trị nhất định. Nó rất hữu ích cho việc tính toán xác suất và lượng tử. Hãy vẽ CDF của phân phối chuẩn:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data points
x = np.linspace(-4, 4, 100)
y = stats.norm.cdf(x)

# Plot CDF
plt.plot(x, y)
plt.title('Cumulative Distribution Function (CDF) of Standard Normal')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()

# Calculate probability P(X <= 1)
print(f"P(X <= 1) = {stats.norm.cdf(1):.4f}")
```

Trang trình bày 13: Mô phỏng Monte Carlo

Mô phỏng Monte Carlo sử dụng việc lấy mẫu ngẫu nhiên lặp đi lặp lại để giải quyết các vấn đề về nguyên tắc có thể mang tính quyết định. Chúng được sử dụng rộng rãi trong tài chính, vật lý và kỹ thuật. Hãy sử dụng mô phỏng Monte Carlo để ước tính π:

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n):
    points_inside_circle = 0
    total_points = n

    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)

    distances = np.sqrt(x**2 + y**2)
    points_inside_circle = np.sum(distances <= 1)

    pi_estimate = 4 * points_inside_circle / total_points
    return pi_estimate, x, y

n = 10000
pi_estimate, x, y = estimate_pi(n)

plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=np.sqrt(x**2 + y**2) <= 1, cmap='coolwarm', alpha=0.5)
plt.title(f'Monte Carlo Pi Estimation\nEstimate: {pi_estimate:.4f}, True: {np.pi:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Để khám phá thêm về phân bố xác suất và ứng dụng của chúng trong Python:

1. Tài liệu SciPy: Hướng dẫn toàn diện về các hàm thống kê và phân bố xác suất. [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
2. "Lý thuyết xác suất: Logic của khoa học" của E. T. Jaynes: Văn bản nền tảng về lý thuyết xác suất. ArXiv: [https://arxiv.org/abs/math/0312635](https://arxiv.org/abs/math/0312635)
3. "Giới thiệu về học thống kê" của James, Witten, Hastie và Tibshirani: Bao gồm các phương pháp học thống kê với các ứng dụng trong R. (Lưu ý: Mặc dù không có trên ArXiv nhưng đây là tài nguyên được công nhận rộng rãi trong lĩnh vực này)
4. "Lập trình xác suất & Phương pháp Bayesian dành cho tin tặc" của Cameron Davidson-Pilon: Giới thiệu thực tế về phương pháp Bayesian và lập trình xác suất. GitHub: [https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
