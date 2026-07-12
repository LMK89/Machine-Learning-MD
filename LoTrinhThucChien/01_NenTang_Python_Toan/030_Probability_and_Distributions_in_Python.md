## Xác suất và phân phối trong Python

Slide 2: Giới thiệu về Xác suất Xác suất là nghiên cứu toán học về khả năng xảy ra các sự kiện. Trong Python, chúng ta có thể sử dụng nhiều thư viện và hàm khác nhau để làm việc với các khái niệm xác suất. Thư viện được sử dụng phổ biến nhất cho mục đích này là NumPy.

Trang trình bày 3: Số ngẫu nhiên Trước khi đi sâu vào xác suất và phân phối, chúng ta cần hiểu cách tạo số ngẫu nhiên trong Python. Mô-đun ngẫu nhiên cung cấp các chức năng tạo số ngẫu nhiên. Ví dụ mã:

```python
import random

# Generate a random float between 0 and 1
random_float = random.random()
print(random_float)

# Generate a random integer between 1 and 6 (inclusive)
random_int = random.randint(1, 6)
print(random_int)
```

Trang trình bày 4: Phân bố xác suất rời rạc Phân bố xác suất rời rạc là phân bố xác suất mô tả khả năng xảy ra các kết quả khác nhau có thể xảy ra đối với một biến ngẫu nhiên có thể nhận một số giá trị đếm được. Trong Python, chúng ta có thể sử dụng các mô-đun toán học và thống kê để làm việc với các phân phối rời rạc. Ví dụ mã:

```python
import math

# Calculate the probability mass function (PMF) for a binomial distribution
n = 10  # Number of trials
p = 0.3  # Probability of success
k = 3  # Number of successes
pmf = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
print(f"Binomial PMF: {pmf}")
```

Trang trình bày 5: Phân bố xác suất liên tục Phân bố xác suất liên tục là phân bố xác suất mô tả khả năng xảy ra các kết quả khác nhau có thể xảy ra đối với một biến ngẫu nhiên có thể nhận bất kỳ giá trị nào trong phạm vi liên tục. Trong Python, chúng ta có thể sử dụng mô-đun scipy.stats để làm việc với các bản phân phối liên tục. Ví dụ mã:

```python
import scipy.stats as stats

# Calculate the probability density function (PDF) for a normal distribution
mu = 0  # Mean
sigma = 1  # Standard deviation
x = 1.5  # Value to evaluate
pdf = stats.norm.pdf(x, mu, sigma)
print(f"Normal PDF at x={x}: {pdf}")
```

Trang trình bày 6: Định lý giới hạn trung tâm Định lý giới hạn trung tâm phát biểu rằng tổng của nhiều biến ngẫu nhiên độc lập và có phân bố giống nhau có xu hướng hướng tới phân bố chuẩn, bất kể phân bố cơ bản là gì. Định lý này là cơ bản trong xác suất và thống kê. Ví dụ mã:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample of 10000 random numbers from a uniform distribution
sample = np.random.uniform(size=10000)

# Calculate the mean and standard deviation of the sample
sample_mean = np.mean(sample)
sample_std = np.std(sample)

# Plot the histogram of the sample
plt.hist(sample, bins=30, density=True)
plt.title("Histogram of Uniform Sample")
plt.show()
```

Trang trình bày 7: Lấy mẫu và khởi động Lấy mẫu và khởi động là các kỹ thuật được sử dụng để ước tính các tham số tổng thể hoặc kiểm tra các giả thuyết dựa trên một mẫu dữ liệu. Trong Python, chúng ta có thể sử dụng mô-đun ngẫu nhiên và NumPy để thực hiện lấy mẫu và khởi động. Ví dụ mã:

```python
import numpy as np

# Generate a population of 1000 random numbers
population = np.random.normal(loc=0, scale=1, size=1000)

# Take a simple random sample of size 100 from the population
sample = np.random.choice(population, size=100, replace=False)

# Perform bootstrapping to estimate the population mean
bootstrap_means = []
for _ in range(1000):
    bootstrap_sample = np.random.choice(sample, size=len(sample), replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))

print(f"Bootstrap estimate of population mean: {np.mean(bootstrap_means)}")
```

Slide 8: Kiểm định giả thuyết Kiểm định giả thuyết là một phương pháp thống kê được sử dụng để đưa ra các suy luận về một tham số tổng thể dựa trên một mẫu dữ liệu. Trong Python, chúng ta có thể sử dụng mô-đun scipy.stats để thực hiện kiểm tra giả thuyết. Ví dụ mã:

```python
import scipy.stats as stats

# Generate two samples
sample1 = np.random.normal(loc=0, scale=1, size=100)
sample2 = np.random.normal(loc=0.5, scale=1, size=100)

# Perform a two-sample t-test
t_stat, p_val = stats.ttest_ind(sample1, sample2)

# Print the results
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_val}")
```

Trang trình bày 9: Khoảng tin cậy Khoảng tin cậy là một phạm vi giá trị có khả năng chứa một tham số tổng thể chưa biết với mức độ tin cậy nhất định. Trong Python, chúng ta có thể sử dụng mô-đun scipy.stats để tính khoảng tin cậy. Ví dụ mã:

```python
import scipy.stats as stats

# Generate a sample
sample = np.random.normal(loc=0, scale=1, size=100)

# Calculate the 95% confidence interval for the population mean
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
n = len(sample)
confidence_interval = stats.norm.interval(0.95, loc=sample_mean, scale=sample_std / np.sqrt(n))

print(f"95% Confidence Interval: {confidence_interval}")
```

Trang trình bày 10: Mô phỏng Monte Carlo Mô phỏng Monte Carlo là một kỹ thuật được sử dụng để ước tính xác suất của các kết quả khác nhau bằng cách chạy thử nhiều lần, thường sử dụng lấy mẫu ngẫu nhiên. Trong Python, chúng ta có thể sử dụng NumPy và các thư viện khác để thực hiện mô phỏng Monte Carlo. Ví dụ mã:

```python
import numpy as np

# Define the function to be simulated
def func(x):
    return x ** 2 + np.random.normal(0, 1)

# Set up the simulation
num_simulations = 10000
x_values = np.linspace(-5, 5, 100)
results = np.zeros((len(x_values), num_simulations))

# Run the simulation
for i in range(num_simulations):
    results[:, i] = [func(x) for x in x_values]

# Calculate the mean and confidence intervals
means = np.mean(results, axis=1)
lower_bounds = np.percentile(results, 2.5, axis=1)
upper_bounds = np.percentile(results, 97.5, axis=1)

# Plot the results
plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.3)
plt.plot(x_values, means, label="Mean")
plt.legend()
plt.show()
```

Trang trình bày 11: Thống kê Bayes Thống kê Bayes là một nhánh của thống kê sử dụng định lý Bayes để cập nhật xác suất của các giả thuyết khi có thêm bằng chứng hoặc thông tin. Trong Python, chúng ta có thể sử dụng các thư viện như PyMC3 để thực hiện phân tích Bayes. Ví dụ mã:

```python
import pymc3 as pm

# Define the data
data = np.random.normal(loc=0, scale=1, size=100)

# Define the Bayesian model
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

    # Run the MCMC sampler
    trace = pm.sample(1000, cores=2)

# Print the summary statistics
print(pm.summary(trace))
```

Trang trình bày 12: Kết luận Trong bài trình bày này, chúng tôi đã đề cập đến nhiều khái niệm và kỹ thuật khác nhau liên quan đến xác suất và phân phối trong Python. Chúng tôi đã khám phá việc tạo số ngẫu nhiên, phân bố xác suất rời rạc và liên tục, Định lý giới hạn trung tâm, lấy mẫu và khởi động, kiểm tra giả thuyết, khoảng tin cậy, mô phỏng Monte Carlo và thống kê Bayes. Python cung cấp các thư viện và công cụ mạnh mẽ để làm việc với xác suất và thống kê, khiến nó trở thành lựa chọn tuyệt vời cho các nhiệm vụ phân tích và lập mô hình dữ liệu.

## Meta:
Nắm vững xác suất và phân phối trong Python

Khai phá sức mạnh của xác suất và phân phối trong Python với loạt TikTok toàn diện của chúng tôi. Từ việc tạo số ngẫu nhiên đến thống kê Bayesian, chúng tôi sẽ hướng dẫn bạn các khái niệm và kỹ thuật chính, hoàn chỉnh với các ví dụ về mã và giải thích rõ ràng. Nâng cao kỹ năng lập mô hình và phân tích dữ liệu của bạn với tài nguyên thiết yếu này dành cho những người đam mê Python và các nhà khoa học dữ liệu đầy tham vọng. Hãy tham gia cùng chúng tôi trong hành trình giáo dục này và nâng cao trình độ Python của bạn lên một tầm cao mới.

Thẻ bắt đầu bằng #: #PythonTutorials #ProbabilityAndDistributions #DataScience #CodeExamples #LearningTikTok #InstitutionalContent
