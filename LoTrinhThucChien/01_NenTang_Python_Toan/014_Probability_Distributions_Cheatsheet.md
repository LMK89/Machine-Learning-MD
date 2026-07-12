## Bảng tính phân phối xác suất
Slide 1: Cơ sở của phân phối xác suất

Phân phối xác suất tạo thành xương sống của mô hình thống kê và học máy. Chúng mô tả khả năng xảy ra các kết quả khác nhau trong một quá trình ngẫu nhiên, cung cấp các công cụ toán học cần thiết để phân tích, suy luận và dự đoán dữ liệu.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate random data from different distributions
normal_data = np.random.normal(loc=0, scale=1, size=1000)
uniform_data = np.random.uniform(low=-3, high=3, size=1000)

# Create visualization
plt.figure(figsize=(12, 6))
plt.hist(normal_data, bins=30, alpha=0.5, label='Normal')
plt.hist(uniform_data, bins=30, alpha=0.5, label='Uniform')
plt.legend()
plt.title('Comparing Normal and Uniform Distributions')
plt.show()

# Calculate basic statistics
print(f"Normal mean: {normal_data.mean():.2f}, std: {normal_data.std():.2f}")
print(f"Uniform mean: {uniform_data.mean():.2f}, std: {uniform_data.std():.2f}")
```

Slide 2: Toán phân phối chuẩn

Phân phối chuẩn, còn được gọi là phân phối Gaussian, được đặc trưng bởi hàm mật độ xác suất (PDF). Nền tảng toán học bao gồm các tham số chính μ (trung bình) và σ (độ lệch chuẩn).

```python
# Mathematical representation of Normal Distribution PDF
"""
PDF formula:
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

where:
$$\mu$$ is the mean
$$\sigma$$ is the standard deviation
"""

def normal_pdf(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2)/(2 * sigma**2))

x = np.linspace(-5, 5, 1000)
pdf = normal_pdf(x, mu=0, sigma=1)

plt.plot(x, pdf)
plt.title('Standard Normal Distribution PDF')
plt.grid(True)
plt.show()
```

Trang trình bày 3: Thực hiện phân phối theo cấp số nhân

Phân bố hàm mũ mô hình hóa thời gian giữa các sự kiện trong quy trình Poisson. Nó thường được sử dụng trong kỹ thuật độ tin cậy và lý thuyết xếp hàng để mô hình hóa các khoảng thời gian giữa các sự kiện độc lập.

```python
def exponential_pdf(x, lambda_param):
    """
    $$f(x) = \lambda e^{-\lambda x}$$
    where λ is the rate parameter
    """
    return lambda_param * np.exp(-lambda_param * x)

x = np.linspace(0, 5, 1000)
lambdas = [0.5, 1, 2]

plt.figure(figsize=(10, 6))
for l in lambdas:
    plt.plot(x, exponential_pdf(x, l), label=f'λ={l}')

plt.title('Exponential Distribution PDF')
plt.legend()
plt.grid(True)
plt.show()

# Generate random samples
samples = np.random.exponential(scale=1/2, size=1000)
print(f"Mean: {samples.mean():.2f} (Expected: {1/2})")
```

Trang trình bày 4: Phân tích phân phối Chi-Square

Phân phối chi bình phương xuất hiện từ tổng bình phương các biến chuẩn chuẩn. Đó là nền tảng trong việc kiểm tra giả thuyết và xây dựng khoảng tin cậy để ước tính phương sai.

```python
def chi_square_pdf(x, df):
    """
    $$f(x) = \frac{x^{(k/2-1)}e^{-x/2}}{2^{k/2}\Gamma(k/2)}$$
    where k is degrees of freedom
    """
    return stats.chi2.pdf(x, df)

x = np.linspace(0, 15, 1000)
dfs = [1, 2, 5]

plt.figure(figsize=(10, 6))
for df in dfs:
    plt.plot(x, chi_square_pdf(x, df), label=f'df={df}')

plt.title('Chi-Square Distribution PDF')
plt.legend()
plt.grid(True)
plt.show()

# Generate chi-square samples
samples = np.random.chisquare(df=2, size=1000)
print(f"Mean: {samples.mean():.2f} (Expected: 2)")
```

Trang trình bày 5: Triển khai phân phối Poisson

Phân phối Poisson mô hình hóa số lượng sự kiện xảy ra trong một khoảng thời gian cố định khi những sự kiện này xảy ra với tỷ lệ trung bình đã biết và độc lập với thời gian kể từ sự kiện cuối cùng.

```python
def poisson_pmf(k, lambda_param):
    """
    $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
    where λ is the rate parameter
    """
    return (lambda_param**k * np.exp(-lambda_param)) / np.math.factorial(k)

k = np.arange(0, 15)
lambdas = [1, 4, 8]

plt.figure(figsize=(10, 6))
for l in lambdas:
    pmf = [poisson_pmf(ki, l) for ki in k]
    plt.plot(k, pmf, 'o-', label=f'λ={l}')

plt.title('Poisson Distribution PMF')
plt.legend()
plt.grid(True)
plt.show()

# Generate Poisson samples
samples = np.random.poisson(lam=4, size=1000)
print(f"Mean: {samples.mean():.2f} (Expected: 4)")
```

Slide 6: Phân phối nhị thức và ứng dụng

Phân phối nhị thức mô hình hóa số lần thành công trong một số thử nghiệm Bernoulli độc lập cố định. Mỗi phép thử có xác suất thành công như nhau và độc lập với các phép thử khác.

```python
def binomial_pmf(n, k, p):
    """
    $$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$
    where:
    n = number of trials
    k = number of successes
    p = probability of success
    """
    return stats.binom.pmf(k, n, p)

n = 20  # number of trials
p = 0.3  # probability of success
k = np.arange(0, n+1)

plt.figure(figsize=(10, 6))
pmf = binomial_pmf(n, k, p)
plt.bar(k, pmf, alpha=0.8)
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.grid(True)
plt.show()

# Generate samples
samples = np.random.binomial(n=20, p=0.3, size=1000)
print(f"Mean: {samples.mean():.2f} (Expected: {n*p})")
```

Trang trình bày 7: Phân phối Beta và ứng dụng Bayesian

Phân phối Beta rất quan trọng trong thống kê Bayes, đóng vai trò là phân phối liên hợp trước cho phân phối Bernoulli và nhị thức. Nó mô hình hóa các xác suất liên tục trong khoảng \[0,1\].

```python
def plot_beta_distribution(alphas, betas):
    """
    $$f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$$
    where B(α,β) is the Beta function
    """
    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(12, 6))

    for a, b in zip(alphas, betas):
        plt.plot(x, stats.beta.pdf(x, a, b),
                label=f'α={a}, β={b}')

    plt.title('Beta Distribution PDF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot different parameter combinations
alphas = [0.5, 5, 1]
betas = [0.5, 1, 3]
plot_beta_distribution(alphas, betas)

# Generate samples
samples = np.random.beta(a=2, b=5, size=1000)
print(f"Mean: {samples.mean():.3f}")
```

Trang trình bày 8: Phân phối chuẩn đa biến

Phân phối chuẩn đa biến mở rộng phân phối chuẩn tới các chiều cao hơn, cần thiết cho việc mô hình hóa các biến ngẫu nhiên tương quan và trong nhiều ứng dụng học máy.

```python
def multivariate_normal_example():
    """
    $$f(x) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}
    \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$
    """
    mean = [0, 0]
    cov = [[1, 0.5],
           [0.5, 2]]

    # Generate samples
    samples = np.random.multivariate_normal(mean, cov, 1000)

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title('Multivariate Normal Distribution Samples')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Calculate empirical correlation
    print(f"Empirical correlation: {np.corrcoef(samples.T)[0,1]:.3f}")
    print(f"Theoretical correlation: {cov[0][1]/np.sqrt(cov[0][0]*cov[1][1]):.3f}")

multivariate_normal_example()
```

Trang trình bày 9: Triển khai phân phối Gamma

Phân phối Gamma khái quát hóa phân bố theo cấp số nhân và được sử dụng rộng rãi trong mô hình hóa thời gian chờ đợi, kiểm tra tuổi thọ và như là phân phối liên hợp trước trong thống kê Bayes.

```python
def plot_gamma_distribution(alphas, betas):
    """
    $$f(x; \alpha, \beta) = \frac{\beta^\alpha x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}$$
    where α is shape and β is rate
    """
    x = np.linspace(0, 10, 1000)
    plt.figure(figsize=(12, 6))

    for a, b in zip(alphas, betas):
        plt.plot(x, stats.gamma.pdf(x, a, scale=1/b),
                label=f'α={a}, β={b}')

    plt.title('Gamma Distribution PDF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot different parameter combinations
alphas = [1, 2, 5]
betas = [1, 2, 1]
plot_gamma_distribution(alphas, betas)

# Generate samples
samples = np.random.gamma(shape=2, scale=1/2, size=1000)
print(f"Mean: {samples.mean():.3f} (Expected: {2/(2)})")
```

Slide 10: Ứng dụng thực tế - Phân tích lưu lượng mạng

Các gói tin đến mạng thường được mô hình hóa bằng cách sử dụng phân phối xác suất. Ví dụ này thể hiện việc phân tích các mẫu lưu lượng truy cập mạng bằng cách sử dụng phân phối Poisson và hàm mũ cho thời gian giữa các lần đến.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Simulate network packet arrivals
np.random.seed(42)
num_packets = 1000
arrival_rate = 5  # packets per second

# Generate inter-arrival times (exponential distribution)
inter_arrival_times = np.random.exponential(1/arrival_rate, num_packets)
arrival_times = np.cumsum(inter_arrival_times)

# Count packets in fixed intervals
interval_size = 1.0  # 1 second intervals
num_intervals = int(np.ceil(arrival_times[-1]))
packet_counts = np.zeros(num_intervals)

for time in arrival_times:
    interval = int(time)
    if interval < num_intervals:
        packet_counts[interval] += 1

# Statistical analysis
mean_packets = np.mean(packet_counts)
std_packets = np.std(packet_counts)

plt.figure(figsize=(12, 6))
plt.hist(packet_counts, bins=20, density=True, alpha=0.7)
x = np.arange(0, max(packet_counts)+1)
plt.plot(x, stats.poisson.pmf(x, mean_packets), 'r-', label='Poisson fit')
plt.title('Network Packet Arrivals Distribution')
plt.xlabel('Packets per Second')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

print(f"Mean packets per second: {mean_packets:.2f}")
print(f"Standard deviation: {std_packets:.2f}")
print(f"Theoretical std (Poisson): {np.sqrt(mean_packets):.2f}")
```

Slide 11: Ứng dụng thực tế - Mô hình hóa rủi ro tài chính

Việc triển khai này thể hiện việc sử dụng phân phối xác suất để lập mô hình lợi nhuận tài chính và ước tính Giá trị rủi ro (VaR) bằng cách sử dụng cả phân phối t thông thường và phân phối t của Sinh viên.

```python
def calculate_var_metrics(returns, confidence_levels=[0.95, 0.99]):
    """
    Returns Value at Risk (VaR) and Expected Shortfall (ES)
    $$VaR_\alpha = \mu + \sigma \Phi^{-1}(\alpha)$$
    where Φ⁻¹ is the inverse CDF of the standard normal distribution
    """
    mu = returns.mean()
    sigma = returns.std()

    results = {}
    for conf in confidence_levels:
        # Normal VaR
        var_normal = -stats.norm.ppf(1-conf, mu, sigma)

        # Student's t VaR (with estimated degrees of freedom)
        t_params = stats.t.fit(returns)
        var_student = -stats.t.ppf(1-conf, *t_params)

        results[conf] = {
            'VaR_normal': var_normal,
            'VaR_student': var_student
        }

    return results

# Generate sample financial returns
np.random.seed(42)
n_days = 1000
returns = np.random.normal(0.0001, 0.01, n_days)

# Add some fat-tail events
returns = np.append(returns, np.random.standard_t(df=3, size=50) * 0.02)

# Calculate VaR
var_results = calculate_var_metrics(returns)

plt.figure(figsize=(12, 6))
plt.hist(returns, bins=50, density=True, alpha=0.7)
x = np.linspace(min(returns), max(returns), 100)
plt.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
         'r-', label='Normal fit')
plt.plot(x, stats.t.pdf(x, *stats.t.fit(returns)),
         'g-', label='Student t fit')
plt.title('Financial Returns Distribution')
plt.legend()
plt.grid(True)
plt.show()

for conf, metrics in var_results.items():
    print(f"\nConfidence Level: {conf*100}%")
    print(f"Normal VaR: {metrics['VaR_normal']:.4f}")
    print(f"Student's t VaR: {metrics['VaR_student']:.4f}")
```

Trang trình bày 12: Ước tính mật độ hạt nhân (KDE)

KDE là một phương pháp phi tham số để ước tính các hàm mật độ xác suất. Nó đặc biệt hữu ích khi dữ liệu không tuân theo phân bố chuẩn và yêu cầu ước tính mật độ linh hoạt.

```python
def kde_estimation(data, bandwidths=[0.1, 0.3, 0.5]):
    """
    $$\hat{f}_h(x) = \frac{1}{nh}\sum_{i=1}^n K\left(\frac{x-x_i}{h}\right)$$
    where K is the kernel function and h is the bandwidth
    """
    x_grid = np.linspace(min(data)-1, max(data)+1, 200)

    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=30, density=True, alpha=0.3, label='Data')

    for bw in bandwidths:
        kde = stats.gaussian_kde(data, bw_method=bw)
        plt.plot(x_grid, kde(x_grid),
                label=f'KDE (bandwidth={bw})')

    plt.title('Kernel Density Estimation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate mixture of normal distributions
np.random.seed(42)
data = np.concatenate([
    np.random.normal(-2, 0.5, 300),
    np.random.normal(1, 1, 700)
])

kde_estimation(data)
print(f"Sample statistics:")
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
print(f"Skewness: {stats.skew(data):.3f}")
print(f"Kurtosis: {stats.kurtosis(data):.3f}")
```

Trang trình bày 13: Triển khai mô hình hỗn hợp

Các mô hình hỗn hợp kết hợp nhiều phân bố xác suất để mô hình hóa các mẫu dữ liệu phức tạp. Triển khai này giới thiệu Mô hình hỗn hợp Gaussian (GMM) với khả năng tối đa hóa kỳ vọng để ước tính tham số.

```python
from sklearn.mixture import GaussianMixture

def fit_gaussian_mixture(data, n_components=2):
    """
    Gaussian Mixture Model:
    $$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k,\Sigma_k)$$
    where πk are mixing coefficients
    """
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data.reshape(-1, 1))

    # Plot results
    x = np.linspace(data.min()-1, data.max()+1, 1000).reshape(-1, 1)
    scores = np.exp(gmm.score_samples(x))
    responsibilities = gmm.predict_proba(x)

    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=50, density=True, alpha=0.5)
    plt.plot(x, scores, 'r-', label='GMM density')

    for i in range(n_components):
        plt.plot(x, responsibilities[:, i] * scores,
                '--', label=f'Component {i+1}')

    plt.title('Gaussian Mixture Model Fit')
    plt.legend()
    plt.grid(True)
    plt.show()

    return gmm

# Generate synthetic data from mixture
np.random.seed(42)
data = np.concatenate([
    np.random.normal(-2, 0.5, 300),
    np.random.normal(2, 1.0, 700)
])

gmm = fit_gaussian_mixture(data)
print("Mixture Parameters:")
for i, (mean, covar, weight) in enumerate(zip(
    gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_
)):
    print(f"\nComponent {i+1}:")
    print(f"Mean: {mean:.3f}")
    print(f"Variance: {covar:.3f}")
    print(f"Weight: {weight:.3f}")
```

Trang trình bày 14: Kiểm tra phân phối và mức độ phù hợp

Kiểm tra thống kê giúp xác định xem dữ liệu có tuân theo một phân phối cụ thể hay không. Việc triển khai này bao gồm nhiều bài kiểm tra mức độ phù hợp và diễn giải của chúng.

```python
def distribution_testing(data, alpha=0.05):
    """
    Implements multiple distribution tests:
    - Shapiro-Wilk test for normality
    - Anderson-Darling test
    - Kolmogorov-Smirnov test
    """
    # Visual QQ plot
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Q-Q Plot")

    plt.subplot(122)
    plt.hist(data, bins='auto', density=True, alpha=0.7)
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(data), np.std(data)),
            'r-', label='Normal fit')
    plt.title("Histogram with Normal Fit")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Statistical tests
    shapiro_stat, shapiro_p = stats.shapiro(data)
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    ad_result = stats.anderson(data, dist='norm')

    print("\nNormality Tests Results:")
    print(f"Shapiro-Wilk test: p-value = {shapiro_p:.4f}")
    print(f"Kolmogorov-Smirnov test: p-value = {ks_p:.4f}")
    print("\nAnderson-Darling test:")
    for i in range(len(ad_result.critical_values)):
        sig = (1 - float(ad_result.significance_level[i])/100)
        print(f"At {sig:.2f} significance level: ", end="")
        if ad_result.statistic < ad_result.critical_values[i]:
            print("Normal")
        else:
            print("Non-normal")

# Generate test data
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
skewed_data = np.random.gamma(2, 2, 1000)

print("Testing Normal Data:")
distribution_testing(normal_data)
print("\nTesting Skewed Data:")
distribution_testing(skewed_data)
```

Trang trình bày 15: Tài nguyên bổ sung

* "Khảo sát về phân bố xác suất với các ứng dụng" - arXiv:1907.09952
* "Các phương pháp thống kê hiện đại cho phân phối đuôi nặng" - arXiv:2104.12883
* "Kiểm tra thống kê phi tham số về phân phối" - arXiv:1904.12956
* "Các phương pháp thực tế để lắp mô hình hỗn hợp" - [https://www.sciencedirect.com/topics/mathematics/mixture-distribution](https://www.sciencedirect.com/topics/mathematics/mixture-distribution)
* "Phương pháp tính toán để kiểm tra phân phối" - [https://dl.acm.org/doi/10.1145/3460120](https://dl.acm.org/doi/10.1145/3460120)
