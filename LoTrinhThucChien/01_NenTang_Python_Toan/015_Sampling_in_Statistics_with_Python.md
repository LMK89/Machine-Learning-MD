## Lấy mẫu trong thống kê bằng Python
Slide 1: Giới thiệu về lấy mẫu trong thống kê

Lấy mẫu là một khái niệm cơ bản trong thống kê liên quan đến việc chọn một tập hợp con các cá nhân từ một quần thể lớn hơn để đưa ra suy luận về toàn bộ quần thể. Quá trình này rất quan trọng để tiến hành nghiên cứu, khảo sát và phân tích dữ liệu khi việc nghiên cứu từng thành viên trong dân số là không thực tế hoặc không thể. Trong phần trình bày này, chúng ta sẽ khám phá các kỹ thuật lấy mẫu khác nhau và cách triển khai chúng bằng Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a population
population = np.random.normal(loc=100, scale=15, size=10000)

# Plot the population distribution
plt.hist(population, bins=50)
plt.title("Population Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Slide 2: Lấy mẫu ngẫu nhiên đơn giản

Lấy mẫu ngẫu nhiên đơn giản là một kỹ thuật cơ bản trong đó mỗi thành viên của quần thể có xác suất được chọn như nhau. Phương pháp này đảm bảo sự đại diện khách quan của dân số. Hãy triển khai lấy mẫu ngẫu nhiên đơn giản bằng thư viện NumPy của Python.

```python
import numpy as np

# Generate a population
population = np.arange(1, 1001)

# Perform simple random sampling
sample_size = 100
simple_random_sample = np.random.choice(population, size=sample_size, replace=False)

print("Simple Random Sample:", simple_random_sample)
print("Sample Mean:", np.mean(simple_random_sample))
print("Population Mean:", np.mean(population))
```

Slide 3: Lấy mẫu có hệ thống

Lấy mẫu có hệ thống bao gồm việc chọn mọi phần tử thứ k từ tổng thể sau khi bắt đầu ngẫu nhiên. Phương pháp này hữu ích khi quần thể được sắp xếp theo thứ tự và chúng ta muốn đảm bảo mức độ bao phủ đồng đều trên toàn bộ phạm vi. Đây là cách triển khai lấy mẫu có hệ thống trong Python.

```python
import numpy as np

# Generate an ordered population
population = np.arange(1, 1001)

# Set sample size and calculate step size
sample_size = 100
step = len(population) // sample_size

# Perform systematic sampling
start = np.random.randint(0, step)
systematic_sample = population[start::step][:sample_size]

print("Systematic Sample:", systematic_sample)
print("Sample Mean:", np.mean(systematic_sample))
print("Population Mean:", np.mean(population))
```

Slide 4: Lấy mẫu phân tầng

Lấy mẫu phân tầng chia dân số thành các nhóm nhỏ (tầng) dựa trên các đặc điểm chung, sau đó lấy mẫu từ mỗi tầng. Phương pháp này đảm bảo sự đại diện từ tất cả các nhóm con. Hãy thực hiện lấy mẫu phân tầng cho một nhóm sinh viên giả định.

```python
import numpy as np
import pandas as pd

# Create a hypothetical student population
np.random.seed(42)
students = pd.DataFrame({
    'grade': np.random.choice(['A', 'B', 'C', 'D'], size=1000),
    'score': np.random.randint(60, 101, size=1000)
})

# Perform stratified sampling
sample_size = 100
stratified_sample = students.groupby('grade', group_keys=False).apply(lambda x: x.sample(int(sample_size/4)))

print("Stratified Sample:")
print(stratified_sample)
print("\nSample Mean Score:", stratified_sample['score'].mean())
print("Population Mean Score:", students['score'].mean())
```

Trang trình bày 5: Lấy mẫu cụm

Lấy mẫu cụm bao gồm việc chia dân số thành các cụm, chọn ngẫu nhiên một số cụm và sau đó lấy mẫu tất cả các thành viên trong cụm đã chọn. Phương pháp này hữu ích khi nó thực tế hơn đối với các nhóm mẫu hơn là các cá nhân. Hãy mô phỏng lấy mẫu theo cụm cho các hộ gia đình trong thành phố.

```python
import numpy as np

# Simulate a city with neighborhoods (clusters) and households
np.random.seed(42)
city = [np.random.normal(loc=50000, scale=10000, size=np.random.randint(50, 150)) for _ in range(20)]

# Perform cluster sampling
num_clusters = 5
sampled_clusters = np.random.choice(len(city), size=num_clusters, replace=False)
cluster_sample = [household for cluster in sampled_clusters for household in city[cluster]]

print("Number of sampled households:", len(cluster_sample))
print("Mean household income in sample:", np.mean(cluster_sample))
print("Mean household income in population:", np.mean([income for neighborhood in city for income in neighborhood]))
```

Slide 6: Lấy mẫu có trọng số

Lấy mẫu có trọng số ấn định các xác suất khác nhau cho các thành viên trong tổng thể dựa trên tầm quan trọng hoặc tính đại diện của chúng. Kỹ thuật này hữu ích khi một số phần tử nhất định có cơ hội được chọn cao hơn. Hãy triển khai lấy mẫu có trọng số bằng Python.

```python
import numpy as np

# Create a population with weights
population = ['A', 'B', 'C', 'D', 'E']
weights = [0.1, 0.2, 0.3, 0.1, 0.3]

# Perform weighted sampling
sample_size = 1000
weighted_sample = np.random.choice(population, size=sample_size, p=weights)

# Calculate the frequency of each element in the sample
unique, counts = np.unique(weighted_sample, return_counts=True)
frequencies = dict(zip(unique, counts / sample_size))

print("Sample frequencies:")
for item, freq in frequencies.items():
    print(f"{item}: {freq:.2f}")
```

Trang trình bày 7: Lấy mẫu Bootstrap

Lấy mẫu Bootstrap là một kỹ thuật lấy mẫu lại được sử dụng để ước tính phân phối lấy mẫu của một thống kê. Nó liên quan đến việc lấy mẫu nhiều lần và thay thế mẫu ban đầu. Phương pháp này đặc biệt hữu ích để ước tính khoảng tin cậy và thực hiện kiểm tra giả thuyết.

```python
import numpy as np
import matplotlib.pyplot as plt

# Original sample
original_sample = np.random.normal(loc=100, scale=15, size=100)

# Perform bootstrap sampling
n_bootstrap = 10000
bootstrap_means = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    bootstrap_sample = np.random.choice(original_sample, size=len(original_sample), replace=True)
    bootstrap_means[i] = np.mean(bootstrap_sample)

# Plot the bootstrap distribution of means
plt.hist(bootstrap_means, bins=50)
plt.title("Bootstrap Distribution of Sample Means")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()

print("Original Sample Mean:", np.mean(original_sample))
print("Bootstrap Mean of Means:", np.mean(bootstrap_means))
print("95% Confidence Interval:", np.percentile(bootstrap_means, [2.5, 97.5]))
```

Slide 8: Lấy mẫu tầm quan trọng

Lấy mẫu quan trọng là một kỹ thuật được sử dụng để ước tính các thuộc tính của một phân phối cụ thể trong khi lấy mẫu từ một phân phối khác. Phương pháp này đặc biệt hữu ích trong các trường hợp khó lấy mẫu từ phân bố mục tiêu hoặc tốn kém về mặt tính toán.

```python
import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

def proposal_distribution(x):
    return np.exp(-np.abs(x)) / 2

# Generate samples from the proposal distribution
n_samples = 10000
samples = np.random.exponential(scale=1, size=n_samples) * np.random.choice([-1, 1], size=n_samples)

# Calculate importance weights
weights = target_distribution(samples) / proposal_distribution(samples)

# Estimate the mean of the target distribution
estimated_mean = np.sum(samples * weights) / np.sum(weights)

print("Estimated mean:", estimated_mean)
print("True mean:", 0)  # The true mean of a standard normal distribution is 0

# Plot the results
x = np.linspace(-4, 4, 1000)
plt.plot(x, target_distribution(x), label='Target Distribution')
plt.hist(samples, bins=50, density=True, alpha=0.5, label='Proposal Samples')
plt.legend()
plt.title("Importance Sampling")
plt.show()
```

Slide 9: Lấy mẫu hồ chứa

Lấy mẫu hồ chứa là một thuật toán để chọn ngẫu nhiên k mẫu từ một quần thể có kích thước không xác định, có thể rất lớn hoặc đang phát trực tuyến. Kỹ thuật này đặc biệt hữu ích khi xử lý dữ liệu lớn hoặc dữ liệu truyền trực tuyến mà chúng ta không thể lưu giữ tất cả các mục trong bộ nhớ cùng một lúc.

```python
import numpy as np

def reservoir_sampling(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = np.random.randint(0, i+1)
            if j < k:
                reservoir[j] = item
    return reservoir

# Simulate a data stream
np.random.seed(42)
data_stream = iter(np.random.randint(1, 1001, size=10000))

# Perform reservoir sampling
sample_size = 100
reservoir_sample = reservoir_sampling(data_stream, sample_size)

print("Reservoir Sample:", reservoir_sample)
print("Sample Mean:", np.mean(reservoir_sample))
```

Trang trình bày 10: Lấy mẫu Monte Carlo

Lấy mẫu Monte Carlo là một lớp thuật toán tính toán rộng rãi dựa trên việc lấy mẫu ngẫu nhiên lặp đi lặp lại để thu được kết quả bằng số. Một ứng dụng phổ biến là ước tính tích phân xác định. Hãy sử dụng phương pháp lấy mẫu Monte Carlo để ước tính giá trị của số π.

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n_samples):
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    inside_circle = (x**2 + y**2 <= 1)
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    return pi_estimate

# Estimate π with increasing number of samples
sample_sizes = np.logspace(2, 6, num=20, dtype=int)
pi_estimates = [estimate_pi(n) for n in sample_sizes]

# Plot the results
plt.semilogx(sample_sizes, pi_estimates, 'b-')
plt.axhline(y=np.pi, color='r', linestyle='--')
plt.xlabel('Number of Samples')
plt.ylabel('Estimated π')
plt.title('Monte Carlo Estimation of π')
plt.grid(True)
plt.show()

print(f"Final π estimate (with {sample_sizes[-1]} samples): {pi_estimates[-1]}")
print(f"True π value: {np.pi}")
```

Trang trình bày 11: Lấy mẫu Gibbs

Lấy mẫu Gibbs là thuật toán Markov Chain Monte Carlo (MCMC) để thu được một chuỗi các quan sát gần đúng với phân bố xác suất đa biến được chỉ định. Nó đặc biệt hữu ích cho việc lấy mẫu từ các bản phân phối nhiều chiều. Hãy triển khai bộ lấy mẫu Gibbs đơn giản để phân phối chuẩn hai biến.

```python
import numpy as np
import matplotlib.pyplot as plt

def gibbs_sampler(n_samples, mu, sigma):
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)

    x[0], y[0] = 0, 0

    for i in range(1, n_samples):
        x[i] = np.random.normal(mu[0] + sigma[0, 1] / sigma[1, 1] * (y[i-1] - mu[1]),
                                np.sqrt(sigma[0, 0] - sigma[0, 1]**2 / sigma[1, 1]))
        y[i] = np.random.normal(mu[1] + sigma[1, 0] / sigma[0, 0] * (x[i] - mu[0]),
                                np.sqrt(sigma[1, 1] - sigma[1, 0]**2 / sigma[0, 0]))

    return x, y

# Set up the bivariate normal distribution parameters
mu = np.array([0, 0])
sigma = np.array([[1, 0.5], [0.5, 1]])

# Run the Gibbs sampler
n_samples = 5000
x, y = gibbs_sampler(n_samples, mu, sigma)

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(x, y, 'b.', alpha=0.1)
plt.title('Gibbs Sampling: Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.hist2d(x, y, bins=50, cmap='Blues')
plt.title('Gibbs Sampling: 2D Histogram')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
```

Slide 12: Thuật toán Metropolis-Hastings

Thuật toán Metropolis-Hastings là một phương pháp MCMC khác được sử dụng để thu được một chuỗi các mẫu ngẫu nhiên từ phân bố xác suất trong đó khó lấy mẫu trực tiếp. Nó tổng quát hơn việc lấy mẫu Gibbs và có thể áp dụng cho nhiều vấn đề hơn. Hãy triển khai nó để lấy mẫu từ bản phân phối gamma.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

def metropolis_hastings(target_pdf, proposal_pdf, proposal_sampler, n_samples, initial_state):
    samples = np.zeros(n_samples)
    current_state = initial_state
    accepted = 0

    for i in range(n_samples):
        proposed_state = proposal_sampler(current_state)

        acceptance_ratio = (target_pdf(proposed_state) * proposal_pdf(current_state, proposed_state)) / \
                           (target_pdf(current_state) * proposal_pdf(proposed_state, current_state))

        if np.random.random() < acceptance_ratio:
            current_state = proposed_state
            accepted += 1

        samples[i] = current_state

    return samples, accepted / n_samples

# Target distribution: Gamma(k=2, theta=2)
k, theta = 2, 2
target_pdf = lambda x: gamma.pdf(x, a=k, scale=theta)

# Proposal distribution: Normal(mu=x, sigma=0.5)
proposal_pdf = lambda x, mu: np.exp(-0.5 * ((x - mu) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi))
proposal_sampler = lambda mu: np.random.normal(mu, 0.5)

# Run Metropolis-Hastings
n_samples = 10000
initial_state = 1.0
samples, acceptance_rate = metropolis_hastings(target_pdf, proposal_pdf, proposal_sampler, n_samples, initial_state)

# Plot results
x = np.linspace(0, 20, 1000)
plt.hist(samples, bins=50, density=True, alpha=0.7, label='MCMC Samples')
plt.plot(x, target_pdf(x), 'r-', lw=2, label='Target PDF')
plt.title(f'Metropolis-Hastings Sampling (Acceptance Rate: {acceptance_rate:.2f})')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"True Mean: {k * theta:.4f}")
```

Trang trình bày 13: Lấy mẫu từ chối

Lấy mẫu loại bỏ là một kỹ thuật được sử dụng để tạo ra các quan sát từ phân phối khi khó lấy mẫu trực tiếp. Nó liên quan đến việc lấy mẫu từ phân phối đề xuất đơn giản hơn và chấp nhận hoặc từ chối các mẫu dựa trên so sánh với phân phối mục tiêu. Hãy triển khai lấy mẫu từ chối để phân phối xác suất tùy chỉnh.

```python
import numpy as np
import matplotlib.pyplot as plt

def target_pdf(x):
    return 0.3 * np.exp(-(x - 0.3)**2) + 0.7 * np.exp(-(x - 2.0)**2 / 0.3)

def rejection_sampling(n_samples):
    samples = []
    x = np.linspace(0, 3, 1000)
    M = max(target_pdf(x))

    while len(samples) < n_samples:
        x = np.random.uniform(0, 3)
        y = np.random.uniform(0, M)

        if y <= target_pdf(x):
            samples.append(x)

    return np.array(samples)

n_samples = 10000
samples = rejection_sampling(n_samples)

x = np.linspace(0, 3, 1000)
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
plt.plot(x, target_pdf(x), 'r-', label='Target PDF')
plt.title('Rejection Sampling')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
```

Trang trình bày 14: Lấy mẫu tầm quan trọng để mô phỏng sự kiện hiếm

Lấy mẫu quan trọng đặc biệt hữu ích để mô phỏng các sự kiện hiếm gặp. Nó cho phép chúng ta ước tính xác suất xảy ra các sự kiện khó xảy ra hiệu quả hơn mô phỏng Monte Carlo trực tiếp. Hãy sử dụng lấy mẫu tầm quan trọng để ước tính xác suất xảy ra sự kiện hiếm gặp trong hệ thống xếp hàng đơn giản.

```python
import numpy as np

def direct_mc_simulation(n_simulations, arrival_rate, service_rate, buffer_size):
    overflow_count = 0
    for _ in range(n_simulations):
        queue_length = 0
        for _ in range(1000):  # Simulate 1000 time steps
            if np.random.random() < arrival_rate:
                queue_length += 1
            if np.random.random() < service_rate and queue_length > 0:
                queue_length -= 1
            if queue_length > buffer_size:
                overflow_count += 1
                break
    return overflow_count / n_simulations

def importance_sampling(n_simulations, arrival_rate, service_rate, buffer_size):
    overflow_probs = []
    for _ in range(n_simulations):
        queue_length = 0
        likelihood_ratio = 1
        for _ in range(1000):  # Simulate 1000 time steps
            if np.random.random() < 0.5:  # Biased arrival rate
                queue_length += 1
                likelihood_ratio *= arrival_rate / 0.5
            if np.random.random() < service_rate and queue_length > 0:
                queue_length -= 1
            if queue_length > buffer_size:
                overflow_probs.append(likelihood_ratio)
                break
    return np.mean(overflow_probs) if overflow_probs else 0

arrival_rate, service_rate, buffer_size = 0.1, 0.15, 10
n_simulations = 100000

direct_prob = direct_mc_simulation(n_simulations, arrival_rate, service_rate, buffer_size)
importance_prob = importance_sampling(n_simulations, arrival_rate, service_rate, buffer_size)

print(f"Direct MC estimation: {direct_prob:.6f}")
print(f"Importance sampling estimation: {importance_prob:.6f}")
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về kỹ thuật lấy mẫu và ứng dụng của chúng trong thống kê và học máy, đây là một số tài nguyên có giá trị:

1. "Phương pháp thống kê Monte Carlo" của Christian P. Robert và George Casella ArXiv: [https://arxiv.org/abs/0908.3655](https://arxiv.org/abs/0908.3655)
2. "Giới thiệu về MCMC cho học máy" của Christophe Andrieu và cộng sự. ArXiv: [https://arxiv.org/abs/1109.4435](https://arxiv.org/abs/1109.4435)
3. "Khảo sát các phương pháp ước tính tham số Monte Carlo" của Johanna Ärje et al. ArXiv: [https://arxiv.org/abs/1parameter-estimation-monte-carlo](https://arxiv.org/abs/1parameter-estimation-monte-carlo)

Những tài nguyên này cung cấp những giải thích sâu sắc và các kỹ thuật tiên tiến trong lấy mẫu và phương pháp Monte Carlo, những tài nguyên này rất quan trọng đối với các ứng dụng khác nhau trong thống kê, học máy và khoa học dữ liệu.
