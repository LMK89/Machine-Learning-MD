## Thống kê Bayesian và Thống kê thường xuyên Các khái niệm và ứng dụng chính
Trang trình bày 1: Bayesian vs Người theo chủ nghĩa thường xuyên - Các khái niệm cốt lõi

Sự khác biệt cơ bản giữa phương pháp tiếp cận Bayes và phương pháp thường xuyên nằm ở cách xử lý xác suất và các tham số. Việc triển khai này thể hiện ước tính tham số cơ bản bằng cách sử dụng cả hai phương pháp thông qua thử nghiệm lật đồng xu đơn giản.

```python
import numpy as np
from scipy import stats

# Simulate coin flips
np.random.seed(42)
n_flips = 100
data = np.random.binomial(1, 0.7, n_flips)  # True probability = 0.7

# Frequentist approach - Maximum Likelihood Estimation
freq_estimate = np.mean(data)
std_error = np.sqrt(freq_estimate * (1 - freq_estimate) / n_flips)
confidence_interval = (freq_estimate - 1.96 * std_error,
                      freq_estimate + 1.96 * std_error)

# Bayesian approach - Beta conjugate prior
prior_a, prior_b = 1, 1  # Uniform prior
posterior_a = prior_a + sum(data)
posterior_b = prior_b + len(data) - sum(data)
bayesian_estimate = posterior_a / (posterior_a + posterior_b)

print(f"Frequentist MLE: {freq_estimate:.3f}")
print(f"95% CI: {confidence_interval}")
print(f"Bayesian MAP: {bayesian_estimate:.3f}")
```

Trang trình bày 2: Triển khai phân phối trước

Phân phối trước thể hiện niềm tin ban đầu của chúng tôi về các tham số trước khi quan sát dữ liệu. Việc triển khai này cho thấy cách tạo và trực quan hóa các loại phân phối trước đó khác nhau thường được sử dụng trong phân tích Bayes.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_prior_distributions():
    x = np.linspace(0, 1, 1000)

    # Different types of priors
    uniform_prior = stats.uniform.pdf(x, 0, 1)
    beta_informative = stats.beta.pdf(x, 30, 10)
    beta_uninformative = stats.beta.pdf(x, 1, 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, uniform_prior, label='Uniform Prior')
    plt.plot(x, beta_informative, label='Informative Beta(30,10)')
    plt.plot(x, beta_uninformative, label='Uninformative Beta(1,1)')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.title('Different Prior Distributions')
    plt.legend()
    plt.grid(True)
    return plt

# Example usage
plot_prior_distributions()
plt.show()
```

Trang trình bày 3: Triển khai chức năng khả năng

Hàm khả năng biểu thị xác suất quan sát dữ liệu của chúng tôi với các giá trị tham số cụ thể. Việc triển khai này trình bày cách tính toán và trực quan hóa các hàm khả năng cho cả trường hợp rời rạc và liên tục.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def compute_likelihood(data, theta_range):
    likelihood = np.zeros(len(theta_range))

    for i, theta in enumerate(theta_range):
        # Compute probability of each observation
        likelihood[i] = np.prod(norm.pdf(data, theta, 1))

    return likelihood

# Generate sample data
np.random.seed(42)
true_mean = 2.5
data = np.random.normal(true_mean, 1, 50)

# Compute likelihood over range of possible means
theta_range = np.linspace(0, 5, 1000)
likelihood = compute_likelihood(data, theta_range)

plt.figure(figsize=(10, 6))
plt.plot(theta_range, likelihood)
plt.axvline(true_mean, color='r', linestyle='--', label='True Mean')
plt.xlabel('Parameter Value (θ)')
plt.ylabel('Likelihood')
plt.title('Likelihood Function')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 4: Tính toán phân phối sau

Phân phối sau kết hợp niềm tin trước đó với dữ liệu được quan sát thông qua định lý Bayes. Việc triển khai này thể hiện tính toán số của các phân phối sau cho mô hình Gaussian với giá trị trung bình chưa xác định.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def compute_posterior(data, prior_mu, prior_sigma, likelihood_sigma):
    n = len(data)
    sample_mean = np.mean(data)

    # Posterior parameters (conjugate normal-normal)
    posterior_sigma = 1 / (1/prior_sigma**2 + n/likelihood_sigma**2)
    posterior_mu = posterior_sigma * (prior_mu/prior_sigma**2 +
                                    n*sample_mean/likelihood_sigma**2)

    # Generate posterior distribution points
    x = np.linspace(posterior_mu - 4*np.sqrt(posterior_sigma),
                    posterior_mu + 4*np.sqrt(posterior_sigma), 1000)
    posterior = stats.norm.pdf(x, posterior_mu, np.sqrt(posterior_sigma))

    return x, posterior, posterior_mu, posterior_sigma

# Example usage
np.random.seed(42)
data = np.random.normal(2.5, 1, 30)
x, posterior, mu, sigma = compute_posterior(data, 0, 2, 1)

plt.figure(figsize=(10, 6))
plt.plot(x, posterior, label=f'Posterior (μ={mu:.2f}, σ={np.sqrt(sigma):.2f})')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.title('Posterior Distribution')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 5: Triển khai Markov Chain Monte Carlo (MCMC)

Phương pháp MCMC cho phép lấy mẫu từ các phân bố hậu nghiệm phức tạp trong đó các giải pháp phân tích khó có thể thực hiện được. Việc triển khai này hiển thị thuật toán Metropolis-Hastings để ước tính tham số.

```python
import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings(data, n_iterations, proposal_width):
    current = np.mean(data)  # Start at MLE
    samples = np.zeros(n_iterations)

    def log_posterior(theta):
        # Log prior (uniform improper prior)
        if theta < 0 or theta > 10:
            return -np.inf
        # Log likelihood (normal)
        return -0.5 * np.sum((data - theta)**2)

    for i in range(n_iterations):
        # Propose new value
        proposal = current + np.random.normal(0, proposal_width)

        # Compute acceptance ratio
        log_ratio = log_posterior(proposal) - log_posterior(current)

        # Accept or reject
        if np.log(np.random.random()) < log_ratio:
            current = proposal

        samples[i] = current

    return samples

# Example usage
np.random.seed(42)
true_mean = 2.5
data = np.random.normal(true_mean, 1, 100)
samples = metropolis_hastings(data, 10000, 0.5)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(samples)
plt.title('MCMC Trace')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')

plt.subplot(122)
plt.hist(samples[1000:], bins=50, density=True)
plt.axvline(true_mean, color='r', linestyle='--', label='True Value')
plt.title('Posterior Samples')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()
```

Trang trình bày 6: Hồi quy tuyến tính Bayes

Hồi quy tuyến tính Bayes mở rộng hồi quy cổ điển bằng cách cung cấp phân phối sau đầy đủ cho các tham số. Việc triển khai này thể hiện một trường hợp đơn giản với tính toán hậu nghiệm phân tích.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # Prior precision
        self.beta = beta   # Likelihood precision

    def fit(self, X, y):
        n = len(X)
        # Compute posterior parameters
        self.S_N = np.linalg.inv(self.alpha * np.eye(2) +
                                self.beta * X.T @ X)
        self.m_N = self.beta * self.S_N @ X.T @ y

        return self

    def predict(self, X_new, return_std=False):
        y_mean = X_new @ self.m_N

        if return_std:
            y_std = np.sqrt(1/self.beta +
                           np.sum(X_new @ self.S_N * X_new, axis=1))
            return y_mean, y_std
        return y_mean

# Generate example data
np.random.seed(42)
X = np.random.uniform(-5, 5, 20)
y = 2 * X + 1 + np.random.normal(0, 1, 20)

# Prepare data
X_design = np.column_stack([np.ones_like(X), X])
X_test = np.linspace(-6, 6, 100)
X_test_design = np.column_stack([np.ones_like(X_test), X_test])

# Fit model
model = BayesianLinearRegression(alpha=0.1, beta=1.0)
model.fit(X_design, y)

# Predict
y_pred, y_std = model.predict(X_test_design, return_std=True)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='black', label='Data')
plt.plot(X_test, y_pred, 'r-', label='Mean prediction')
plt.fill_between(X_test, y_pred - 2*y_std, y_pred + 2*y_std,
                 color='r', alpha=0.2, label='95% CI')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Bayesian Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 7: So sánh việc kiểm tra giả thuyết thường xuyên và giả thuyết Bayes

Việc triển khai này thể hiện sự khác biệt chính giữa kiểm tra giả thuyết Thường xuyên (giá trị p) và kiểm tra giả thuyết Bayes (yếu tố Bayes) bằng cách sử dụng một ví dụ đơn giản về so sánh hai nhóm.

```python
import numpy as np
from scipy import stats
from scipy.special import betaln

def bayes_factor_t_test(x1, x2):
    n1, n2 = len(x1), len(x2)
    t_stat, p_val = stats.ttest_ind(x1, x2)

    # Calculate Bayes Factor (JZS prior)
    df = n1 + n2 - 2
    t2 = t_stat**2
    bf10 = np.exp(betaln(0.5, df/2) - betaln(0.5, 0.5) +
                  0.5 * (np.log(2/df) + t2/(1 + t2/df)))

    return t_stat, p_val, bf10

# Generate example data
np.random.seed(42)
group1 = np.random.normal(0, 1, 30)
group2 = np.random.normal(0.5, 1, 30)

# Perform both tests
t_stat, p_val, bf10 = bayes_factor_t_test(group1, group2)

print(f"Frequentist t-test:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.3f}")
print(f"\nBayesian analysis:")
print(f"Bayes Factor (BF10): {bf10:.3f}")

# Visualize distributions
plt.figure(figsize=(10, 6))
plt.hist(group1, alpha=0.5, label='Group 1', density=True)
plt.hist(group2, alpha=0.5, label='Group 2', density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Group Distributions')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: Lựa chọn mô hình Bayesian

Triển khai lựa chọn mô hình Bayesian bằng cách sử dụng Tiêu chí thông tin Bayesian (BIC) và so sánh bằng chứng mô hình để lựa chọn giữa các mô hình cạnh tranh.

```python
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

class BayesianModelSelection:
    def __init__(self, models):
        self.models = models
        self.bic_scores = []
        self.evidence = []

    def compute_bic(self, X, y):
        for degree in self.models:
            # Create polynomial features
            X_poly = np.polynomial.polynomial.polyvander(X, degree)

            # Fit model
            model = LinearRegression()
            model.fit(X_poly, y)

            # Compute predictions and residuals
            y_pred = model.predict(X_poly)
            residuals = y - y_pred

            # Compute BIC
            n = len(y)
            mse = np.mean(residuals**2)
            bic = n * np.log(mse) + (degree + 1) * np.log(n)

            self.bic_scores.append(bic)

    def plot_results(self, X, y):
        plt.figure(figsize=(12, 5))

        # Plot data and fitted models
        plt.subplot(121)
        plt.scatter(X, y, c='black', label='Data')

        X_test = np.linspace(X.min(), X.max(), 100)
        for degree in self.models:
            X_poly = np.polynomial.polynomial.polyvander(X, degree)
            X_test_poly = np.polynomial.polynomial.polyvander(X_test, degree)

            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_test_poly)

            plt.plot(X_test, y_pred, label=f'Degree {degree}')

        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Model Fits')
        plt.legend()

        # Plot BIC scores
        plt.subplot(122)
        plt.plot(self.models, self.bic_scores, 'o-')
        plt.xlabel('Model Degree')
        plt.ylabel('BIC Score')
        plt.title('Model Selection via BIC')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Generate example data
np.random.seed(42)
X = np.linspace(-3, 3, 50)
y = 1 + 2*X + 0.5*X**2 + np.random.normal(0, 0.5, 50)

# Perform model selection
model_selector = BayesianModelSelection(models=[1, 2, 3, 4])
model_selector.compute_bic(X, y)
model_selector.plot_results(X, y)
```

Slide 9: Mô hình phân cấp Bayesian

Các mô hình phân cấp nắm bắt nhiều cấp độ biến đổi của dữ liệu. Việc triển khai này thể hiện một mô hình phân cấp đơn giản để phân tích dữ liệu được nhóm với việc gộp một phần thông tin giữa các nhóm.

```python
import numpy as np
import pymc3 as pm

def hierarchical_model(data_groups):
    with pm.Model() as hierarchical_model:
        # Hyperpriors
        mu = pm.Normal('mu', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)

        # Group-level parameters
        group_means = pm.Normal('group_means',
                              mu=mu,
                              sd=sigma,
                              shape=len(data_groups))

        # Observations
        for idx, data in enumerate(data_groups):
            pm.Normal(f'obs_{idx}',
                     mu=group_means[idx],
                     sd=1,
                     observed=data)

        # Inference
        trace = pm.sample(2000, tune=1000, return_inferencedata=False)

    return trace

# Generate example data
np.random.seed(42)
true_means = [1, 3, -2]
groups = [np.random.normal(mu, 1, 30) for mu in true_means]

# Fit model
trace = hierarchical_model(groups)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Plot raw data
plt.subplot(121)
plt.boxplot(groups)
plt.plot([true_means], 'r*', markersize=10, label='True Means')
plt.xlabel('Group')
plt.ylabel('Value')
plt.title('Raw Data Distribution')

# Plot posterior distributions
plt.subplot(122)
for i in range(len(groups)):
    pm.plot_posterior(trace, var_names=[f'group_means__{i}'])
plt.title('Posterior Distributions')

plt.tight_layout()
plt.show()
```

Trang trình bày 10: Phát hiện điểm thay đổi Bayes

Việc triển khai này trình bày cách phát hiện các thay đổi trong dữ liệu chuỗi thời gian bằng cách sử dụng suy luận Bayes để xác định các điểm mà phân phối cơ bản thay đổi.

```python
import numpy as np
from scipy import stats

class BayesianChangePoint:
    def __init__(self, data):
        self.data = data
        self.n = len(data)

    def compute_probability_change(self):
        probs = np.zeros(self.n)

        for t in range(1, self.n-1):
            # Split data at potential change point
            segment1 = self.data[:t]
            segment2 = self.data[t:]

            # Compute log likelihood for each segment
            ll1 = np.sum(stats.norm.logpdf(segment1,
                                         np.mean(segment1),
                                         np.std(segment1)))
            ll2 = np.sum(stats.norm.logpdf(segment2,
                                         np.mean(segment2),
                                         np.std(segment2)))

            # Total log likelihood with change point
            ll_change = ll1 + ll2

            # Log likelihood without change point
            ll_no_change = np.sum(stats.norm.logpdf(self.data,
                                                   np.mean(self.data),
                                                   np.std(self.data)))

            # Compute probability of change point
            probs[t] = 1 / (1 + np.exp(ll_no_change - ll_change))

        return probs

# Generate example data with change point
np.random.seed(42)
n_points = 200
change_point = 100
data = np.concatenate([np.random.normal(0, 1, change_point),
                      np.random.normal(3, 1, n_points-change_point)])

# Detect change points
detector = BayesianChangePoint(data)
change_probs = detector.compute_probability_change()

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(211)
plt.plot(data)
plt.axvline(x=change_point, color='r', linestyle='--',
            label='True Change Point')
plt.ylabel('Value')
plt.title('Time Series Data')
plt.legend()

plt.subplot(212)
plt.plot(change_probs)
plt.xlabel('Time')
plt.ylabel('Change Probability')
plt.title('Change Point Probability')
plt.grid(True)

plt.tight_layout()
plt.show()
```

Trang trình bày 11: Mạng thần kinh Bayes

Triển khai Mạng thần kinh Bayes bằng cách sử dụng suy luận biến phân để ước tính độ không đảm bảo trong các dự đoán. Cách tiếp cận này kết hợp học sâu với suy luận Bayes để định lượng độ không đảm bảo chắc chắn.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.zeros(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_sigma, -3)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_sigma, -3)

    def forward(self, x):
        weight = Normal(self.weight_mu, F.softplus(self.weight_sigma))
        bias = Normal(self.bias_mu, F.softplus(self.bias_sigma))

        w = weight.rsample()
        b = bias.rsample()

        return F.linear(x, w, b)

class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = BayesianLinear(input_dim, hidden_dim)
        self.l2 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x, num_samples=1):
        predictions = []
        for _ in range(num_samples):
            h = F.relu(self.l1(x))
            y = self.l2(h)
            predictions.append(y)
        return torch.stack(predictions)

# Example usage
torch.manual_seed(42)
X = torch.linspace(-3, 3, 100).reshape(-1, 1)
y = X.pow(2) + 0.2 * torch.randn_like(X)

model = BayesianNN(1, 20, 1)
predictions = model(X, num_samples=100)

mean = predictions.mean(0)
std = predictions.std(0)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
plt.plot(X, mean, 'r-', label='Mean prediction')
plt.fill_between(X.squeeze(),
                 (mean - 2*std).squeeze(),
                 (mean + 2*std).squeeze(),
                 alpha=0.2, label='95% CI')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Bayesian Neural Network Predictions')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 12: Tối ưu hóa Bayes

Việc triển khai này cho thấy cách thực hiện tối ưu hóa Bayes bằng Quy trình Gaussian để điều chỉnh siêu tham số, kết hợp tính không chắc chắn trong quy trình tối ưu hóa.

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, bounds):
        self.bounds = bounds
        self.X = []
        self.y = []

    def acquisition_function(self, X, model):
        mu, sigma = model.predict(X.reshape(-1, 1))
        best_y = np.max(self.y) if self.y else 0

        with np.errstate(divide='warn'):
            Z = (mu - best_y) / sigma
            ei = (mu - best_y) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def propose_location(self, model):
        def objective(X):
            return -self.acquisition_function(X, model)

        X_init = np.random.uniform(self.bounds[0], self.bounds[1], 10)
        best_value = float("inf")
        best_X = None

        for x0 in X_init:
            result = minimize(objective, x0=x0, bounds=[self.bounds])
            if result.fun < best_value:
                best_value = result.fun
                best_X = result.x

        return best_X

    def add_observation(self, X, y):
        self.X.append(X)
        self.y.append(y)

# Example usage
def objective_function(x):
    return -(x - 2)**2 + 10

optimizer = BayesianOptimizer(bounds=(-5, 5))

# Optimization loop
for i in range(10):
    if len(optimizer.X) == 0:
        X_next = np.random.uniform(-5, 5)
    else:
        X = np.array(optimizer.X).reshape(-1, 1)
        y = np.array(optimizer.y).reshape(-1, 1)

        # Fit GP model (simplified for example)
        X_next = optimizer.propose_location(model)

    y_next = objective_function(X_next)
    optimizer.add_observation(X_next, y_next)

# Plot results
plt.figure(figsize=(10, 6))
X_plot = np.linspace(-5, 5, 100)
y_plot = [objective_function(x) for x in X_plot]

plt.plot(X_plot, y_plot, 'b-', label='True function')
plt.scatter(optimizer.X, optimizer.y, c='r', label='Observations')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Bayesian Optimization Progress')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 13: Thử nghiệm A/B Bayesian

Triển khai thử nghiệm A/B Bayesian để so sánh hai biến thể, kết hợp kiến ​​thức trước đó và tính toán xác suất cải thiện sau.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class BayesianABTest:
    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def update_posterior(self, successes, trials):
        alpha_post = self.alpha_prior + successes
        beta_post = self.beta_prior + (trials - successes)
        return alpha_post, beta_post

    def probability_b_better_than_a(self, samples=10000):
        theta_a = np.random.beta(self.alpha_a, self.beta_a, samples)
        theta_b = np.random.beta(self.alpha_b, self.beta_b, samples)
        return np.mean(theta_b > theta_a)

    def fit(self, data_a, data_b):
        # Update posteriors for both variants
        self.alpha_a, self.beta_a = self.update_posterior(
            sum(data_a), len(data_a))
        self.alpha_b, self.beta_b = self.update_posterior(
            sum(data_b), len(data_b))

        self.prob_b_better = self.probability_b_better_than_a()

    def plot_posteriors(self):
        x = np.linspace(0, 1, 1000)
        plt.figure(figsize=(10, 6))

        # Plot posterior distributions
        plt.plot(x, stats.beta.pdf(x, self.alpha_a, self.beta_a),
                'b-', label='Variant A')
        plt.plot(x, stats.beta.pdf(x, self.alpha_b, self.beta_b),
                'r-', label='Variant B')

        plt.xlabel('Conversion Rate')
        plt.ylabel('Density')
        plt.title('Posterior Distributions of Conversion Rates')
        plt.legend()
        plt.grid(True)

        # Add probability annotation
        plt.text(0.1, 1, f'P(B > A) = {self.prob_b_better:.3f}',
                bbox=dict(facecolor='white', alpha=0.5))

        return plt

# Example usage
np.random.seed(42)

# Generate example data
n_a, n_b = 1000, 1000
p_a, p_b = 0.05, 0.07  # True conversion rates

data_a = np.random.binomial(1, p_a, n_a)
data_b = np.random.binomial(1, p_b, n_b)

# Perform Bayesian A/B test
ab_test = BayesianABTest()
ab_test.fit(data_a, data_b)

# Plot results
ab_test.plot_posteriors()
plt.show()

print(f"Summary Statistics:")
print(f"Variant A: {sum(data_a)}/{len(data_a)} = {np.mean(data_a):.3f}")
print(f"Variant B: {sum(data_b)}/{len(data_b)} = {np.mean(data_b):.3f}")
print(f"Probability B is better: {ab_test.prob_b_better:.3f}")
```

Trang trình bày 14: Phân tích chuỗi thời gian Bayes

Triển khai mô hình chuỗi thời gian cấu trúc Bayes với các thành phần xu hướng và mùa vụ để dự báo.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class BayesianTimeSeriesModel:
    def __init__(self, seasonality=7):
        self.seasonality = seasonality

    def decompose_series(self, data):
        n = len(data)

        # Extract trend using simple moving average
        window = self.seasonality
        trend = np.convolve(data, np.ones(window)/window, mode='valid')
        trend = np.pad(trend, (window//2, window//2), mode='edge')

        # Extract seasonality
        detrended = data - trend
        seasonal = np.zeros(self.seasonality)
        for i in range(self.seasonality):
            seasonal[i] = np.mean(detrended[i::self.seasonality])

        # Normalize seasonal component
        seasonal = seasonal - np.mean(seasonal)

        # Generate full seasonal component
        seasonal_full = np.tile(seasonal, n//self.seasonality + 1)[:n]

        # Calculate residuals
        residuals = data - trend - seasonal_full

        return trend, seasonal_full, residuals

    def forecast(self, data, steps_ahead=30):
        trend, seasonal, residuals = self.decompose_series(data)

        # Forecast trend using simple linear extrapolation
        x = np.arange(len(trend))
        slope, intercept = np.polyfit(x, trend, 1)
        trend_forecast = slope * (x[-1] + np.arange(1, steps_ahead + 1)) + intercept

        # Forecast seasonal component
        seasonal_forecast = np.tile(seasonal[-self.seasonality:],
                                  steps_ahead//self.seasonality + 1)[:steps_ahead]

        # Generate prediction intervals using residual distribution
        std_resid = np.std(residuals)
        lower = trend_forecast + seasonal_forecast - 1.96 * std_resid
        upper = trend_forecast + seasonal_forecast + 1.96 * std_resid

        return trend_forecast + seasonal_forecast, lower, upper

# Generate example data
np.random.seed(42)
n_points = 200
t = np.arange(n_points)
trend = 0.05 * t
seasonal = 5 * np.sin(2 * np.pi * t / 7)
noise = np.random.normal(0, 1, n_points)
data = trend + seasonal + noise

# Fit model and generate forecast
model = BayesianTimeSeriesModel(seasonality=7)
forecast, lower, upper = model.forecast(data, steps_ahead=30)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t, data, 'b-', label='Observed')
plt.plot(np.arange(n_points, n_points + 30), forecast, 'r-',
         label='Forecast')
plt.fill_between(np.arange(n_points, n_points + 30),
                 lower, upper, color='r', alpha=0.2,
                 label='95% CI')
plt.axvline(x=n_points, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Bayesian Time Series Forecast')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

* Bài viết ArXiv về Học sâu Bayesian: [https://arxiv.org/abs/2007.06823](https://arxiv.org/abs/2007.06823)
* Hướng dẫn toàn diện về lập trình xác suất: [https://arxiv.org/abs/1809.10756](https://arxiv.org/abs/1809.10756)
* Khảo sát về Tối ưu hóa Bayes: [https://arxiv.org/abs/1807.02811](https://arxiv.org/abs/1807.02811)
* Phương pháp Bayesian hiện đại trong học máy: [https://arxiv.org/abs/1505.02965](https://arxiv.org/abs/1505.02965)
* Hướng dẫn thực hành về Mạng thần kinh Bayes: [https://arxiv.org/abs/2006.11695](https://arxiv.org/abs/2006.11695)
