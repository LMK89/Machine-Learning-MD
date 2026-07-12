## Khả năng cận biên và các yếu tố Bayes để so sánh mô hình Bayes

Trang trình bày 1: Nguyên tắc cơ bản về khả năng cận biên

Khả năng cận biên thể hiện xác suất quan sát dữ liệu theo một mô hình cụ thể bằng cách tích hợp trên tất cả các giá trị tham số có thể có. Khái niệm cơ bản này tạo thành xương sống của các phương pháp so sánh và lựa chọn mô hình Bayes trong suy luận thống kê.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def marginal_likelihood_normal(data, mu_prior, sigma_prior, sigma_likelihood):
    n = len(data)
    sample_mean = np.mean(data)

    # Calculate posterior parameters
    sigma_posterior = 1 / (1/sigma_prior**2 + n/sigma_likelihood**2)
    mu_posterior = sigma_posterior * (mu_prior/sigma_prior**2 +
                                    n*sample_mean/sigma_likelihood**2)

    # Calculate marginal likelihood
    ml = stats.norm.pdf(data, loc=mu_prior, scale=np.sqrt(sigma_prior**2 +
                                                         sigma_likelihood**2))
    return np.prod(ml)

# Example usage
data = np.random.normal(2, 1, 100)
ml = marginal_likelihood_normal(data, mu_prior=0, sigma_prior=2, sigma_likelihood=1)
print(f"Marginal Likelihood: {ml:.10f}")
```

Trang trình bày 2: Triển khai yếu tố Bayes

Các yếu tố Bayes cung cấp thước đo định lượng để so sánh hai mô hình cạnh tranh bằng cách lấy tỷ lệ khả năng cận biên tương ứng của chúng, đưa ra cách tiếp cận Bayes tự nhiên để kiểm tra giả thuyết và lựa chọn mô hình.

```python
def bayes_factor(data, model1_params, model2_params):
    # Calculate marginal likelihoods for both models
    ml1 = marginal_likelihood_normal(data, **model1_params)
    ml2 = marginal_likelihood_normal(data, **model2_params)

    # Computing Bayes Factor
    bf = ml1 / ml2

    # Interpret Bayes Factor
    if bf > 100:
        interpretation = "Decisive evidence for Model 1"
    elif bf > 10:
        interpretation = "Strong evidence for Model 1"
    elif bf > 3.2:
        interpretation = "Substantial evidence for Model 1"
    elif bf > 1:
        interpretation = "Weak evidence for Model 1"
    else:
        interpretation = f"Evidence supports Model 2 (BF = {1/bf:.2f})"

    return bf, interpretation

# Example usage
model1 = {'mu_prior': 2, 'sigma_prior': 1, 'sigma_likelihood': 1}
model2 = {'mu_prior': 0, 'sigma_prior': 1, 'sigma_likelihood': 1}

bf, interp = bayes_factor(data, model1, model2)
print(f"Bayes Factor: {bf:.2f}")
print(f"Interpretation: {interp}")
```

Trang trình bày 3: Triển khai phân phối trước

Phân phối trước gói gọn niềm tin của chúng tôi về các giá trị tham số trước khi quan sát dữ liệu. Việc triển khai này trình bày cách tạo và trực quan hóa các bản phân phối trước đó khác nhau để so sánh mô hình Bayes.

```python
def create_prior_distribution(prior_type, params, n_samples=10000):
    if prior_type == 'normal':
        samples = np.random.normal(params['mu'], params['sigma'], n_samples)
    elif prior_type == 'uniform':
        samples = np.random.uniform(params['low'], params['high'], n_samples)
    elif prior_type == 'beta':
        samples = np.random.beta(params['a'], params['b'], n_samples)

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.7)
    plt.title(f'{prior_type.capitalize()} Prior Distribution')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.show()

    return samples

# Example usage
normal_params = {'mu': 0, 'sigma': 1}
uniform_params = {'low': -3, 'high': 3}
beta_params = {'a': 2, 'b': 5}

normal_samples = create_prior_distribution('normal', normal_params)
uniform_samples = create_prior_distribution('uniform', uniform_params)
beta_samples = create_prior_distribution('beta', beta_params)
```

Slide 4: Tính toán bằng chứng mô hình

Việc tính toán bằng chứng mô hình bao gồm việc tích hợp hàm khả năng trên tất cả các giá trị tham số có thể có được tính theo phân bố trước đó. Việc triển khai này sử dụng tích hợp số để tính toán bằng chứng mô hình.

```python
def compute_model_evidence(data, prior_samples, likelihood_func):
    n_samples = len(prior_samples)
    evidences = np.zeros(n_samples)

    for i, theta in enumerate(prior_samples):
        # Calculate likelihood for each parameter value
        likelihood = likelihood_func(data, theta)
        evidences[i] = likelihood

    # Monte Carlo integration
    model_evidence = np.mean(evidences)

    return model_evidence

def gaussian_likelihood(data, theta):
    return np.prod(stats.norm.pdf(data, loc=theta, scale=1))

# Example usage
data = np.random.normal(2, 1, 100)
prior_samples = np.random.normal(0, 2, 1000)

evidence = compute_model_evidence(data, prior_samples, gaussian_likelihood)
print(f"Model Evidence: {evidence:.10f}")
```

Trang trình bày 5: Khung so sánh Bayesian đa mô hình

Một khuôn khổ toàn diện để so sánh nhiều mô hình thống kê sử dụng suy luận Bayes yêu cầu tính toán tỷ lệ bằng chứng và xác suất hậu nghiệm trên tất cả các kết hợp mô hình trong khi tính toán độ phức tạp và độ phù hợp của mô hình.

```python
import numpy as np
from scipy import stats

class BayesianModelComparison:
    def __init__(self, models, data):
        self.models = models
        self.data = data
        self.n_models = len(models)
        self.evidences = np.zeros(self.n_models)
        self.bayes_factors = np.zeros((self.n_models, self.n_models))

    def compute_evidence(self, model_idx):
        model = self.models[model_idx]
        likelihood = stats.norm.pdf(self.data, loc=model['mean'],
                                  scale=model['std']).prod()
        prior = stats.norm.pdf(model['mean'], loc=0, scale=1)
        return likelihood * prior

    def compute_bayes_factors(self):
        for i in range(self.n_models):
            self.evidences[i] = self.compute_evidence(i)

        for i in range(self.n_models):
            for j in range(self.n_models):
                self.bayes_factors[i,j] = self.evidences[i] / self.evidences[j]
        return self.bayes_factors

# Example usage
data = np.random.normal(2, 1, 100)
models = [
    {'mean': 0, 'std': 1},
    {'mean': 2, 'std': 1},
    {'mean': -1, 'std': 2}
]

comparison = BayesianModelComparison(models, data)
bf_matrix = comparison.compute_bayes_factors()
print("Bayes Factors Matrix:\n", bf_matrix)
```

Trang trình bày 6: Ước tính khả năng cận biên

Triển khai các phương pháp tích hợp Monte Carlo để ước tính khả năng cận biên khi các giải pháp phân tích khó thực hiện, sử dụng lấy mẫu tầm quan trọng để cải thiện độ chính xác của ước tính.

```python
def estimate_marginal_likelihood(data, n_samples=10000):
    # Parameter space sampling
    theta_samples = np.random.normal(0, 2, n_samples)

    # Likelihood calculation
    likelihoods = np.zeros(n_samples)
    for i, theta in enumerate(theta_samples):
        likelihoods[i] = np.sum(stats.norm.logpdf(data, theta, 1))

    # Log-sum-exp trick for numerical stability
    max_likelihood = np.max(likelihoods)
    marginal = np.log(np.mean(np.exp(likelihoods - max_likelihood))) + max_likelihood

    return np.exp(marginal)

# Example usage
data = np.random.normal(1.5, 1, 50)
ml_estimate = estimate_marginal_likelihood(data)
print(f"Estimated Marginal Likelihood: {ml_estimate:.6f}")
```

Slide 7: Tích hợp số cho bằng chứng

Các kỹ thuật tích hợp số nâng cao để tính toán bằng chứng mô hình bằng phương pháp cầu phương thích ứng, cung cấp các ước tính chính xác hơn cho các phân bố sau phức tạp.

```python
def adaptive_quadrature_evidence(data, bounds, n_points=100):
    # Grid points for integration
    theta_grid = np.linspace(bounds[0], bounds[1], n_points)

    # Calculate posterior at each point
    def integrand(theta):
        likelihood = np.prod(stats.norm.pdf(data, theta, 1))
        prior = stats.norm.pdf(theta, 0, 2)
        return likelihood * prior

    # Composite Simpson's rule
    posterior_values = np.array([integrand(theta) for theta in theta_grid])
    h = (bounds[1] - bounds[0]) / (n_points - 1)

    evidence = h/3 * (posterior_values[0] + posterior_values[-1] +
                      4*np.sum(posterior_values[1:-1:2]) +
                      2*np.sum(posterior_values[2:-1:2]))

    return evidence

# Example usage
data = np.random.normal(0.5, 1, 30)
bounds = [-5, 5]
evidence = adaptive_quadrature_evidence(data, bounds)
print(f"Model Evidence: {evidence:.8f}")
```

Slide 8: Triển khai thang đo của Jeffreys

Triển khai thực tế thang đo của Jeffreys để giải thích các yếu tố Bayes, bao gồm định lượng độ không đảm bảo và trực quan hóa sức mạnh bằng chứng.

```python
def interpret_bayes_factor(bf, uncertainty=0.1):
    # Add random noise to simulate uncertainty
    bf_with_uncertainty = bf * (1 + np.random.normal(0, uncertainty))

    interpretation = {
        'strength': '',
        'support': 0,
        'uncertainty': uncertainty * bf
    }

    if bf_with_uncertainty >= 100:
        interpretation['strength'] = 'Decisive'
        interpretation['support'] = 4
    elif bf_with_uncertainty >= 10:
        interpretation['strength'] = 'Strong'
        interpretation['support'] = 3
    elif bf_with_uncertainty >= 3.2:
        interpretation['strength'] = 'Substantial'
        interpretation['support'] = 2
    elif bf_with_uncertainty >= 1:
        interpretation['strength'] = 'Weak'
        interpretation['support'] = 1
    else:
        interpretation['strength'] = 'Negative'
        interpretation['support'] = 0

    return interpretation

# Example usage
test_bfs = [1.5, 5.0, 15.0, 150.0]
for bf in test_bfs:
    result = interpret_bayes_factor(bf)
    print(f"BF = {bf:.1f}: {result['strength']} evidence "
          f"(support level: {result['support']})")
```

Trang trình bày 9: Phân tích độ nhạy trước đó

Triển khai phân tích độ nhạy để đánh giá mức độ ảnh hưởng của các phân phối trước khác nhau đến khả năng biên và tính toán hệ số Bayes.

```python
def sensitivity_analysis(data, prior_params_range):
    results = []
    for prior_std in prior_params_range:
        # Calculate marginal likelihood with different priors
        prior_samples = np.random.normal(0, prior_std, 1000)
        evidence = compute_model_evidence(data, prior_samples, gaussian_likelihood)

        # Store results
        results.append({
            'prior_std': prior_std,
            'evidence': evidence,
            'log_evidence': np.log(evidence)
        })

    return results

# Example usage
data = np.random.normal(1, 1, 50)
prior_stds = np.linspace(0.1, 5, 20)
sensitivity_results = sensitivity_analysis(data, prior_stds)

for result in sensitivity_results[:5]:  # Show first 5 results
    print(f"Prior std: {result['prior_std']:.2f}, "
          f"Log Evidence: {result['log_evidence']:.4f}")
```

Trang trình bày 10: Tính trung bình của mô hình Bayes

Triển khai Tính trung bình mô hình Bayes (BMA) để kết hợp các dự đoán từ nhiều mô hình được tính theo xác suất sau của chúng.

```python
def bayesian_model_averaging(models, data, new_x):
    # Calculate model weights (posterior probabilities)
    evidences = np.array([compute_model_evidence(data, m['params'],
                         m['likelihood']) for m in models])
    weights = evidences / np.sum(evidences)

    # Make predictions
    predictions = np.zeros_like(new_x)
    for i, model in enumerate(models):
        pred = model['predict'](new_x, model['params'])
        predictions += weights[i] * pred

    return predictions, weights

# Example prediction function
def predict_linear(x, params):
    return params[0] + params[1] * x

# Example usage
x_new = np.linspace(-5, 5, 100)
models = [
    {'params': [0, 1], 'likelihood': gaussian_likelihood,
     'predict': predict_linear},
    {'params': [1, 2], 'likelihood': gaussian_likelihood,
     'predict': predict_linear}
]

predictions, model_weights = bayesian_model_averaging(models, data, x_new)
```

Trang trình bày 11: Xác thực chéo để so sánh mô hình

Triển khai các yếu tố Bayes được xác thực chéo để cung cấp khả năng so sánh mô hình mạnh mẽ hơn khi xử lý dữ liệu hạn chế.

```python
def cross_validated_bayes_factors(data, models, k_folds=5):
    n_samples = len(data)
    fold_size = n_samples // k_folds
    cv_evidences = np.zeros((len(models), k_folds))

    for fold in range(k_folds):
        # Split data
        test_idx = slice(fold*fold_size, (fold+1)*fold_size)
        train_idx = list(set(range(n_samples)) - set(range(*test_idx.indices(n_samples))))

        train_data = data[train_idx]
        test_data = data[test_idx]

        # Calculate evidence for each model
        for i, model in enumerate(models):
            cv_evidences[i, fold] = compute_model_evidence(test_data,
                                   model['prior_samples'], model['likelihood_func'])

    # Average across folds
    mean_evidences = np.mean(cv_evidences, axis=1)
    cv_bayes_factors = mean_evidences[:, None] / mean_evidences

    return cv_bayes_factors

# Example usage
cv_bf = cross_validated_bayes_factors(data, models)
print("Cross-validated Bayes Factors:\n", cv_bf)
```

Trang trình bày 12: Trực quan hóa bằng chứng mẫu

Triển khai các công cụ trực quan để so sánh bằng chứng mô hình và các yếu tố Bayes giữa các mô hình và thông số khác nhau.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_model_comparison(models, evidences, bayes_factors):
    plt.figure(figsize=(12, 5))

    # Plot 1: Model Evidences
    plt.subplot(1, 2, 1)
    plt.bar(range(len(models)), evidences)
    plt.title('Model Evidences')
    plt.xlabel('Model')
    plt.ylabel('Log Evidence')

    # Plot 2: Bayes Factors Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(np.log(bayes_factors), annot=True, cmap='RdYlBu')
    plt.title('Log Bayes Factors')
    plt.xlabel('Model j')
    plt.ylabel('Model i')

    plt.tight_layout()
    plt.show()

# Example usage
n_models = 3
evidences = np.random.uniform(1, 10, n_models)
bayes_factors = evidences[:, None] / evidences
visualize_model_comparison(range(n_models), np.log(evidences), bayes_factors)
```

Trang trình bày 13: Tài nguyên bổ sung

1. [https://arxiv.org/abs/1503.08755](https://arxiv.org/abs/1503.08755) - "Tính toán các yếu tố Bayes bằng cách khái quát hóa Tỷ lệ mật độ Savage-Dickey"
2. [https://arxiv.org/abs/1101.0955](https://arxiv.org/abs/1101.0955) - "Lựa chọn mô hình Bayes và lấy trung bình mô hình"
3. [https://arxiv.org/abs/1911.11876](https://arxiv.org/abs/1911.11876) - "Hướng dẫn lấy mẫu cầu"
4. [https://arxiv.org/abs/1804.03610](https://arxiv.org/abs/1804.03610) - "Đánh giá mô hình Bayes thực tế bằng cách sử dụng xác thực chéo một lần"
5. [https://arxiv.org/abs/1601.00850](https://arxiv.org/abs/1601.00850) - "Tính toán các yếu tố Bayes để đưa ra quyết định dựa trên bằng chứng"
