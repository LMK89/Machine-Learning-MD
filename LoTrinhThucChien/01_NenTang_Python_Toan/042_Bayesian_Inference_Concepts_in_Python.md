## Khái niệm suy luận Bayes trong Python
Trang trình bày 1: Tìm hiểu suy luận Bayes

Suy luận Bayes là một phương pháp thống kê sử dụng định lý Bayes để cập nhật xác suất của một giả thuyết khi có thêm bằng chứng. Nó tạo thành nền tảng để hiểu ước lượng cận biên, ước lượng hậu nghiệm tối đa và ước lượng hậu nghiệm tối đa cận biên.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define prior, likelihood, and posterior
def prior(theta):
    return 1 if 0 <= theta <= 1 else 0

def likelihood(data, theta):
    return theta**np.sum(data) * (1-theta)**(len(data) - np.sum(data))

def posterior(data, theta):
    return likelihood(data, theta) * prior(theta)

# Generate some data
data = np.random.binomial(1, 0.7, 100)

# Plot posterior distribution
theta_range = np.linspace(0, 1, 1000)
posterior_values = [posterior(data, theta) for theta in theta_range]

plt.plot(theta_range, posterior_values)
plt.title('Posterior Distribution')
plt.xlabel('θ')
plt.ylabel('Probability Density')
plt.show()
```

Slide 2: Ước tính cận biên

Ước tính cận biên liên quan đến việc tích hợp các tham số gây phiền toái để tập trung vào các tham số quan tâm. Nó rất hữu ích khi xử lý các mô hình phức tạp với nhiều tham số.

```python
import numpy as np
from scipy import integrate

def joint_distribution(x, y):
    return np.exp(-(x**2 + y**2))

def marginal_x(x):
    return integrate.quad(lambda y: joint_distribution(x, y), -np.inf, np.inf)[0]

x_range = np.linspace(-3, 3, 100)
marginal_values = [marginal_x(x) for x in x_range]

plt.plot(x_range, marginal_values)
plt.title('Marginal Distribution of X')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.show()
```

Trang trình bày 3: Ước tính tối đa A Posteriori (MAP)

Ước tính MAP tìm ra phương thức phân phối sau, cân bằng niềm tin trước đó với dữ liệu được quan sát. Nó thường được sử dụng trong học máy để ước tính tham số.

```python
import numpy as np
from scipy.optimize import minimize_scalar

def neg_log_posterior(theta, data):
    if theta <= 0 or theta >= 1:
        return np.inf
    return -np.sum(data) * np.log(theta) - (len(data) - np.sum(data)) * np.log(1 - theta)

data = np.random.binomial(1, 0.7, 1000)

result = minimize_scalar(neg_log_posterior, args=(data,), bounds=(0, 1), method='bounded')
map_estimate = result.x

print(f"MAP estimate: {map_estimate}")
```

Trang trình bày 4: Ước tính mức tối đa cận biên A Posteriori (MMAP)

MMAP kết hợp ước tính cận biên và MAP, tối đa hóa xác suất hậu nghiệm của một tập hợp con các biến trong khi tích hợp phần còn lại. Nó hữu ích trong các mô hình phân cấp.

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy import integrate

def joint_posterior(x, y, data):
    return np.exp(-(x**2 + y**2)) * np.prod(data**x * (1-data)**y)

def marginal_posterior_x(x, data):
    return integrate.quad(lambda y: joint_posterior(x, y, data), 0, 1)[0]

def neg_log_marginal_posterior(x, data):
    return -np.log(marginal_posterior_x(x, data))

data = np.random.beta(2, 3, 100)

result = minimize_scalar(neg_log_marginal_posterior, args=(data,), bounds=(0, 1), method='bounded')
mmap_estimate = result.x

print(f"MMAP estimate: {mmap_estimate}")
```

Slide 5: So sánh các phương pháp ước tính

Hãy so sánh các ước tính cận biên, MAP và MMAP bằng một ví dụ đơn giản để làm nổi bật sự khác biệt và trường hợp sử dụng của chúng.

```python
import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize_scalar

# Generate data
true_alpha, true_beta = 2, 5
data = np.random.beta(true_alpha, true_beta, 1000)

# Marginal estimation (using method of moments)
sample_mean = np.mean(data)
sample_var = np.var(data)
marginal_alpha = sample_mean * (sample_mean * (1 - sample_mean) / sample_var - 1)
marginal_beta = (1 - sample_mean) * (sample_mean * (1 - sample_mean) / sample_var - 1)

# MAP estimation
def neg_log_posterior(params):
    alpha, beta = params
    return -np.sum(beta.logpdf(data, alpha, beta))

map_result = minimize_scalar(neg_log_posterior, bounds=((0, 10), (0, 10)), method='L-BFGS-B')
map_alpha, map_beta = map_result.x

# MMAP estimation (simplified for this example)
def marginal_posterior_alpha(alpha):
    return beta.pdf(data, alpha, true_beta).prod()

mmap_result = minimize_scalar(lambda a: -np.log(marginal_posterior_alpha(a)), bounds=(0, 10), method='bounded')
mmap_alpha = mmap_result.x

print(f"True parameters: alpha={true_alpha}, beta={true_beta}")
print(f"Marginal estimate: alpha={marginal_alpha:.2f}, beta={marginal_beta:.2f}")
print(f"MAP estimate: alpha={map_alpha:.2f}, beta={map_beta:.2f}")
print(f"MMAP estimate: alpha={mmap_alpha:.2f} (beta fixed at {true_beta})")
```

Trang trình bày 6: Ví dụ thực tế: Khử nhiễu hình ảnh

Khử nhiễu ảnh là một ứng dụng phổ biến của suy luận Bayes. Chúng tôi sẽ sử dụng ước tính MAP để loại bỏ nhiễu khỏi hình ảnh.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def add_noise(image, noise_level):
    return np.clip(image + np.random.normal(0, noise_level, image.shape), 0, 1)

def prior(image, lambda_param):
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return lambda_param * np.sum(np.abs(convolve(image, kernel)))

def likelihood(noisy_image, clean_image, noise_level):
    return np.sum((noisy_image - clean_image)**2) / (2 * noise_level**2)

def map_estimate(noisy_image, lambda_param, noise_level, num_iterations):
    denoised = noisy_image.()
    for _ in range(num_iterations):
        grad_prior = lambda_param * convolve(np.sign(convolve(denoised, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))), np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
        grad_likelihood = (denoised - noisy_image) / noise_level**2
        denoised -= 0.1 * (grad_prior + grad_likelihood)
    return np.clip(denoised, 0, 1)

# Generate and denoise image
image = np.zeros((100, 100))
image[25:75, 25:75] = 1
noisy_image = add_noise(image, 0.1)
denoised_image = map_estimate(noisy_image, 0.1, 0.1, 50)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(noisy_image, cmap='gray')
ax2.set_title('Noisy Image')
ax3.imshow(denoised_image, cmap='gray')
ax3.set_title('Denoised Image (MAP)')
plt.show()
```

Slide 7: Ví dụ thực tế: Phân loại văn bản

Hãy sử dụng Naive Bayes, một bộ phân loại xác suất đơn giản dựa trên định lý Bayes để phân loại văn bản. Ví dụ này thể hiện việc sử dụng ước tính MAP trong một tình huống thực tế.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample data
texts = [
    "I love this movie", "Great film", "Awesome movie",
    "Terrible movie", "Waste of time", "Awful film",
    "Neutral opinion", "Average movie", "It was okay"
]
labels = ["positive", "positive", "positive",
          "negative", "negative", "negative",
          "neutral", "neutral", "neutral"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier (uses MAP estimation)
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Classify a new text
new_text = ["This movie was quite interesting"]
new_text_vec = vectorizer.transform(new_text)
prediction = clf.predict(new_text_vec)
print(f"Prediction for '{new_text[0]}': {prediction[0]}")
```

Slide 8: Ước tính cận biên trong thực tế

Ước lượng cận biên thường được sử dụng trong các mô hình phân cấp. Hãy chứng minh điều này bằng một ví dụ đơn giản về ước tính sự phân bố chiều cao trong dân số.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data from a hierarchical model
num_groups = 5
num_samples_per_group = 50

true_population_mean = 170
true_population_std = 5
true_group_means = np.random.normal(true_population_mean, true_population_std, num_groups)

data = []
for group_mean in true_group_means:
    group_data = np.random.normal(group_mean, 3, num_samples_per_group)
    data.extend(group_data)

# Estimate marginal distribution
estimated_mean = np.mean(data)
estimated_std = np.std(data)

# Plot results
x = np.linspace(150, 190, 1000)
true_dist = norm.pdf(x, true_population_mean, true_population_std)
estimated_dist = norm.pdf(x, estimated_mean, estimated_std)

plt.hist(data, bins=30, density=True, alpha=0.7, label='Data')
plt.plot(x, true_dist, 'r-', label='True Distribution')
plt.plot(x, estimated_dist, 'g--', label='Estimated Distribution')
plt.xlabel('Height (cm)')
plt.ylabel('Density')
plt.legend()
plt.title('Marginal Estimation of Population Height Distribution')
plt.show()

print(f"True population mean: {true_population_mean:.2f}")
print(f"Estimated population mean: {estimated_mean:.2f}")
print(f"True population std: {true_population_std:.2f}")
print(f"Estimated population std: {estimated_std:.2f}")
```

Trang trình bày 9: Ước tính MAP trong Machine Learning

Ước tính MAP được sử dụng rộng rãi trong học máy, đặc biệt đối với các mô hình hồi quy chính quy. Hãy triển khai hồi quy sườn, sử dụng chính quy L2, làm ví dụ về ước tính MAP.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
poly = PolynomialFeatures(degree=10)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit ridge regression (MAP estimation)
ridge = Ridge(alpha=0.1)
ridge.fit(X_train_poly, y_train)

# Predict
X_plot = np.linspace(0, 5, 100)[:, np.newaxis]
X_plot_poly = poly.transform(X_plot)
y_plot = ridge.predict(X_plot_poly)

# Plot results
plt.scatter(X_train, y_train, color='r', label='Training data')
plt.scatter(X_test, y_test, color='b', label='Test data')
plt.plot(X_plot, y_plot, color='g', label='Ridge regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Ridge Regression (MAP Estimation)')
plt.show()

print(f"Train score: {ridge.score(X_train_poly, y_train):.4f}")
print(f"Test score: {ridge.score(X_test_poly, y_test):.4f}")
```

Trang trình bày 10: Ước tính MMAP trong các mô hình phân cấp

Ước lượng MMAP đặc biệt hữu ích trong các mô hình phân cấp. Hãy triển khai một mô hình phân cấp đơn giản để ước tính cấp độ kỹ năng của người chơi trong trò chơi.

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Generate synthetic data
num_players = 10
num_games_per_player = 50
true_skills = np.random.normal(1500, 200, num_players)
game_results = []

for i in range(num_players):
    player_results = np.random.normal(true_skills[i], 100, num_games_per_player)
    game_results.extend(player_results)

game_results = np.array(game_results)
player_indices = np.repeat(np.arange(num_players), num_games_per_player)

# Define the model
def log_likelihood(skills, game_results, player_indices):
    return np.sum(norm.logpdf(game_results, skills[player_indices], 100))

def log_prior(skills, mu, sigma):
    return np.sum(norm.logpdf(skills, mu, sigma))

def neg_log_posterior(params, game_results, player_indices):
    skills, mu, sigma = params[:num_players], params[-2], params[-1]
    return -(log_likelihood(skills, game_results, player_indices) + log_prior(skills, mu, sigma))

# Perform MMAP estimation
initial_guess = np.concatenate([np.full(num_players, 1500), [1500, 200]])
result = minimize(neg_log_posterior, initial_guess, args=(game_results, player_indices), method='L-BFGS-B')

estimated_skills, estimated_mu, estimated_sigma = result.x[:num_players], result.x[-2], result.x[-1]

print("True skills:", true_skills)
print("Estimated skills:", estimated_skills)
print(f"Estimated population mean: {estimated_mu:.2f}")
print(f"Estimated population std: {estimated_sigma:.2f}")

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(true_skills, estimated_skills)
plt.plot([1000, 2000], [1000, 2000], 'r--')
plt.xlabel('True Skills')
plt.ylabel('Estimated Skills')
plt.title('MMAP Estimation of Player Skills')
plt.show()
```

Trang trình bày 11: So sánh các phương pháp ước tính: Một ví dụ thực tế

Hãy so sánh ước tính cận biên, MAP và MMAP bằng một ví dụ đơn giản về ước tính các tham số của phân phối chuẩn.

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Generate data
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, 1000)

# Marginal estimation (Method of Moments)
marginal_mu = np.mean(data)
marginal_sigma = np.std(data)

# MAP estimation
def neg_log_posterior(params, data):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    return -np.sum(norm.logpdf(data, mu, sigma)) - norm.logpdf(mu, 0, 10) - norm.logpdf(sigma, 1, 1)

map_result = minimize(neg_log_posterior, [0, 0], args=(data,))
map_mu, map_sigma = map_result.x[0], np.exp(map_result.x[1])

# MMAP estimation (integrating out sigma)
def neg_log_marginal_posterior(mu, data):
    n = len(data)
    s2 = np.sum((data - mu)**2) / n
    return n/2 * np.log(s2) + n/2 * np.log(2*np.pi) + 1/2 * np.log(n) + (mu**2) / (2 * 100)

mmap_result = minimize(lambda mu: neg_log_marginal_posterior(mu, data), 0)
mmap_mu = mmap_result.x[0]
mmap_sigma = np.sqrt(np.sum((data - mmap_mu)**2) / len(data))

print(f"True parameters: mu={true_mu}, sigma={true_sigma}")
print(f"Marginal estimate: mu={marginal_mu:.2f}, sigma={marginal_sigma:.2f}")
print(f"MAP estimate: mu={map_mu:.2f}, sigma={map_sigma:.2f}")
print(f"MMAP estimate: mu={mmap_mu:.2f}, sigma={mmap_sigma:.2f}")

# Plot results
x = np.linspace(0, 10, 100)
plt.hist(data, bins=30, density=True, alpha=0.7, label='Data')
plt.plot(x, norm.pdf(x, true_mu, true_sigma), 'k-', label='True')
plt.plot(x, norm.pdf(x, marginal_mu, marginal_sigma), 'r--', label='Marginal')
plt.plot(x, norm.pdf(x, map_mu, map_sigma), 'g--', label='MAP')
plt.plot(x, norm.pdf(x, mmap_mu, mmap_sigma), 'b--', label='MMAP')
plt.legend()
plt.title('Comparison of Estimation Methods')
plt.show()
```

Slide 12: Lựa chọn mô hình Bayesian

Lựa chọn mô hình Bayes sử dụng khả năng cận biên để so sánh các mô hình khác nhau. Hãy thực hiện một ví dụ đơn giản so sánh các mô hình hồi quy đa thức ở các mức độ khác nhau.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import logsumexp

# Generate data
np.random.seed(42)
X = np.linspace(0, 1, 20)
y = 2 * X**2 - X + 1 + np.random.normal(0, 0.1, 20)

# Define models
def polynomial_basis(X, degree):
    return np.column_stack([X**i for i in range(degree+1)])

def log_marginal_likelihood(X, y, degree):
    phi = polynomial_basis(X, degree)
    N, M = phi.shape
    alpha, beta = 1e-6, 1

    S_N_inv = alpha * np.eye(M) + beta * phi.T @ phi
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N @ phi.T @ y

    return -0.5 * (N * np.log(2*np.pi) - M * np.log(alpha) + N * np.log(beta) +
                   np.log(np.linalg.det(S_N_inv)) +
                   beta * (y.T @ y - m_N.T @ S_N_inv @ m_N))

# Compare models
max_degree = 5
log_evidences = [log_marginal_likelihood(X, y, d) for d in range(1, max_degree+1)]
posterior_probs = np.exp(log_evidences - logsumexp(log_evidences))

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(range(1, max_degree+1), posterior_probs, 'bo-')
plt.xlabel('Polynomial Degree')
plt.ylabel('Posterior Probability')
plt.title('Model Comparison')

plt.subplot(122)
plt.scatter(X, y, c='r', label='Data')
for d in range(1, max_degree+1):
    phi = polynomial_basis(X, d)
    w = np.linalg.inv(phi.T @ phi) @ phi.T @ y
    X_plot = np.linspace(0, 1, 100)
    y_plot = polynomial_basis(X_plot, d) @ w
    plt.plot(X_plot, y_plot, label=f'Degree {d}')
plt.legend()
plt.title('Polynomial Fits')

plt.tight_layout()
plt.show()

print("Posterior probabilities:")
for d, p in enumerate(posterior_probs, 1):
    print(f"Degree {d}: {p:.4f}")
```

Slide 13: Những cân nhắc và hạn chế thực tế

Khi áp dụng các phương pháp ước lượng biên, MAP và MMAP, hãy cân nhắc:

1. Độ phức tạp tính toán: Ước tính cận biên thường yêu cầu tích hợp, điều này có thể tốn kém về mặt tính toán đối với các bài toán nhiều chiều.
2. Lựa chọn trước: Việc lựa chọn trước có thể tác động đáng kể đến ước tính MAP và MMAP, đặc biệt với dữ liệu hạn chế.
3. Xác định sai mô hình: Các phương pháp này giả định mô hình là đúng nhưng điều này có thể không phải lúc nào cũng đúng trong thực tế.
4. Định lượng độ không chắc chắn: Ước tính điểm không nắm bắt được toàn bộ phân bố sau. Hãy cân nhắc sử dụng các phương pháp Markov Chain Monte Carlo (MCMC) để có phân tích Bayes hoàn chỉnh hơn.

```python
# Pseudocode for MCMC (Metropolis-Hastings algorithm)
def metropolis_hastings(log_posterior, initial_state, num_samples):
    current_state = initial_state
    samples = []

    for _ in range(num_samples):
        proposed_state = propose_new_state(current_state)
        acceptance_ratio = exp(log_posterior(proposed_state) - log_posterior(current_state))

        if random.uniform(0, 1) < acceptance_ratio:
            current_state = proposed_state

        samples.append(current_state)

    return samples

# Usage
posterior_samples = metropolis_hastings(log_posterior_function, initial_guess, 10000)
```

Trang trình bày 14: Tài nguyên bổ sung

Để khám phá thêm về các phương pháp ước tính và suy luận Bayes, hãy xem xét các tài nguyên sau:

1. "Phân tích dữ liệu Bayes" của Andrew Gelman và cộng sự. (2013) ArXiv: [https://arxiv.org/abs/2011.01808](https://arxiv.org/abs/2011.01808)
2. "Nhận dạng mẫu và học máy" của Christopher Bishop (2006) ArXiv: [https://arxiv.org/abs/1011.0175](https://arxiv.org/abs/1011.0175)
3. "Học máy: Quan điểm xác suất" của Kevin Murphy (2012) ArXiv: [https://arxiv.org/abs/1504.04623](https://arxiv.org/abs/1504.04623)
4. "Mô hình đồ họa xác suất: Nguyên tắc và kỹ thuật" của Daphne Koller và Nir Friedman (2009) ArXiv: [https://arxiv.org/abs/1302.6808](https://arxiv.org/abs/1302.6808)

Các tài nguyên này cung cấp thông tin chuyên sâu về các chủ đề được thảo luận trong bài trình bày này và cung cấp các khái niệm nâng cao hơn về suy luận Bayes và học máy.
