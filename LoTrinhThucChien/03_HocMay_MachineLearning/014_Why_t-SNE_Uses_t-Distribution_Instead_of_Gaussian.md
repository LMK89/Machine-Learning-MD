## Tại sao t-SNE sử dụng phân phối thay vì Gaussian

Slide 1: Giới thiệu về t-SNE

t-SNE (T-distributed Stochastic Neighbor Embedding) là một kỹ thuật giảm kích thước phổ biến được sử dụng để hiển thị dữ liệu nhiều chiều. Đây là một cải tiến so với thuật toán SNE ban đầu, với điểm khác biệt chính là công việc sử dụng phân phối thay vì phân phối Gaussian. Thay đổi này sẽ giải quyết một số chế độ hạn chế của SNE và cung cấp kết quả hiển thị tốt hơn.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Generate sample high-dimensional data
np.random.seed(42)
data = np.random.randn(1000, 50)  # 1000 samples, 50 dimensions

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data)

# Visualize the result
plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.show()
```

Trang trình bày 2: Tìm hiểu về SNE (Nhúng hàng ngẫu nhiên)

SNE, tiền thân của t-SNE, sử dụng Gaussian phân bố để mô hình hóa sự giống nhau giữa các điểm trong cả không gian chiều cao và chiều thấp. Nó nhắm đến mục tiêu bảo vệ vùng lân cận của dữ liệu cấu trúc an toàn khi giảm kích thước.

```python
    return np.exp(-np.sum((x - y)**2) / (2 * sigma**2))

# Example: Calculate Gaussian similarity between two points
point1 = np.array([1, 2, 3])
point2 = np.array([2, 3, 4])
similarity = gaussian_similarity(point1, point2)
print(f"Gaussian similarity: {similarity}")
```

Slide 3: Các chế độ của SNE

SNE phải đối mặt với một vấn đề được gọi là "vấn đề đông đúc". Trong không gian nhiều chiều, có thể tích hình cầu tăng theo cấp số nhân với bán kính của nó, điều này có thể dẫn đến hầu hết các điểm đều đều nhau. Khi được chiếu ở các kích thước nhỏ hơn, điều này có thể dẫn đến các tập trung ở trung tâm hình ảnh.

```python

# Generate high-dimensional data
n_points = 1000
n_dims = [2, 10, 50, 100]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, dim in enumerate(n_dims):
    data = np.random.randn(n_points, dim)
    distances = np.linalg.norm(data[:100] - data[0], axis=1)

    sns.histplot(distances, kde=True, ax=axes[i])
    axes[i].set_title(f'{dim} dimensions')
    axes[i].set_xlabel('Distance from first point')

plt.tight_layout()
plt.show()
```

Slide 4: Giới thiệu về t-Distribution

Phân phối t, còn được gọi là phân phối của Sinh viên, là phân phối xác thực phát sinh khi ước tính giá trị trung bình của một tổng thể có chuẩn phân phối trong các vấn đề có kích thước nhỏ và độ lệch của tổng thể không được xác định.

```python

# Generate t-distribution
df = 1  # Degrees of freedom
x = np.linspace(-10, 10, 1000)
y = stats.t.pdf(x, df)

# Plot t-distribution
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f't-distribution (df={df})')
plt.plot(x, stats.norm.pdf(x), label='Normal distribution')
plt.title('t-distribution vs Normal distribution')
plt.legend()
plt.show()
```

Trang trình bày 5: Tại sao t-SNE sử dụng phân phối t

t-SNE thay thế phân phối Gaussian trong phân phối không chiều sâu. Phân phối t có dấu thăng hơn so với phân bố Gaussian, giúp giảm bớt vấn đề đông đúc. Nó cho phép các điểm ở khoảng cách vừa phải trong không có chiều cao được mô hình hóa bằng khoảng cách lớn hơn trong không có chiều rộng.

```python
    return (1 + np.sum((x - y)**2) / df) ** (-(df + 1) / 2)

# Compare Gaussian and t-distribution similarities
distances = np.linspace(0, 5, 100)
gaussian_sim = [gaussian_similarity(np.array([0]), np.array([d])) for d in distances]
t_sim = [t_similarity(np.array([0]), np.array([d])) for d in distances]

plt.figure(figsize=(10, 6))
plt.plot(distances, gaussian_sim, label='Gaussian')
plt.plot(distances, t_sim, label='t-distribution')
plt.title('Similarity vs Distance: Gaussian vs t-distribution')
plt.legend()
plt.xlabel('Distance')
plt.ylabel('Similarity')
plt.show()
```

Slide 6: Công thức toán học của t-SNE

t-SNE xác định các điểm giống nhau của dữ liệu $x\_j$ với $x\_i$ bằng cách sử dụng phân phối Gaussian trong không gian nhiều chiều:

$p\_{j|i} = \\frac{\\exp(-||x\_i - x\_j||^2 / 2\\sigma\_i^2)}{\\sum\_{k \\neq i} \\exp(-||x\_i - x\_k||^2 / 2\\sigma\_i^2)}$

Trong không gian ít chiều, nó sử dụng phân phối theo một cấp độ:

$q\_{ij} = \\frac{(1 + ||y\_i - y\_j||^2)^{-1}}{\\sum\_{k \\neq l} (1 + ||y\_k - y\_l||^2)^{-1}}$

```python
    diff = X[i] - X[j]
    return np.exp(-np.dot(diff, diff) / (2 * sigma**2))

def low_dim_similarity(Y, i, j):
    diff = Y[i] - Y[j]
    return 1 / (1 + np.dot(diff, diff))

# Example usage
X = np.random.randn(100, 50)  # High-dimensional data
Y = np.random.randn(100, 2)   # Low-dimensional embedding
i, j = 0, 1
sigma = 1.0

p_ij = high_dim_similarity(X, i, j, sigma)
q_ij = low_dim_similarity(Y, i, j)

print(f"High-dimensional similarity: {p_ij}")
print(f"Low-dimensional similarity: {q_ij}")
```

Slide 7: Ưu điểm của phân phối trong t-SNE

Các dấu sau của phân bố t được phép thực hiện khoảng cách trung thực hơn giữa các điểm ở khoảng cách vừa phải trong không gian nhiều chiều. Điều này giúp phân tích các cụm kết quả hơn và giảm xu hướng các tập trung ở trung tâm hình ảnh.

```python

# Generate sample data with clusters
np.random.seed(42)
n_samples = 300
X = np.concatenate([
    np.random.randn(n_samples, 50) + np.array([2] * 50),
    np.random.randn(n_samples, 50) + np.array([-2] * 50),
    np.random.randn(n_samples, 50)
])

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=np.repeat(['A', 'B', 'C'], n_samples))
plt.title('t-SNE Visualization of Clustered Data')
plt.show()
```

Trang trình bày 8: Độ dốc tính toán trong t-SNE

Độ dốc của phân kỳ Kullback-Leibler giữa các phân phối P và Q cung cấp quá trình ưu tiên tối đa trong t-SNE. Công việc sử dụng phân phối đơn giản hóa công việc tính toán độ dốc này:

$\\frac{\\partial C}{\\partial y\_i} = 4 \\sum\_j (p\_{ij} - q\_{ij})(y\_i - y\_j)(1 + ||y\_i - y\_j||^2)^{-1}$

```python
    n = Y.shape[0]
    dY = np.zeros_like(Y)

    for i in range(n):
        diff = Y[i] - Y
        dist_sq = np.sum(diff**2, axis=1)
        q = (1 + dist_sq)**-1
        q[i] = 0

        dY[i] = 4 * np.sum((P[i] - Q[i])[:, np.newaxis] * diff * q[:, np.newaxis], axis=0)

    return dY

# Example usage (simplified)
n, d = 100, 2
Y = np.random.randn(n, d)
P = np.random.rand(n, n)
Q = np.random.rand(n, n)

gradient = tsne_gradient(Y, P, Q)
print("Gradient shape:", gradient.shape)
```

Trang trình bày 9: Sự rối loạn trong t-SNE

Sự hỗn loạn là một siêu tham số trong t-SNE giúp cân bằng sự chú ý giữa các cạnh cục bộ và toàn cầu của dữ liệu. Nó liên quan đến số lượng hàng xóm gần nhất mà mỗi điểm xem xét một cách hiệu quả. Phức tạp giá trị thường nằm trong khoảng từ 5 đến 50.

```python

def compute_perplexity(distances, sigmas):
    P = np.exp(-distances / (2 * sigmas**2))
    sumP = np.sum(P, axis=1)
    H = np.log2(sumP) + np.sum(P * np.log2(P), axis=1) / sumP
    return 2**H

# Generate sample data
X = np.random.randn(500, 50)

# Compute distances
nbrs = NearestNeighbors(n_neighbors=50, metric='euclidean').fit(X)
distances, _ = nbrs.kneighbors(X)

# Compute perplexity for different sigma values
sigmas = np.logspace(-1, 1, 20)
perplexities = [np.mean(compute_perplexity(distances, sigma)) for sigma in sigmas]

plt.figure(figsize=(10, 6))
plt.semilogx(sigmas, perplexities)
plt.title('Average Perplexity vs Sigma')
plt.xlabel('Sigma')
plt.ylabel('Perplexity')
plt.show()
```

Trang trình bày 10: Cường độ sớm trong t-SNE

Phóng đại đại sớm là một kỹ thuật được sử dụng trong t-SNE để tạo ra cấu trúc tổng thể tốt hơn. Nó liên quan đến việc nhân xác thực nhiều chiều của các lần đầu tiên với một hệ số (thường là 4-12) để khuyến khích thành các cụm phân tán rộng rãi.

```python
    P_exaggerated = P.copy()
    for i in range(n_iter):
        P_exaggerated *= exaggeration_factor
        # Perform t-SNE iteration here
        # ...
    return P_exaggerated

# Example usage
P = np.random.rand(100, 100)
P_exaggerated = early_exaggeration(P)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(P, cmap='viridis')
plt.title('Original P')
plt.subplot(122)
plt.imshow(P_exaggerated, cmap='viridis')
plt.title('Exaggerated P')
plt.tight_layout()
plt.show()
```

Slide 11: Ví dụ thực tế: Nhận dạng chữ viết tay

t-SNE thường được sử dụng để trực tiếp hóa các dữ liệu nhiều chiều, hạn chế như hình ảnh. Vui lòng áp dụng t-SNE cho MNIST data file bao gồm các chữ số viết tay.

```python
from sklearn.manifold import TSNE
import seaborn as sns

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='deep')
plt.title('t-SNE Visualization of MNIST Digits')
plt.legend(title='Digit')
plt.show()
```

Slide 12: Ví dụ thực tế: Phân tích biểu hiện gen

t-SNE được sử dụng rộng rãi trong sinh học để hiển thị dữ liệu biểu hiện. Đây là một ví dụ đơn giản sử dụng tổng hợp dữ liệu hiện tại.

```python

# Generate synthetic gene expression data
n_samples = 1000
n_genes = 50
n_conditions = 3

data = np.random.randn(n_samples, n_genes)
conditions = np.random.choice(['Control', 'Treatment A', 'Treatment B'], n_samples)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data)

# Visualize the result
plt.figure(figsize=(12, 8))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=conditions, palette='deep')
plt.title('t-SNE Visualization of Gene Expression Data')
plt.legend(title='Condition')
plt.show()
```

Trang trình bày 13: Giới hạn và cân nhắc của t-SNE

Mặc dù t-SNE mạnh mẽ nhưng nó cũng có những chế độ hạn chế. Nó có thể tiết kiệm chi phí cho các tính toán đối với các dữ liệu lớn, có thể tạo ra các kết quả khác nhau trong nhiều lần chạy tính chất ngẫu nhiên của nó và đôi khi có thể tạo ra các hình ảnh trực tiếp sai lệch nếu không được sử dụng cẩn thận.

```python

def compare_tsne_runtime(n_samples_list, n_features=50):
    runtimes = []
    for n_samples in n_samples_list:
        data = np.random.randn(n_samples, n_features)
        start_time = time.time()
        TSNE(n_components=2).fit_transform(data)
        end_time = time.time()
        runtimes.append(end_time - start_time)
    return runtimes

n_samples_list = [100, 500, 1000, 5000]
runtimes = compare_tsne_runtime(n_samples_list)

plt.figure(figsize=(10, 6))
plt.plot(n_samples_list, runtimes, marker='o')
plt.title('t-SNE Runtime vs Dataset Size')
plt.xlabel('Number of Samples')
plt.ylabel('Runtime (seconds)')
plt.show()
```

Trang trình bày 14: Kết luận và các phương pháp hay nhất

Việc t-SNE sử dụng phân phối thay vì phân phối Gaussian trong không gian chiều thấp giải quyết vấn đề đông đúc và cung cấp hình ảnh trực quan tốt hơn về dữ liệu chiều cao. Khi sử dụng t-SNE, hãy cân nhắc thử nghiệm các phức tạp giá trị khác nhau, chạy nhiều lần để đảm bảo tính ổn định và cẩn thận trong quá trình giải mã khoảng cách giữa các cụm được phân tách rõ ràng.

```python
    results = {}
    for perplexity in perplexities:
        results[perplexity] = []
        for _ in range(n_runs):
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=None)
            result = tsne.fit_transform(X)
            results[perplexity].append(result)
    return results

# Example usage
X = np.random.randn(500, 50)
best_practices_results = tsne_best_practices(X)

# Visualize results for different perplexities
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, (perplexity, runs) in enumerate(best_practices_results.items()):
    for run in runs:
        axes[idx].scatter(run[:, 0], run[:, 1], alpha=0.5)
    axes[idx].set_title(f'Perplexity: {perplexity}')
plt.tight_layout()
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về t-SNE và các ứng dụng của nó, đây là một số tài nguyên có giá trị:

1. Bài viết gốc về t-SNE: "Trực quan hóa dữ liệu bằng t-SNE" của Laurens van der Maaten và Geoffrey Hinton (2008) ArXiv URL: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
2. "Cách sử dụng hiệu quả t-SNE" của Martin Wattenberg, Fernanda Viégas và Ian Johnson Có tại: [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
3. "Tăng tốc t-SNE bằng thuật toán dựa trên cây" của Laurens van der Maaten (2014) URL ArXiv: [https://arxiv.org/abs/1301.3342](https://arxiv.org/abs/1301.3342)

Tài nguyên này cung cấp giải pháp sâu sắc về lý thuyết, chi tiết phát triển khai và phương pháp thực hành tốt nhất của t-SNE để sử dụng hiệu quả trong các vấn đề phân tích dữ liệu khác nhau.
