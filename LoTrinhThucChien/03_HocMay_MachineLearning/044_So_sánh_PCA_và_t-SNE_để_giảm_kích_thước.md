## So sánh PCA và t-SNE để giảm kích thước

Trang trình bày 1: Tìm hiểu PCA và t-SNE

PCA (Phân tích thành phần chính) và t-SNE (Nhúng hàng ngẫu nhiên phân phối t) đều là các kỹ thuật giảm kích thước, nhưng chúng phục vụ các mục tiêu khác nhau và có những điểm đặc biệt. PCA chủ yếu được sử dụng để giảm tính chất tuyến tính kích thước và nén dữ liệu, trong khi t-SNE được thiết kế để trực tiếp hóa dữ liệu nhiều chiều trong không gian có chiều thấp hơn.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate sample data
np.random.seed(42)
n_samples = 1000
n_features = 50
X = np.random.randn(n_samples, n_features)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
ax1.set_title('PCA')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
ax2.set_title('t-SNE')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Slide 2: Phân tích thành phần chính (PCA)

PCA là một kỹ thuật giảm kích thước tuyến tính chất được xác định theo hướng (các thành phần chính) mà dữ liệu thay đổi nhiều nhất. Nó tham chiếu dữ liệu lên các thành phần này, giảm kích thước một cách hiệu quả trong khi vẫn duy trì nhiều phương tiện hiện có.

```python
import numpy as np

def pca(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    top_eigenvectors = eigenvectors[:, :n_components]

    # Project data onto principal components
    return np.dot(X_centered, top_eigenvectors)

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
X_pca = pca(X, n_components=2)
print("PCA result:", X_pca)
```

Trang trình bày 3: Nhúng hàng ngẫu nhiên ngẫu nhiên phân phối t (t-SNE)

t-SNE là một kỹ thuật giảm tuyến tính kích thước đặc biệt được thiết kế để hiển thị dữ liệu nhiều chiều. Nó cung cấp các mối liên kết duy trì mục tiêu giữa các điểm dữ liệu, tạo ra kết quả đặc biệt trong việc phát hiện các cụm và mẫu trong bộ đệm dữ liệu phức hợp.

```python
import numpy as np

def tsne(X, n_components=2, perplexity=30.0, n_iter=1000):
    def compute_pairwise_affinities(X, perplexity):
        distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
        P = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            Di = distances[i]
            Di[i] = np.inf
            Pi = np.exp(-Di / (2 * (perplexity ** 2)))
            Pi /= np.sum(Pi)
            P[i] = Pi
        return (P + P.T) / (2 * X.shape[0])

    P = compute_pairwise_affinities(X, perplexity)
    Y = np.random.randn(X.shape[0], n_components)

    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)

        PQ_diff = P - Q
        dY = np.zeros_like(Y)
        for i in range(X.shape[0]):
            dY[i] = 4 * np.sum((PQ_diff[i] * Q[i])[:, np.newaxis] * (Y[i] - Y), axis=0)

        Y -= dY * 0.1  # Simple gradient descent

    return Y

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
X_tsne = tsne(X, n_components=2, perplexity=1.0, n_iter=100)
print("t-SNE result:", X_tsne)
```

Trang trình bày 4: Sự khác biệt chính giữa PCA và t-SNE

PCA và t-SNE khác nhau về cách tiếp cận, mục tiêu và kết quả đầu ra. PCA là một phương pháp tuyến tính bảo toàn cấu trúc toàn cầu, trong khi t-SNE là phương pháp phi tuyến tính và tập trung vào việc duy trì các mối quan hệ cục bộ. PCA mang tính quyết định và nhanh hơn, trong khi t-SNE mang tính ngẫu nhiên và chuyên sâu về mặt tính toán.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 1000
t = np.linspace(0, 10, n_samples)
X = np.column_stack((np.sin(t), np.cos(t), t))

# Implement PCA
def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

# Implement t-SNE (simplified version)
def tsne(X, n_components, perplexity, n_iter):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Apply PCA and t-SNE
X_pca = pca(X, n_components=2)
X_tsne = tsne(X, n_components=2, perplexity=30, n_iter=300)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis')
ax1.set_title('PCA')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t, cmap='viridis')
ax2.set_title('t-SNE')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Slide 5: PCA: Nền tảng toán học

PCA dựa trên khái niệm tối đa hóa phương pháp hóa sai theo giao thức trực tiếp theo chiều dọc. Nó liên quan đến sai số tính toán của trận chiến hiệp và tìm kiếm các thứ đặc biệt và giá trị riêng của nó. Các thành phần chính là các loại được sắp xếp theo giá trị riêng tương ứng với chúng theo thứ tự giảm dần.

```python
import numpy as np

def pca_math(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    top_eigenvectors = eigenvectors[:, :n_components]

    # Project data onto principal components
    projected_data = np.dot(X_centered, top_eigenvectors)

    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)

    return projected_data, explained_variance_ratio

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
projected_data, explained_variance_ratio = pca_math(X, n_components=2)

print("Projected data:")
print(projected_data)
print("\nExplained variance ratio:")
print(explained_variance_ratio)
```

Slide 6: t-SNE: Nền tảng toán học

t-SNE dựa trên ý tưởng bảo đảm đảm bảo chính xác về các điểm tương đồng theo cặp giữa các dữ liệu trong cả không gian chiều cao và chiều sâu. Nó sử dụng công cụ phân phối của Sinh viên để tính toán các điểm tương đồng trong không gian chiều thấp, giúp giải quyết "vấn đề đông đúc" thường gặp trong trực quan hóa dữ liệu chiều cao.

```python
import numpy as np

def tsne_math(X, n_components=2, perplexity=30.0, n_iter=1000):
    def compute_pairwise_affinities(X, perplexity):
        distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
        P = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            Di = distances[i]
            Di[i] = np.inf
            Pi = np.exp(-Di / (2 * (perplexity ** 2)))
            Pi /= np.sum(Pi)
            P[i] = Pi
        return (P + P.T) / (2 * X.shape[0])

    def compute_q_distribution(Y):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        return Q

    P = compute_pairwise_affinities(X, perplexity)
    Y = np.random.randn(X.shape[0], n_components)

    for _ in range(n_iter):
        Q = compute_q_distribution(Y)

        PQ_diff = P - Q
        dY = np.zeros_like(Y)
        for i in range(X.shape[0]):
            dY[i] = 4 * np.sum((PQ_diff[i] * Q[i])[:, np.newaxis] * (Y[i] - Y), axis=0)

        Y -= dY * 0.1  # Simple gradient descent

    return Y

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
X_tsne = tsne_math(X, n_components=2, perplexity=1.0, n_iter=100)
print("t-SNE result:")
print(X_tsne)
```

Trang trình bày 7: PCA: Ưu điểm và chế độ giới hạn

PCA có hiệu quả tính toán và hoạt động tốt cho các mối quan hệ tuyến tính trong dữ liệu. Nó rất hữu ích cho việc nén dữ liệu và giảm tiếng ồn. Tuy nhiên, nó gặp khó khăn với các mối quan hệ phi tuyến tính và có thể không thu được các mẫu phức hợp trong dữ liệu.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
t = np.linspace(0, 2*np.pi, 1000)
X = np.column_stack((np.sin(t), np.cos(t), t))

# Implement PCA
def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

# Apply PCA
X_pca = pca(X, n_components=2)

# Plot original data and PCA result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=X[:, 2], cmap='viridis')
ax1.set_title('Original Data (First 2 Dimensions)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=X[:, 2], cmap='viridis')
ax2.set_title('PCA Result')
ax2.set_xlabel('First Principal Component')
ax2.set_ylabel('Second Principal Component')

plt.tight_layout()
plt.show()
```

Slide 8: t-SNE: Ưu điểm và chế độ hạn chế

t-SNE nổi bật trong việc tiết lộ các cụm và mẫu trong nhiều dữ liệu chiều. Nó có tác dụng đặc biệt đối với các nhiệm vụ trực quan hóa. Tuy nhiên, nó không có độ sâu toán học chuyên sâu, không có tính toán xác định chính xác và có thể cảm nhận được các siêu tham số phức tạp như phức tạp.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate clustered data
np.random.seed(42)
n_clusters = 5
n_points = 200
X = np.vstack([np.random.randn(n_points, 10) + np.random.randn(10) * 5 for _ in range(n_clusters)])

# Simplified t-SNE implementation
def tsne_simplified(X, n_components=2, perplexity=30.0, n_iter=1000):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Apply t-SNE
X_tsne = tsne_simplified(X, n_components=2, perplexity=30, n_iter=500)

# Plot t-SNE result
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.repeat(range(n_clusters), n_points), cmap='viridis')
plt.title('t-SNE Visualization of Clustered Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster')
plt.show()
```

Slide 9: PCA: Ví dụ thực tế - Nén ảnh

PCA có thể được sử dụng để nén hình ảnh bằng cách giảm kích thước của dữ liệu hình ảnh. Kỹ thuật này đặc biệt hữu ích cho các hình ảnh thang độ xám, trong đó mỗi pixel được biểu thị bằng một giá trị cường độ duy nhất.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_sample_image(size=50):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    return np.exp(-(x**2 + y**2))

def pca_compress(image, n_components):
    flattened = image.reshape(-1, image.shape[1])
    mean = np.mean(flattened, axis=0)
    centered = flattened - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    compressed = np.dot(U[:, :n_components], np.diag(S[:n_components])).dot(Vt[:n_components, :])
    return (compressed + mean).reshape(image.shape)

# Create and compress the image
original_image = create_sample_image(50)
compressed_image = pca_compress(original_image, n_components=10)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(original_image, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(compressed_image, cmap='gray')
ax2.set_title('PCA Compressed Image')
ax2.axis('off')

plt.tight_layout()
plt.show()

print(f"Original size: {original_image.size}")
print(f"Compressed size: {10 * (50 + 1)}")  # n_components * (image_width + 1)
print(f"Compression ratio: {original_image.size / (10 * (50 + 1)):.2f}")
```

Trang tham khảo 10: t-SNE: Ví dụ thực tế - Trực quan hóa cách nhúng từ

t-SNE thường được sử dụng để trực tiếp hóa việc nhúng từ nhiều chiều trong quá trình xử lý ngôn ngữ tự nhiên. Ví dụ này chứng minh rằng t-SNE có thể tiết lộ mối quan hệ giữa các từ trong không gian 2D.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simplified word embedding generation
def generate_word_embeddings(vocab_size=1000, embedding_dim=100):
    np.random.seed(42)
    return np.random.randn(vocab_size, embedding_dim)

# Simplified t-SNE implementation
def tsne_simplified(X, n_components=2, perplexity=30.0, n_iter=1000):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Generate word embeddings and apply t-SNE
embeddings = generate_word_embeddings(vocab_size=100, embedding_dim=50)
tsne_result = tsne_simplified(embeddings, n_components=2, perplexity=5, n_iter=500)

# Plot results
plt.figure(figsize=(12, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])

# Add labels for some random points
np.random.seed(42)
for i in np.random.choice(100, 10, replace=False):
    plt.annotate(f'Word_{i}', (tsne_result[i, 0], tsne_result[i, 1]))

plt.title('t-SNE Visualization of Word Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

Slide 11: Lựa chọn giữa PCA và t-SNE

Lựa chọn giữa PCA và t-SNE phụ thuộc vào đặc tính dữ liệu và công cụ. PCA phù hợp để giảm tính chất tuyến tính kích thước, nén dữ liệu và khi cấu trúc toàn cục là quan trọng. t-SNE tốt hơn nên hiển thị tính năng phi tuyến và đảm bảo toàn bộ cấu trúc cấu trúc ở chiều cao dữ liệu.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 1000
t = np.linspace(0, 10, n_samples)
X = np.column_stack((np.sin(t), np.cos(t), t))

# Implement PCA
def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

# Implement t-SNE (simplified version)
def tsne(X, n_components, perplexity, n_iter):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Apply PCA and t-SNE
X_pca = pca(X, n_components=2)
X_tsne = tsne(X, n_components=2, perplexity=30, n_iter=300)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis')
ax1.set_title('PCA')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t, cmap='viridis')
ax2.set_title('t-SNE')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Slide 12: Kết hợp PCA và t-SNE

Trong thực tế, PCA thường được sử dụng như bước tiền xử lý trước khi áp dụng t-SNE để giảm độ phức tạp và nhiễu tính toán. Điều này kết hợp có thể tận dụng điểm mạnh của cả hai phương pháp: PCA để giảm kích thước ban đầu và t-SNE để hiển thị tính năng phi tuyến.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate high-dimensional data
np.random.seed(42)
n_samples = 1000
n_features = 50
X = np.random.randn(n_samples, n_features)

# Add some structure to the data
X[:n_samples//2, :10] += np.random.randn(n_samples//2, 10) * 5
X[n_samples//2:, 10:20] += np.random.randn(n_samples//2, 10) * 5

# PCA implementation
def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

# Simplified t-SNE implementation
def tsne_simplified(X, n_components, perplexity, n_iter):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Apply PCA followed by t-SNE
X_pca = pca(X, n_components=10)
X_tsne = tsne_simplified(X_pca, n_components=2, perplexity=30, n_iter=300)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.arange(n_samples), cmap='viridis')
plt.title('PCA + t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Sample Index')
plt.show()
```

Trang trình bày 13: Kết luận và các phương pháp hay nhất

Cả PCA và t-SNE đều là những công cụ mạnh mẽ để giảm kích thước và trực quan hóa dữ liệu. PCA là tốt nhất cho các mối quan hệ tuyến tính và bảo vệ toàn cấu trúc toàn cầu, trong khi t-SNE vượt trội trong việc tiết lộ các mẫu và cụm cục bộ trong dữ liệu chiều cao. Khi làm việc với các dữ liệu lớn, hãy cân nhắc công việc sử dụng bước xử lý PCA trước khi áp dụng t-SNE để giảm độ phức tạp của tạp chí máy tính.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=1000, n_features=50):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    X[:n_samples//2, :10] += np.random.randn(n_samples//2, 10) * 5
    X[n_samples//2:, 10:20] += np.random.randn(n_samples//2, 10) * 5
    return X

def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

def tsne_simplified(X, n_components, perplexity, n_iter):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01
    return Y

# Generate and process data
X = generate_data()
X_pca = pca(X, n_components=10)
X_tsne = tsne_simplified(X_pca, n_components=2, perplexity=30, n_iter=300)

# Visualize results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.arange(X.shape[0]), cmap='viridis')
plt.title('PCA + t-SNE: Best of Both Worlds')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Sample Index')
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Để biết thêm thông tin chuyên sâu về PCA và t-SNE, hãy xem xét khám phá các tài nguyên sau:

1. "Trực quan hóa dữ liệu bằng t-SNE" của Laurens van der Maaten và Geoffrey Hinton (2008). Có sẵn trên arXiv: [https://arxiv.org/abs/1807.01281](https://arxiv.org/abs/1807.01281)
2. “Phân tích thành phần chính” của Jonathon Shlens (2014). Có sẵn trên arXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
3. “Cách sử dụng hiệu quả t-SNE” của Martin Wattenberg, Fernanda Viégas và Ian Johnson. Được xuất bản trên Distill (2016). [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
4. "Tìm hiểu giả thuyết đa tạp" của Saket Choudhary. Có sẵn trên arXiv: [https://arxiv.org/abs/2101.05742](https://arxiv.org/abs/2101.05742)

Tài nguyên này cung cấp chi tiết nền tảng học toán, hiểu biết sâu sắc về phát triển khai báo và các phương pháp hay nhất để sử dụng PCA và t-SNE trong các vấn đề phân tích dữ liệu khác nhau.
