## Phân tích các thành phần chính trong Python
Slide 1: Giới thiệu về Phân tích thành phần chính

Phân tích thành phần chính (PCA) là một kỹ thuật giảm kích thước giúp biến đổi dữ liệu có chiều cao thành không có chiều thấp hơn trong khi vẫn giữ được càng nhiều phương pháp càng tốt. Nó xác định các hướng (các thành phần chính) mà dữ liệu thay đổi nhiều nhất.

```python
import numpy as np
from matplotlib import pyplot as plt

# Generate sample 2D data
np.random.seed(42)
data = np.random.randn(100, 2)
data = data @ [[2, 1], [1, 3]]  # Create correlation

plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.axis('equal')
plt.title('Sample 2D Data with Correlation')
```

Slide 2: Cơ sở toán học

PCA dựa trên công việc tìm kiếm các thứ đặc biệt và có giá trị của ma trận chiến đấu. Các phương pháp được đặc biệt định hướng theo phương sai cực đại, trong khi giá trị riêng biểu thị lượng phương sai được giải thích theo từng hướng. Công thức toán học của PCA là:

X' = ​​​​​​X - μ C = (1/n) X'ᵀX' Cv = λv

Trong đó X là ma trận, μ là giá trị trung bình, C là ma trận hiệp phương sai, λ là giá trị riêng và v là riêng.

Slide 3: Nền tảng học toán

```python
def calculate_pca_components(data):
    # Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Calculate covariance matrix
    cov_matrix = np.cov(centered_data.T)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, mean
```

Slide 4: Data Preprocessing

Before applying PCA, data must be preprocessed by centering (subtracting the mean) and optionally scaling. Scaling is crucial when features have different units or variances.

Slide 5: Code for Data Preprocessing

```python
def preprocess_data(X):
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Scale the data
    X_scaled = X_centered / np.std(X_centered, axis=0)

    return X_scaled
```

Slide 6: thích phương sai

Phương pháp giải quyết tỷ lệ không thích hợp bởi từng thành phần chính giúp xác định số lượng thành phần cần giữ lại. Điều này được tính toán bằng cách chia từng giá trị riêng cho tổng của tất cả các giá trị riêng.

Trang trình bày 7: Mã cho phương sai được giải thích

```python
def calculate_explained_variance(eigenvalues):
    # Calculate proportion of variance explained
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance

    # Calculate cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    return explained_variance_ratio, cumulative_variance
```

Slide 8: Real-life Example - Image Compression

PCA can be used for image compression by reducing the dimensionality of image data while preserving important features.

Slide 9: Code for Image Compression Example

```python
def compress_image(image, n_components):
    # Reshape image to 2D array
    h, w = image.shape
    X = image.reshape(h, w)

    # Apply PCA
    eigenvalues, eigenvectors, mean = calculate_pca_components(X)

    # Project data onto principal components
    projected = (X - mean) @ eigenvectors[:, :n_components]

    # Reconstruct image
    reconstructed = projected @ eigenvectors[:, :n_components].T + mean

    return reconstructed.reshape(h, w)
```

Slide 10: Ví dụ thực tế - Phân tích màu sắc

PCA có thể phân tích phần bổ sung màu sắc trong hình ảnh, hữu ích cho các ứng dụng thị giác máy tính như phát hiện đối tượng và hiểu cảnh.

Trang trình bày 11: Ví dụ về mã phân tích màu sắc

```python
def analyze_colors(image):
    # Reshape image to 2D array (pixels x RGB)
    pixels = image.reshape(-1, 3)

    # Apply PCA
    eigenvalues, eigenvectors, mean = calculate_pca_components(pixels)

    # Project colors onto principal components
    projected_colors = (pixels - mean) @ eigenvectors

    return projected_colors, eigenvectors, mean
```

Slide 12: Implementation from Scratch

Here's a complete implementation of PCA without using specialized libraries, useful for understanding the underlying mechanics.

Slide 13: Code for Implementation from Scratch

```python
def pca_from_scratch(X):
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Calculate covariance matrix
    n_samples = X.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvects = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvects = eigenvects[:, idx]

    return eigenvals, eigenvects, X_mean
```

Trang trình bày 14: Tài nguyên bổ sung

Để hiểu sâu hơn về PCA, hãy tham khảo các bài viết được trình duyệt sau:

1. " Hướng dẫn phân tích thành phần chính" - Jonathon Shlens ArXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
2. "Phân tích thành phần chính: Đánh giá và những phát triển gần đây" - Ian Jolliffe & Jorge Cadima ArXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
