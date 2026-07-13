## Làm công việc giảm kích thước bằng phân tích thành phần chính
Slide 1: Giới thiệu về Phân tích thành phần chính

Phân tích thành phần chính (PCA) là một kỹ thuật giảm kích thước giúp chuyển đổi dữ liệu nhiều chiều thành một hệ thống mới trong đó các tính năng không tương quan. Thành phần chính đầu tiên được hướng dẫn bởi tối đa phương sai, với các thành phần tiếp theo theo giao dịch trực tiếp với các thành phần trước đó.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 4)  # 100 samples, 4 features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate covariance matrix
cov_matrix = np.cov(X_scaled.T)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvectors by eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues:", eigenvalues)
print("Explained variance ratio:", eigenvalues / np.sum(eigenvalues))
```

Slide 2: Cơ sở toán học của PCA

Nền tảng học toán của PCA liên quan đến việc tìm kiếm các công việc đặc biệt và có giá trị của ma trận Hiệp phương sai. Điều này được dành riêng cho các thành phần chính, trong khi các giá trị riêng biệt biểu thị phương pháp độ không phù hợp với từng thành phần.

```python
# Mathematical formulation in LaTeX notation
$$
\text{Covariance Matrix} = \Sigma = \frac{1}{n-1}X^TX
$$

$$
\text{Eigendecomposition}: \Sigma v = \lambda v
$$

$$
\text{Transformed Data} = X W
$$

# where X is the centered data matrix
# W is the matrix of eigenvectors
# λ represents eigenvalues
```

Trang trình bày 3: Triển khai PCA từ đầu

```python
def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Select top n_components
    components = eigenvecs[:, :n_components]

    # Project data
    X_transformed = X_centered @ components

    return X_transformed, components, eigenvals

# Example usage
X_transformed, components, eigenvals = pca_from_scratch(X_scaled, 2)
print("Transformed shape:", X_transformed.shape)
```

Slide 4: thích phương sai và lựa chọn giải thích thành phần

Việc hiểu sai phương pháp được giải quyết theo từng thành phần chính là rất quan trọng để xác định số lượng thành phần tối ưu cần giữ lại. Phân tích này giúp cân bằng việc giảm kích thước với công việc bảo toàn thông tin.

```python
def plot_explained_variance(eigenvalues):
    import matplotlib.pyplot as plt

    # Calculate cumulative explained variance ratio
    total_var = np.sum(eigenvalues)
    cum_var_ratio = np.cumsum(eigenvalues) / total_var

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), cum_var_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid(True)

    return cum_var_ratio

# Example usage
explained_variance_ratio = plot_explained_variance(eigenvalues)
print("Explained variance ratios:", explained_variance_ratio)
```

Slide 5: PCA với Scikit-learn

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

# Load real-world dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Initialize and fit PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_pca.shape)
print("Components retained:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Trang trình bày 6: Ví dụ thực tế - Nén hình ảnh

Phân tích thành phần chính có thể được sử dụng một cách hiệu quả để nén hình ảnh bằng cách giảm kích thước của dữ liệu hình ảnh trong khi vẫn duy trì thông tin hình ảnh cần thiết. Ví dụ này có thể thực hiện việc nén một hình ảnh thang độ xám.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image

def compress_image(image_array, n_components):
    # Standardize pixel values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(image_array)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Reconstruct image
    X_reconstructed = pca.inverse_transform(X_pca)
    X_reconstructed = scaler.inverse_transform(X_reconstructed)

    return X_reconstructed, pca.explained_variance_ratio_

# Example usage
img = np.random.rand(100, 100)  # Example grayscale image
compressed_img, var_ratio = compress_image(img, n_components=20)

print(f"Original size: {img.size}")
print(f"Compressed size: {compressed_img.size}")
print(f"Compression ratio: {img.size/compressed_img.size:.2f}")
```

Slide 7: Lựa chọn tính năng bằng PCA

PCA có thể xác định những tính chất gốc đóng góp đáng kể nhất cho các thành phần chính, cho phép đưa ra quyết định lựa chọn tính năng sáng suốt trong bộ dữ liệu nhiều chiều.

```python
def analyze_feature_importance(pca_model, feature_names):
    # Get absolute value of component loadings
    loadings = np.abs(pca_model.components_)

    # Calculate feature importance scores
    importance = np.sum(loadings, axis=0)
    importance = importance / np.sum(importance)

    # Create feature importance dictionary
    feature_importance = dict(zip(feature_names, importance))

    # Sort by importance
    sorted_features = sorted(feature_importance.items(),
                           key=lambda x: x[1],
                           reverse=True)

    return sorted_features

# Example with breast cancer dataset
pca = PCA()
pca.fit(X)
important_features = analyze_feature_importance(pca, data.feature_names)
print("Top 5 most important features:")
for feature, importance in important_features[:5]:
    print(f"{feature}: {importance:.3f}")
```

Trang trình bày 8: PCA tăng dần cho dữ liệu lớn

Khi xử lý dữ liệu quá lớn để phù hợp với bộ nhớ, PCA tăng dần cho phép xử lý dữ liệu theo thời gian trong khi vẫn duy trì khả năng tương thích về mặt toán học với PCA tiêu chuẩn.

```python
from sklearn.decomposition import IncrementalPCA

def incremental_pca_processing(data_generator, n_components, batch_size):
    # Initialize incremental PCA
    ipca = IncrementalPCA(n_components=n_components)

    # Process data in batches
    for batch in data_generator:
        ipca.partial_fit(batch)

    return ipca

# Example with simulated data stream
def generate_batches(n_batches, batch_size, n_features):
    for _ in range(n_batches):
        yield np.random.randn(batch_size, n_features)

# Process batches
ipca = incremental_pca_processing(
    generate_batches(10, 1000, 50),
    n_components=10,
    batch_size=1000
)

print("Number of components:", ipca.n_components_)
print("Explained variance ratio:", ipca.explained_variance_ratio_)
```

Trang trình bày 9: PCA để phát hiện điều bất ngờ

PCA có thể được sử dụng để phát hiện sự cố bất ngờ bằng cách xác định dữ liệu có chế độ tái sử dụng lỗi khi được tham chiếu và quay lại từ thành phần chính không gian.

```python
def detect_anomalies(X, n_components, threshold_percentile=95):
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Reconstruct data
    X_reconstructed = pca.inverse_transform(X_pca)

    # Calculate reconstruction error
    reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)

    # Set threshold
    threshold = np.percentile(reconstruction_error, threshold_percentile)

    # Identify anomalies
    anomalies = reconstruction_error > threshold

    return anomalies, reconstruction_error

# Example usage
X = np.random.randn(1000, 10)
X[0] = X[0] * 10  # Create an obvious anomaly

anomalies, errors = detect_anomalies(X, n_components=5)
print("Number of anomalies detected:", np.sum(anomalies))
print("Reconstruction error for first sample:", errors[0])
```

Trang trình bày 10: PCA để phân tích chuỗi thời gian

PCA có thể trích xuất các mẫu có ý nghĩa từ đa thời gian biến đổi của dữ liệu chuỗi bằng cách xác định các thành phần chính giải thích thời gian khác biệt lớn nhất giữa các kênh thời gian khác nhau.

```python
def analyze_time_series_pca(time_series_data, n_components):
    # Standardize the time series
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(time_series_data)

    # Apply PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    # Calculate component contributions
    reconstructed = pca.inverse_transform(components)
    reconstruction_error = np.mean((X_scaled - reconstructed) ** 2)

    return components, pca.explained_variance_ratio_, reconstruction_error

# Generate example multivariate time series
np.random.seed(42)
t = np.linspace(0, 10, 1000)
signals = np.column_stack([
    np.sin(t),
    np.sin(2*t),
    np.sin(3*t),
    np.random.normal(0, 0.1, len(t))
])

components, var_ratio, error = analyze_time_series_pca(signals, 2)
print(f"Explained variance ratios: {var_ratio}")
print(f"Reconstruction error: {error}")
```

Trang trình bày 11: Triển khai hạt nhân PCA

Hệ thống truyền tải PCA mở rộng PCA của hạt nhân để xử lý các mối quan hệ phi tuyến tính trong dữ liệu bằng cách tham chiếu dữ liệu vào không công cụ có chiều cao hơn bằng hạt nhân thủ thuật.

```python
from sklearn.preprocessing import KernelCenterer
from scipy.linalg import eigh

def kernel_pca(X, n_components, kernel='rbf', gamma=1.0):
    def rbf_kernel(X, Y=None):
        if Y is None:
            Y = X
        return np.exp(-gamma * np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))

    # Compute kernel matrix
    K = rbf_kernel(X)

    # Center kernel matrix
    centerer = KernelCenterer()
    K_centered = centerer.fit_transform(K)

    # Eigendecomposition
    eigenvals, eigenvecs = eigh(K_centered)

    # Sort eigenvectors in descending order
    indices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[indices]
    eigenvecs = eigenvecs[:, indices]

    # Select top components
    return eigenvecs[:, :n_components] * np.sqrt(eigenvals[:n_components])

# Example usage with nonlinear data
X = np.vstack([
    np.random.randn(100, 2) * 0.5,
    np.random.randn(100, 2) * 2.0 + 2
])

X_kpca = kernel_pca(X, n_components=2, gamma=2.0)
print("Transformed shape:", X_kpca.shape)
```

Trang trình bày 12: PCA để giảm tiếng ồn

PCA có thể loại bỏ nhiễu dữ liệu bằng cách xây dựng lại tín hiệu chỉ sử dụng quan trọng nhất của các thành phần chính, lọc các thành phần có khả năng gây nhiễu.

```python
def denoise_with_pca(X, n_components):
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_denoised = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_denoised)

    # Calculate noise reduction metrics
    noise_reduction = np.mean((X - X_reconstructed) ** 2)
    signal_retention = np.sum(pca.explained_variance_ratio_)

    return X_reconstructed, noise_reduction, signal_retention

# Generate noisy data
clean_signal = np.sin(np.linspace(0, 10, 1000))
noise = np.random.normal(0, 0.2, 1000)
noisy_signal = clean_signal + noise

# Reshape for PCA
X = noisy_signal.reshape(-1, 10)
X_denoised, noise_red, signal_ret = denoise_with_pca(X, n_components=3)

print(f"Noise reduction: {noise_red:.4f}")
print(f"Signal retention: {signal_ret:.4f}")
```

Trang trình bày 13: Tài nguyên bổ sung

* "Phân tích thành phần cơ bản trong tính toán tuyến tính đại số và ý nghĩa của việc phân tích dữ liệu"
    * [https://arxiv.org/abs/2108.03247](https://arxiv.org/abs/2108.03247)
* "Hướng dẫn phân tích thành phần chính cho các ứng dụng trong R"
    * [https://arxiv.org/abs/2009.10835](https://arxiv.org/abs/2009.10835)
* "Phân tích thành phần chính mạnh mạnh: Khảo sát và phát triển gần đây"
    * [https://arxiv.org/abs/1705.10403](https://arxiv.org/abs/1705.10403)
* "Phân tích thành phần gốc tăng dần: Khảo sát toàn diện"
    * Để biết các chiến lược phát triển khai chi tiết, hãy tìm kiếm "Triển khai PCA tăng dần" trên Google Scholar
* "Phân tích các thành phần chính và ứng dụng của nó trong dạng nhận dạng khuôn mặt"
    * Có sẵn thông tin Thư viện kỹ thuật số IEEE hoặc tìm kiếm Google Scholar
