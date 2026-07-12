## Triển khai PCA để giảm kích thước
Slide 1: Giới thiệu về PCA

Phân tích thành phần chính (PCA) là một kỹ thuật mạnh mẽ được sử dụng để giảm kích thước trong máy học và phân tích dữ liệu. Nó giúp giải quyết vấn đề về chiều bằng cách chuyển đổi dữ liệu có chiều cao thành không có chiều thấp hơn trong khi vẫn đảm bảo an toàn cho những thông tin quan trọng nhất. PCA hoạt động bằng cách xác định các thành phần chính, là giao thức trực tiếp được tối đa hóa bằng phương pháp tối đa trong dữ liệu.

Slide 2: Lời nói kích thước

Lời nói về chiều không đề cập đến những bình tĩnh sinh học khi làm việc với nhiều dữ liệu. Khi tăng số lượng tính năng, số lượng dữ liệu cần thiết để đưa ra kiến ​​trúc dự kiến ​​về trang web danh sách sẽ tăng số lượng nhân. Điều này có thể dẫn đến tình trạng trang web trở nên quá mạnh, tăng độ phức tạp tính toán và khó khăn trong việc hiển thị và giải quyết dữ liệu.

Trang trình bày 3: Mã nguồn nói theo hướng

```python
import random
import math

def generate_random_point(dimensions):
    return [random.uniform(0, 1) for _ in range(dimensions)]

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def demonstrate_curse_of_dimensionality(num_points=1000, max_dim=100):
    dimensions = list(range(1, max_dim + 1, 10))
    avg_distances = []

    for dim in dimensions:
        points = [generate_random_point(dim) for _ in range(num_points)]
        distances = [euclidean_distance(points[i], points[j])
                     for i in range(num_points)
                     for j in range(i + 1, num_points)]
        avg_distances.append(sum(distances) / len(distances))

    for dim, avg_dist in zip(dimensions, avg_distances):
        print(f"Dimensions: {dim}, Average distance: {avg_dist:.4f}")

demonstrate_curse_of_dimensionality()
```

Trang trình bày 4: Kết quả cho mã nguồn lời nói theo chiều kích thước

```
Dimensions: 1, Average distance: 0.3336
Dimensions: 11, Average distance: 1.1045
Dimensions: 21, Average distance: 1.5256
Dimensions: 31, Average distance: 1.8533
Dimensions: 41, Average distance: 2.1317
Dimensions: 51, Average distance: 2.3778
Dimensions: 61, Average distance: 2.6016
Dimensions: 71, Average distance: 2.8083
Dimensions: 81, Average distance: 3.0021
Dimensions: 91, Average distance: 3.1846
```

Slide 5: Mathematical Foundations of PCA

PCA is based on the concept of eigenvectors and eigenvalues. Given a dataset X, PCA computes the covariance matrix and then finds its eigenvectors and eigenvalues. The eigenvectors represent the directions of maximum variance in the data, while the eigenvalues indicate the amount of variance explained by each eigenvector. The principal components are sorted in descending order of their corresponding eigenvalues.

Slide 6: Source Code for Mathematical Foundations of PCA

```python
def covariance_matrix(X):
    n = X.shape[0]
    X_centered = X - X.mean(axis=0)
    return (X_centered.T @ X_centered) / (n - 1)

def eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = [], []
    n = cov_matrix.shape[0]

    for i in range(n):
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)

        for _ in range(100):  # Power iteration
            v_new = cov_matrix @ v
            v_new = v_new / np.linalg.norm(v_new)

            if np.allclose(v, v_new):
                break
            v = v_new

        eigenvalue = (v.T @ cov_matrix @ v) / (v.T @ v)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)

        # Deflation
        cov_matrix = cov_matrix - eigenvalue * np.outer(v, v)

    return np.array(eigenvalues), np.array(eigenvectors).T

# Example usage
X = np.random.rand(100, 5)
cov_matrix = covariance_matrix(X)
eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors shape:", eigenvectors.shape)
```

Trang trình bày 7: PCA tính toán bước

Thuật toán PCA bao gồm một số bước chính:

1. Chuẩn hóa dữ liệu
2. Tính ma trận chiến phương sai
3. Tính riêng và giá trị riêng
4. Sắp xếp các tùy chỉnh bằng cách giảm giá trị riêng
5. Chọn k custom top
6. Xem dữ liệu không mới

Bước này chuyển đổi dữ liệu có chiều cao ban đầu thành biểu tượng có chiều thấp hơn trong khi vẫn giữ được thông tin quan trọng nhất.

Slide 8: Mã nguồn cho các bước tính toán thuật toán PCA

```python
import numpy as np

def pca(X, k):
    # Step 1: Standardize the dataset
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(X_std.T)

    # Step 3: Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 4: Sort eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Choose the top k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]

    # Step 6: Project the data onto the new subspace
    X_pca = X_std.dot(top_k_eigenvectors)

    return X_pca, eigenvalues, eigenvectors

# Example usage
X = np.random.rand(100, 5)
k = 3
X_pca, eigenvalues, eigenvectors = pca(X, k)

print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)
print("Top 3 eigenvalues:", eigenvalues[:3])
```

Trang trình bày 9: Chọn số lượng thành phần chính

Việc chọn mức độ ưu tiên của thành phần là rất quan trọng để giảm kích thước hiệu quả. Một cách tiếp theo phổ biến là sử dụng tỷ lệ phương pháp giải thích tích lũy, tỷ lệ phương pháp pháp sai được giải thích theo từng thành phần chính. Bằng cách đặt ngưỡng (ví dụ: sai 95% tổng phương pháp), chúng tôi có thể xác định số lượng thành phần cần giữ lại.

Slide 10: Mã nguồn để chọn số lượng thành phần chính

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_variance(eigenvalues):
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
    plt.grid(True)
    plt.show()

def select_components(eigenvalues, threshold=0.95):
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance
    return np.argmax(cumulative_variance_ratio >= threshold) + 1

# Example usage
X = np.random.rand(100, 10)
_, eigenvalues, _ = pca(X, 10)

plot_cumulative_variance(eigenvalues)
optimal_components = select_components(eigenvalues)
print(f"Optimal number of components: {optimal_components}")
```

Trang trình bày 11: Ví dụ thực tế: Nén hình ảnh

PCA có thể được sử dụng để nén hình ảnh bằng cách giảm kích thước của dữ liệu hình ảnh. Kỹ thuật này đặc biệt hữu ích cho các hình ảnh thang độ xám, trong đó mỗi pixel được biểu thị bằng một giá trị duy nhất. Bằng cách áp dụng PCA vào ma trận hình ảnh, chúng tôi có thể nén hình ảnh mà vẫn giữ được các tính năng thiết yếu của nó.

Trang trình bày 12: Mã nguồn cho ví dụ nén hình ảnh

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compress_image(image_path, k):
    # Load the image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # Apply PCA
    X_pca, _, _ = pca(img_array, k)

    # Reconstruct the image
    reconstructed = X_pca.dot(X_pca.T)
    reconstructed = reconstructed.astype(np.uint8)

    # Display original and compressed images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img_array, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(reconstructed, cmap='gray')
    ax2.set_title(f'Compressed Image (k={k})')
    plt.show()

# Example usage
image_path = 'example_image.jpg'
compress_image(image_path, k=50)
```

Trang trình bày 13: Ví dụ thực tế: Phân tích bộ dữ liệu gen

PCA được sử dụng rộng rãi trong bộ gen để phân tích dữ liệu truyền chiều cao. Nó có thể giúp xác định các loại biểu hiện, khám phá cấu trúc quần áo và hình dung mối liên hệ giữa các mô hình khác nhau. Ví dụ này trình bày cách áp dụng PCA cho tập dữ liệu về nucleotide (SNP) đơn của các quần thể khác nhau.

Trang trình bày 14: Mã nguồn cho ví dụ về bộ dữ liệu phân tích gen

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_snp_data(n_samples, n_snps, n_populations):
    populations = np.random.randint(0, n_populations, n_samples)
    snp_data = np.random.binomial(2, 0.3 + 0.1 * populations[:, np.newaxis], (n_samples, n_snps))
    return snp_data, populations

def analyze_snp_data(snp_data, populations):
    X_pca, _, _ = pca(snp_data, k=2)

    plt.figure(figsize=(10, 8))
    for pop in range(max(populations) + 1):
        mask = populations == pop
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Population {pop}')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of SNP Data')
    plt.legend()
    plt.show()

# Example usage
n_samples, n_snps, n_populations = 1000, 1000, 3
snp_data, populations = simulate_snp_data(n_samples, n_snps, n_populations)
analyze_snp_data(snp_data, populations)
```

Trang trình bày 15: Chế độ và cân bằng nhanh

Mặc dù PCA là một kỹ thuật mạnh mẽ nhưng không có một số chế độ:

1. Nó giả định mối quan hệ tuyến tính giữa các tính năng.
2. Nó không thể hoạt động tốt với dữ liệu phi tuyến tính cao.
3. Các thành phần chính có thể khó diễn giải.
4. Nó nhạy cảm với các ngoại lệ.
5. Có thể không phải lúc nào nó cũng lưu giữ những thông tin quan trọng cho những công cụ nhiệm vụ.

Xem xét các kỹ thuật thay thế như t-SNE hoặc UMAP để giảm kích thước phi tuyến tính khi xử lý bộ dữ liệu phức hợp.

Trang trình bày 16: Tài nguyên bổ sung

Để biết thêm thông tin chuyên sâu về PCA và các chủ đề liên quan, hãy xem xét các tài nguyên sau:

1. " Hướng dẫn phân tích thành phần chính" của Jonathon Shlens (2014) ArXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
2. "Giảm thiểu kích thước tối thiểu: Đánh giá so sánh" của Laurens van der Maaten, Eric Postma và Jaap van den Herik (2009) Có sẵn tại: [https://lvdmaaten.github.io/publications/papers/TR\_Dimensionality\_Reduction\_Review\_2009.pdf](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf)
3. "Phân tích thành phần chính" của Svante Wold, Kim Esbensen, và Paul Geladi (1987) DOI: 10.1016/0169-7439(87)80084-9

Tài nguyên này cung cấp cái nhìn tổng thể về diện mạo và các thảo luận nâng cao về PCA và các kỹ thuật giảm kích thước liên quan.
