## Kỹ thuật thu nhỏ kích thước trong Python
Trang trình bày 1: Phân tích thành phần chính (PCA)

Phân vùng thành phần chính là một cơ sở kỹ thuật giảm kích thước giúp chuyển đổi dữ liệu có chiều cao thành chiều thấp hơn trong khi vẫn duy trì phương pháp tối đa sai lệch. Nó hoạt động bằng cách xác định các hướng dẫn trực tiếp giao thức (các thành phần chính) nắm bắt các biến thể quan trọng nhất trong dữ liệu.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 4)

# Initialize and fit PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative variance ratio: {np.cumsum(pca.explained_variance_ratio_)}")
```

Slide 2: Cơ sở toán học của PCA

Nền tảng học toán của PCA xoay quanh công việc phân tích riêng của ma trận Hiệp phương sai. Các thành phần chính là những thứ được đặt riêng tương ứng với giá trị riêng biệt lớn nhất của ma trận Hiệp phương sai.

```python
def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    components = eigenvectors[:, :n_components]

    # Project data
    return np.dot(X_centered, components)
```

Slide 3: t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction technique that emphasizes the preservation of local structure in the data. It's particularly effective for visualizing high-dimensional data by maintaining the relative distances between points.

```python
from sklearn.manifold import TSNE
import seaborn as sns

def apply_tsne(X, perplexity=30, n_components=2):
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1])
    plt.title('t-SNE visualization')
    plt.show()
```

Slide 4: UMAP (Xấp xỉ và cho phép đa tạp đều)

UMAP là một thuật toán giảm kích thước hiện đại kết hợp các nền tảng kỹ thuật từ công việc học đa dạng và dữ liệu phân tích trên bảng. Nó thường cung cấp khả năng bảo trì toàn bộ cấu trúc tốt hơn t-SNE trong khi vẫn duy trì hiệu quả tính toán.

```python
import umap
import pandas as pd

def apply_umap(X, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                       min_dist=min_dist,
                       random_state=42)
    X_umap = reducer.fit_transform(X)

    # Create DataFrame for visualization
    df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])

    # Plot results
    plt.figure(figsize=(10, 8))
    plt.scatter(df_umap['UMAP1'], df_umap['UMAP2'], alpha=0.6)
    plt.title('UMAP projection')
    plt.show()
```

Trang trình bày 5: Bộ mã hóa tự động để giảm kích thước

Bộ cung cấp tự động mã hóa hóa cung cấp cách tiếp cận dựa trên thần kinh mạng để giảm kích thước, học cách biểu diễn nén dữ liệu đầu vào thông tin qua trình giải mã hóa mã hóa giúp giảm thiểu lỗi tái sinh.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def create_autoencoder(input_dim, encoding_dim):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)

    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Full autoencoder
    autoencoder = Model(input_layer, decoded)

    # Encoder model
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder
```

Slide 6: Ứng dụng thực tế - Giảm kích thước hình ảnh

Ví dụ này có thể thực hiện việc giảm kích thước trên dữ liệu MNIST, so sánh các phương pháp PCA, t-SNE và UMAP để trực tiếp hóa hình ảnh dữ liệu nhiều chiều trong không gian 2D cho các tác vụ nhận dạng mẫu.

```python
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
digits = load_digits()
X = digits.data
y = digits.target
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting
plt.figure(figsize=(12, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA of MNIST digits')
plt.colorbar()
plt.show()
```

Trang trình bày 7: Triển khai hạt nhân PCA

Hệ thống truyền tải kernel mở rộng PCA PCA bằng cách sử dụng các phương thức kernel để thực hiện giảm kích thước trong khi không có tính năng ẩn, làm cho nó có khả năng thu thập các mẫu tuyến tính trong dữ liệu thông tin qua các chức năng kernel khác nhau.

```python
from sklearn.decomposition import KernelPCA
import numpy as np

def apply_kernel_pca(X, n_components=2, kernel='rbf'):
    # Initialize and fit KernelPCA
    kpca = KernelPCA(n_components=n_components,
                     kernel=kernel,
                     random_state=42)
    X_kpca = kpca.fit_transform(X)

    # Compute explained variance (approximated)
    explained_var = np.var(X_kpca, axis=0)
    explained_var_ratio = explained_var / np.sum(explained_var)

    return X_kpca, explained_var_ratio
```

Trang trình bày 8: Nhúng tuyến tính cục bộ (LLE)

LLE là một kỹ thuật học đa dạng giúp bảo vệ toàn bộ bộ dữ liệu hình học bằng cách xây dựng lại từng điểm từ các điểm lân cận, tạo ra hiệu ứng đặc biệt đối với dữ liệu nằm trên đa tuyến phi tuyến.

```python
from sklearn.manifold import LocallyLinearEmbedding

def apply_lle(X, n_neighbors=10, n_components=2):
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                n_components=n_components,
                                random_state=42)
    X_lle = lle.fit_transform(X)

    # Reconstruction error
    error = lle.reconstruction_error_
    print(f"LLE Reconstruction error: {error}")

    return X_lle
```

Slide 9: Thực hiện phân tích nhân tố

Phân tích nhân tố giả định rằng các biến thể quan sát có thể được mô tả hóa dưới dạng kết hợp tính chất của các yếu tố tiềm ẩn ẩn giấu không được quan sát cộng với các thuật ngữ số sai, cung cấp cách tiếp cận cụ thể để giảm kích thước.

```python
from sklearn.decomposition import FactorAnalysis
import numpy as np

def apply_factor_analysis(X, n_components=2):
    # Initialize and fit Factor Analysis
    fa = FactorAnalysis(n_components=n_components,
                       random_state=42)
    X_fa = fa.fit_transform(X)

    # Get components and noise variances
    components = fa.components_
    noise_variance = fa.noise_variance_

    return X_fa, components, noise_variance
```

Trang trình bày 10: Tỷ lệ chia tối đa chiều (MDS)

MDS đục mục duy trì khoảng cách giữa các điểm trong không gian nhiều chiều khi tham chiếu các kích thước thấp hơn, để cung cấp các hệ thống biến thể và phi hệ thống cho các loại bảo trì toàn bộ các khoảng cách khác nhau.

```python
from sklearn.manifold import MDS
import numpy as np

def apply_mds(X, n_components=2, metric=True):
    # Initialize and fit MDS
    mds = MDS(n_components=n_components,
              metric=metric,
              random_state=42)
    X_mds = mds.fit_transform(X)

    # Compute stress (goodness of fit)
    stress = mds.stress_
    print(f"MDS Stress: {stress}")

    return X_mds
```

Trang trình bày 11: So sánh phân tích tổng hợp dữ liệu

Việc phát triển này tạo ra một dữ liệu tổng hợp có cấu hình đã biết để so sánh hiệu quả của các kích thước kỹ thuật giảm khác nhau, cung cấp dữ liệu để đánh giá hiệu suất của chúng.

```python
from sklearn.datasets import make_swiss_roll
import numpy as np
from sklearn.metrics import trustworthiness

def compare_reduction_methods(n_samples=1000):
    # Generate swiss roll dataset
    X, color = make_swiss_roll(n_samples, random_state=42)

    # Apply different methods
    methods = {
        'PCA': PCA(n_components=2),
        'tSNE': TSNE(n_components=2, random_state=42),
        'UMAP': umap.UMAP(random_state=42),
        'MDS': MDS(n_components=2, random_state=42)
    }

    results = {}
    for name, method in methods.items():
        X_reduced = method.fit_transform(X)
        # Calculate trustworthiness
        trust = trustworthiness(X, X_reduced, n_neighbors=10)
        results[name] = {'embedding': X_reduced, 'trust': trust}
        print(f"{name} trustworthiness: {trust:.4f}")

    return results
```

Slide 12: Ứng dụng thực tế - Phân tích biểu hiện gen

Trong ứng dụng thực tế này, chúng tôi phân tích dữ liệu biểu thị chiều cao hiện tại, bằng cách giảm kích thước bằng chứng có thể tiết lộ các mẫu ẩn trong bộ dữ liệu sinh học.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def analyze_gene_expression(expression_matrix):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expression_matrix)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)

    # Apply UMAP for visualization
    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(X_pca)

    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    return X_umap, explained_var, cumulative_var
```

Slide 13: Công thức toán học trong việc giảm kích thước công việc

Tổng quan về các nền tảng toán học làm cơ sở cho các kỹ thuật giảm kích thước khác nhau, được trình bày giống nhau với các nền tảng phương pháp và cơ sở lý thuyết của chúng.

```python
# Mathematical formulations for different techniques

# PCA objective function
"""
$$\arg\max_{W} \frac{1}{n}\sum_{i=1}^n (x_i^T W)^T (x_i^T W)$$
"""

# t-SNE probability computation
"""
$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$
"""

# UMAP fuzzy topological representation
"""
$$\mu_{Z}(x) = \exp(-\frac{d(x,Z)}{\rho_0})$$
"""

# Autoencoder loss function
"""
$$L(x,x') = ||x - x'||^2 + \lambda \sum_{l=1}^{L} ||W^{(l)}||_F^2$$
"""
```

Trang trình bày 14: Tài nguyên bổ sung

* ArXiv: "Khảo sát về kỹ thuật giảm kích thước" - [https://arxiv.org/abs/2007.07844](https://arxiv.org/abs/2007.07844)
* ArXiv: "Tìm hiểu về UMAP" - [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
* ArXiv: "Trực quan hóa dữ liệu bằng t-SNE" - [https://arxiv.org/abs/1807.01882](https://arxiv.org/abs/1807.01882)
* Tài nguyên chung: [https://scikit-learn.org/stable/modules/manifold.html](https://scikit-learn.org/stable/modules/manifold.html)
* Bộ sưu tập tập hướng dẫn: [https://towardsdatascience.com/directionality-reduction-techniques-comparison-573cd6b357cb](https://towardsdatascience.com/directionality-reduction-techniques-comparison-573cd6b357cb)
