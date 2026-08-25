##Thuật toán giảm kích thước trong Python
Slide 1: Giới thiệu về Phân tích thành phần chính (PCA)

Thành phần phân vùng chính là kỹ thuật giảm cơ sở kích thước, chuyển đổi dữ liệu có chiều cao thành chiều thấp hơn trong khi vẫn duy trì phương pháp sai tối đa. Nó hoạt động bằng cách tìm kiếm các trục trực tiếp (các thành phần chính) nắm bắt các mẫu quan trọng nhất trong dữ liệu.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 10)  # 100 samples, 10 features

# Initialize and fit PCA
pca = PCA()
X_transformed = pca.fit_transform(X)

# Calculate explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Mathematical representation (not rendered):
# $$C = \frac{1}{n} X^T X$$
# $$\lambda v = Cv$$

print(f"Explained variance ratio: {explained_variance}")
```

Trang trình bày 2: Triển khai t-SNE

t-Distributed Stochastic Neighbor Embedding chiếm ưu thế trong công việc bảo vệ toàn bộ cấu trúc cục bộ trong chiều cao dữ liệu bằng cách cài đặt mô hình phân bố xác thực của các điểm tương đồng theo cặp giữa các điểm trong cả không chiều cao và chiều thấp.

```python
from sklearn.manifold import TSNE
import numpy as np

# Generate high-dimensional data
X = np.random.randn(1000, 50)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto')
X_tsne = tsne.fit_transform(X)

# Visualize results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title('t-SNE Visualization')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
```

Slide 3: Tổng quan về UMAP

Xấp xỉ và phức tạp Đồng bộ nền tảng lý thuyết hợp lý nhất của công việc học đa tạp với kết quả tính toán hiệu quả, tạo ra kết quả đặc biệt đối với mô-đun dữ liệu lớn nhất trong khi vẫn đảm bảo toàn bộ cấu trúc cục bộ và toàn cục.

```python
import umap
import numpy as np
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X = digits.data

# Apply UMAP
reducer = umap.UMAP(n_neighbors=15,
                   min_dist=0.1,
                   n_components=2,
                   random_state=42)
X_umap = reducer.fit_transform(X)

# Visualization
plt.scatter(X_umap[:, 0], X_umap[:, 1],
           c=digits.target, cmap='Spectral')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
```

Trang trình bày 4: Triển khai bộ mã hóa tự động

Bộ cung cấp tự động hóa mã hóa cung cấp cách tiếp cận dựa trên mạng thần kinh để giảm kích thước, học cách biểu diễn nén dữ liệu đầu vào thông tin qua trình mã hóa giải mã hóa giúp giảm thiểu lỗi tái tạo.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define autoencoder architecture
input_dim = 784  # Example: MNIST dimensions
encoding_dim = 32

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Create and compile model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
```

Trang trình bày 5: Triển khai hạt nhân PCA

Hệ thống truyền tải PCA mở rộng Kernel PCA bằng cách sử dụng các phương thức kernel để nắm bắt các mối quan hệ phi tuyến tính trong dữ liệu, cho phép giảm kích thước cho các tệp có mẫu phi tuyến tính chất phức tạp.

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

# Generate nonlinear data
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05)

# Apply Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Original Data')
plt.subplot(122)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.title('Kernel PCA Transformation')
```

Slide 6: Ứng dụng thực tế - Phân tích biểu hiện gen

Gen biểu hiện dữ liệu thường chứa đặc tính hàng (gen) tương ứng với ít mẫu đối số. Việc phát triển này có thể hiện thực hóa việc giảm kích thước để trực quan hóa các mối quan hệ vật liệu sinh học phức hợp.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Simulate gene expression data
np.random.seed(42)
n_genes = 1000
n_samples = 100
gene_expr = np.random.normal(0, 1, (n_samples, n_genes))

# Preprocessing
scaler = StandardScaler()
gene_expr_scaled = scaler.fit_transform(gene_expr)

# Apply PCA
pca = PCA(n_components=2)
gene_expr_pca = pca.fit_transform(gene_expr_scaled)

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(gene_expr_pca[:, 0], gene_expr_pca[:, 1])
plt.title('Gene Expression PCA')
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

Trang trình bày 7: Phân tích phân tích tuyến tính (LDA)

LDA thực hiện giảm kích thước khi tối đa hóa khả năng phân tách lớp, tạo ra kết quả đặc biệt đối với các nhiệm vụ học có giám sát trong đó công việc duy trì sự phân tách lớp là rất quan trọng.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification

# Generate classified data
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=15, n_redundant=5,
                         n_classes=3, random_state=42)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('LDA Transformation')
```

Trang trình bày 8: Nhúng tuyến tính cục bộ (LLE)

LLE bảo đảm toàn bộ hình học địa phương của dữ liệu nhiều chiều bằng cách tái cấu trúc từng điểm từ các điểm lân cận, cung cấp kỹ thuật giảm kích thước phi tuyến hiệu quả cho công việc học đa dạng.

```python
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import make_swiss_roll

# Generate swiss roll dataset
X, color = make_swiss_roll(n_samples=1000, random_state=42)

# Apply LLE
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                            method='modified', random_state=42)
X_lle = lle.fit_transform(X)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 2], c=color)
plt.title('Original Swiss Roll')
plt.subplot(122)
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=color)
plt.title('LLE Transformation')
```

Slide 9: Triển khai Isomap

Hệ thống truyền MDS mở rộng Isomap bằng cách thay thế khoảng cách Euclide bằng khoảng cách trắc địa, thu được hiệu quả hình học nội tại của đa tạp phi tuyến trong dữ liệu.

```python
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler

# Generate complex nonlinear data
t = np.pi * np.linspace(0, 1, 1000)
X = np.column_stack([
    np.sin(2*t), np.cos(3*t),
    np.sin(4*t), np.cos(5*t)
])

# Apply Isomap
isomap = Isomap(n_neighbors=10, n_components=2)
X_iso = isomap.fit_transform(X)

# Scale results for visualization
scaler = MinMaxScaler()
X_iso_scaled = scaler.fit_transform(X_iso)

plt.scatter(X_iso_scaled[:, 0], X_iso_scaled[:, 1], c=t)
plt.colorbar(label='Position in original curve')
plt.title('Isomap Embedding')
```

Slide 10: Phân tích nhân tố

Phân tích nhân tố khám phá các tiềm ẩn giải thích mối tương quan giữa các biến thể được quan sát, cung cấp khả năng giảm chiều có thể giải quyết được, đặc biệt hữu ích trong khoa học tâm lý và xã hội.

```python
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Generate correlated data
n_samples = 1000
n_features = 10
X = np.random.multivariate_normal(
    mean=np.zeros(n_features),
    cov=np.eye(n_features) + 0.3*np.ones((n_features, n_features)),
    size=n_samples
)

# Apply Factor Analysis
fa = FactorAnalysis(n_components=3, random_state=42)
X_fa = fa.fit_transform(X)

# Print factor loadings
print("Factor loadings:")
print(fa.components_.T)
```

Slide 11: Ứng dụng thực tế - Giảm kích thước hình ảnh

Việc phát triển này có thể thực hiện việc giảm kích thước trên hình ảnh dữ liệu, thường được sử dụng trong các tác vụ thị giác máy tính để trích xuất tính năng và hiển thị trực tiếp các bộ dữ liệu hình ảnh nhiều chiều.

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=y, cmap='Spectral')
plt.colorbar(scatter)
plt.title('t-SNE visualization of digits dataset')
print(f"Original shape: {X.shape}, Reduced shape: {X_tsne.shape}")
```

Trang trình bày 12: Triển khai PCA thưa thớt

PCA thưa thớt đưa ra các loại thưa thớt cho hệ thống truyền tải PCA, tạo ra các thành phần chính có ít hệ thống khác 0 hơn, giúp nâng cao khả năng giải quyết và các tính năng lựa chọn.

```python
from sklearn.decomposition import SparsePCA
import numpy as np

# Generate synthetic sparse data
n_samples, n_features = 100, 50
rng = np.random.RandomState(42)
data = rng.randn(n_samples, n_features)

# Apply Sparse PCA
spca = SparsePCA(n_components=5, alpha=1, random_state=42)
X_sparse = spca.fit_transform(data)

# Analyze sparsity
components_sparsity = np.mean(spca.components_ == 0)
print(f"Components sparsity: {components_sparsity:.2%}")

# Visualize first two components
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(spca.components_[0])
plt.title('First Sparse Component')
plt.subplot(122)
plt.plot(spca.components_[1])
plt.title('Second Sparse Component')
```

Trang trình bày 13: PCA tăng dần cho các dữ liệu lớn

PCA tăng dần cho phép giảm kích thước trên dữ liệu quá lớn để phù hợp với bộ nhớ bằng cách xử lý dữ liệu theo thời gian, giúp nó phù hợp với dữ liệu lớn của ứng dụng.

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

# Generate large dataset simulation
def data_generator(n_batches, batch_size, n_features):
    for _ in range(n_batches):
        yield np.random.randn(batch_size, n_features)

# Initialize Incremental PCA
ipca = IncrementalPCA(n_components=10)

# Process data in batches
n_batches = 10
batch_size = 100
n_features = 50

for batch in data_generator(n_batches, batch_size, n_features):
    ipca.partial_fit(batch)

# Show explained variance ratio
print("Explained variance ratio:", ipca.explained_variance_ratio_)
print("Total explained variance:", sum(ipca.explained_variance_ratio_))
```

Slide 14: So sánh hiệu quả của các phương pháp giảm thiểu

Việc phát triển này so sánh các kỹ thuật giảm kích thước khác nhau về thời gian tính toán và lỗi tái tạo trên một tập dữ liệu được tiêu chuẩn hóa.

```python
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from time import time
import numpy as np

# Generate dataset
X = np.random.randn(1000, 50)

# Compare methods
methods = {
    'PCA': PCA(n_components=2),
    'MDS': MDS(n_components=2),
    't-SNE': TSNE(n_components=2)
}

results = {}
for name, method in methods.items():
    start_time = time()
    transformed = method.fit_transform(X)
    results[name] = {
        'time': time() - start_time,
        'shape': transformed.shape
    }

# Print results
for name, metrics in results.items():
    print(f"{name}:")
    print(f"Time: {metrics['time']:.2f}s")
    print(f"Output shape: {metrics['shape']}\n")
```

Trang trình bày 15: Tài nguyên bổ sung

* Tổng quan về kỹ thuật giảm kích thước:
    * [https://arxiv.org/abs/2106.04716](https://arxiv.org/abs/2106.04716)
* Những tiến bộ hiện đại trong kiến trúc bộ mã hóa tự động:
    * [https://arxiv.org/abs/2003.05991](https://arxiv.org/abs/2003.05991)
* Phân tích so sánh các phương pháp học tập đa dạng:
    * [https://arxiv.org/abs/2009.01796](https://arxiv.org/abs/2009.01796)
* Phương pháp học sâu để giảm kích thước:
    * [https://arxiv.org/abs/2102.07559](https://arxiv.org/abs/2102.07559)
* Cơ sở lý thuyết của t-SNE và UMAP:
    * [https://www.google.com/search?q=theoretical+foundations+of+tsne+and+umap+paper](https://www.google.com/search?q=theoretical+foundations+of+tsne+and+umap+paper)
