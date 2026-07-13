## Phương pháp giảm kích thước để đảm bảo an toàn dữ liệu
Trang trình bày 1: Phân tích thành phần chính (PCA) để giảm kích thước

PCA là một kỹ thuật mạnh mẽ để giảm kích thước trong khi vẫn duy trì sai sót tối đa phương pháp trong dữ liệu. Nó hoạt động bằng cách xác định các thành phần chính, là giao thức trực tiếp thu được các mẫu quan trọng nhất trong dữ liệu.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot results
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Reduced Data')
plt.show()
```

Slide 2: thích sai tỷ lệ phương pháp giải thích

Phương pháp tỷ lệ sai được giải thích giúp chúng tôi hiểu được lượng thông tin được giữ lại sau khi giảm kích thước. Phương pháp tỷ lệ đại diện cho phương pháp này không được giải thích thích hợp bởi từng thành phần chính.

```python
# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()

print(f"Total variance explained: {sum(explained_variance_ratio):.2f}")
```

Slide 3: Chọn số lượng sự kiện

Việc lựa chọn đúng số lượng sự kiện là rất quan trọng. Chúng tôi có thể sử dụng phương pháp không thích hợp để giải thích tích lũy nhằm xác định số lượng thành phần cần giữ lại trong khi vẫn duy trì lượng thông tin mong muốn.

```python
# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot cumulative explained variance ratio
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()

# Find the number of components needed to explain 95% of variance
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components needed to explain 95% of variance: {n_components_95}")
```

Trang trình bày 4: Tái tạo dữ liệu từ kích thước giảm

Sau khi giảm kích thước, chúng tôi có thể xây dựng lại bản gốc của dữ liệu để đánh giá chất lượng của việc giảm kích thước. Quá trình này giúp chúng tôi hiểu được số lượng thông tin bị mất trong quá trình giảm kích thước.

```python
# Reduce dimensionality and reconstruct
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

# Calculate reconstruction error
reconstruction_error = np.mean(np.sum((X - X_reconstructed) ** 2, axis=1))
print(f"Mean reconstruction error: {reconstruction_error:.4f}")

# Visualize original vs reconstructed data (first two dimensions)
plt.scatter(X[:, 0], X[:, 1], label='Original', alpha=0.5)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], label='Reconstructed', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original vs Reconstructed Data')
plt.legend()
plt.show()
```

Trang trình bày 5: Ví dụ thực tế: Nén hình ảnh

PCA có thể được sử dụng để nén hình ảnh bằng cách giảm kích thước của dữ liệu hình ảnh. Ví dụ này chứng minh rằng PCA có thể nén hình ảnh thang độ xám trong khi vẫn giữ được các tính năng chính của nó.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load digit dataset
digits = load_digits()
X = digits.data
y = digits.target

# Select a single digit image
digit_image = X[0].reshape(8, 8)

# Apply PCA with different numbers of components
n_components_list = [2, 5, 10, 20, 30, 40]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Image Reconstruction with Different Numbers of Components')

for i, n_comp in enumerate(n_components_list):
    pca = PCA(n_components=n_comp)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)

    reconstructed_image = X_reconstructed[0].reshape(8, 8)

    ax = axes[i // 3, i % 3]
    ax.imshow(reconstructed_image, cmap='gray')
    ax.set_title(f'{n_comp} Components')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 6: PCA hạt nhân để giảm kích thước phi tuyến tính

PCA kernel mở rộng PCA để xử lý các mối quan hệ phi tuyến tính trong dữ liệu bằng cách tham chiếu các tính năng cấm đầu vào không có chiều cao hơn bằng cách sử dụng kernel.

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# Generate non-linear data
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Apply Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Original Data')
plt.subplot(122)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.title('Kernel PCA Transformed Data')
plt.tight_layout()
plt.show()
```

Trang trình bày 7: PCA tăng dần cho dữ liệu lớn

Khi xử lý dữ liệu lớn không vừa với bộ nhớ, PCA tăng dần cho phép chúng tôi thực hiện giảm kích thước bằng cách xử lý dữ liệu theo khối.

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

# Generate a large dataset
n_samples, n_features = 10000, 100
np.random.seed(42)
X = np.random.randn(n_samples, n_features)

# Apply Incremental PCA
batch_size = 1000
ipca = IncrementalPCA(n_components=10, batch_size=batch_size)

for i in range(0, n_samples, batch_size):
    ipca.partial_fit(X[i:i+batch_size])

# Transform the data
X_reduced = ipca.transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {ipca.explained_variance_ratio_.sum():.2f}")
```

Trang trình bày 8: PCA thưa thớt để lựa chọn tính năng

PCA thưa thớt mang lại lợi ích cho PCA với việc lựa chọn tính năng bằng cách thực hiện tính năng thưa thớt trong các thành phần chính, dẫn đến kết quả dễ hiểu hơn.

```python
from sklearn.decomposition import SparsePCA
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples, n_features = 100, 20
X = np.random.randn(n_samples, n_features)

# Apply Sparse PCA
spca = SparsePCA(n_components=5, alpha=1, random_state=42)
X_sparse = spca.fit_transform(X)

# Visualize component sparsity
plt.figure(figsize=(12, 6))
plt.imshow(spca.components_.T, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Sparse PCA Components')
plt.xlabel('Principal Component')
plt.ylabel('Original Feature')
plt.tight_layout()
plt.show()

print(f"Sparsity ratio: {np.sum(spca.components_ == 0) / spca.components_.size:.2f}")
```

Trang trình bày 9: Ví dụ thực tế: Phân tích tài liệu văn bản

PCA có thể được sử dụng để phân tích và trực quan hóa mối liên hệ giữa các tài liệu văn bản bằng cách giảm kích thước bằng cách biểu thị số ngôn ngữ kỹ thuật của chúng.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample documents
documents = [
    "Machine learning is a subfield of artificial intelligence",
    "Natural language processing deals with text and speech",
    "Deep learning uses neural networks with many layers",
    "Computer vision focuses on image and video analysis",
    "Reinforcement learning is about decision making and rewards"
]

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for i, doc in enumerate(documents):
    plt.annotate(f"Doc {i+1}", (X_pca[i, 0], X_pca[i, 1]))
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Text Documents')
plt.tight_layout()
plt.show()
```

Trang trình bày 10: SVD được cắt giảm (LSA) cho dữ liệu thưa thớt

SVD đã được cắt giảm, còn được gọi là Phân tích ẩn tiềm ẩn (LSA) trong xử lý văn bản, đặc biệt hữu ích để giảm kích thước của da thưa thớt, ví dụ như ma trận TF-IDF.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Sample documents (reusing from previous slide)
documents = [
    "Machine learning is a subfield of artificial intelligence",
    "Natural language processing deals with text and speech",
    "Deep learning uses neural networks with many layers",
    "Computer vision focuses on image and video analysis",
    "Reinforcement learning is about decision making and rewards"
]

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X_svd[:, 0], X_svd[:, 1])
for i, doc in enumerate(documents):
    plt.annotate(f"Doc {i+1}", (X_svd[i, 0], X_svd[i, 1]))
plt.xlabel('First SVD Component')
plt.ylabel('Second SVD Component')
plt.title('Truncated SVD of Text Documents')
plt.tight_layout()
plt.show()

print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.2f}")
```

Trang trình bày 11: t-SNE để giảm và hiển thị kích thước đặc tính tuyến tính

t-SNE (t-Distributed Stochastic Neighbor Embedding) là một kỹ thuật mạnh mẽ để hiển thị dữ liệu chiều cao trong không gian 2D hoặc 3D trong khi vẫn đảm bảo toàn cấu trúc cục bộ.

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE Visualization of Digits Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.show()
```

Trang trình bày 12: UMAP để giảm kích thước tuyến tính nhanh chóng

UMAP (Xấp xỉ và được phép tham chiếu đa dạng đồng nhất) là một thuật toán mới hơn cung cấp khả năng tính toán nhanh hơn và bảo đảm cấu trúc toàn cầu tốt hơn so với t-SNE.

```python
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('UMAP Visualization of Digits Dataset')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.show()
```

Slide 13: So sánh các kỹ thuật giảm kích thước

Các kỹ thuật giảm kích thước khác nhau có điểm mạnh và điểm yếu khác nhau. Trang trình bày sự so sánh này giữa PCA, t-SNE và UMAP trên cùng một dữ liệu tệp để tạo sự khác biệt nổi bật của chúng.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply different dimensionality reduction techniques
pca = PCA(n_components=2, random_state=42)
tsne = TSNE(n_components=2, random_state=42)
umap_reducer = umap.UMAP(random_state=42)

X_pca = pca.fit_transform(X)
X_tsne = tsne.fit_transform(X)
X_umap = umap_reducer.fit_transform(X)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
techniques = [('PCA', X_pca), ('t-SNE', X_tsne), ('UMAP', X_umap)]

for ax, (name, data) in zip(axes, techniques):
    scatter = ax.scatter(data[:, 0], data[:, 1], c=y, cmap='viridis')
    ax.set_title(f'{name} Visualization')
    ax.set_xlabel(f'{name} Component 1')
    ax.set_ylabel(f'{name} Component 2')

plt.colorbar(scatter, ax=axes[-1])
plt.tight_layout()
plt.show()
```

Trang trình bày 14: Chọn kỹ thuật giảm kích thước phù hợp

Việc lựa chọn phương pháp giảm kích thước phù hợp phụ thuộc vào nhiều yếu tố khác nhau như kích thước tập dữ liệu, tỷ lệ thớt, kích thước đầu ra mong muốn và nhu cầu về khả năng giải mã hoặc trực tiếp hóa hóa. Dưới đây là hướng dẫn giúp bạn lựa chọn:

1. PCA: Sử dụng khi mối quan hệ tuyến tính tính là đủ và bạn cần các thành phần có thể hiểu được hoặc tính toán nhanh.
2. Kernel PCA: Áp dụng cho các mối quan hệ phi tuyến tính khi bạn có đủ khả năng chi trả chi phí tính toán cao hơn.
3. Tăng PCA: Chọn các tệp lớn không phù hợp với bộ nhớ.
4. PCA thưa thớt: Lựa chọn tính năng phổ biến và khả năng giải quyết trong không gian nhiều chiều.
5. SVD được cắt giảm (LSA): Thích hợp cho da thưa thớt, đặc biệt là trong quá trình xử lý văn bản.
6. t-SNE: Sử dụng để trực quan hóa dữ liệu nhiều chiều ở dạng 2D hoặc 3D, bảo vệ toàn bộ cấu trúc cục bộ.
7. UMAP: Choose to tính toán nhanh hơn và đảm bảo toàn cấu trúc toàn cục tốt hơn so với t-SNE.

Trang trình bày 15: Chọn kỹ thuật giảm kích thước phù hợp

Việc lựa chọn phương pháp giảm kích thước phù hợp phụ thuộc vào nhiều yếu tố khác nhau như kích thước tập dữ liệu, tỷ lệ thớt, kích thước đầu ra mong muốn và nhu cầu về khả năng giải mã hoặc trực tiếp hóa hóa. Sơ đồ hướng dẫn này đã được quyết định:

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_decision_flowchart():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Start", "Linear?"),
        ("Linear?", "PCA"),
        ("Linear?", "Non-linear"),
        ("PCA", "Large dataset?"),
        ("Large dataset?", "Incremental PCA"),
        ("Large dataset?", "Standard PCA"),
        ("Non-linear", "Visualization?"),
        ("Visualization?", "t-SNE/UMAP"),
        ("Visualization?", "Kernel PCA"),
        ("Kernel PCA", "Sparse data?"),
        ("Sparse data?", "Truncated SVD"),
        ("Sparse data?", "Standard Kernel PCA")
    ])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=3000, font_size=8, arrows=True)

    edge_labels = {("Start", "Linear?"): "Start",
                   ("Linear?", "PCA"): "Yes",
                   ("Linear?", "Non-linear"): "No",
                   ("PCA", "Large dataset?"): "",
                   ("Large dataset?", "Incremental PCA"): "Yes",
                   ("Large dataset?", "Standard PCA"): "No",
                   ("Non-linear", "Visualization?"): "",
                   ("Visualization?", "t-SNE/UMAP"): "Yes",
                   ("Visualization?", "Kernel PCA"): "No",
                   ("Kernel PCA", "Sparse data?"): "",
                   ("Sparse data?", "Truncated SVD"): "Yes",
                   ("Sparse data?", "Standard Kernel PCA"): "No"}

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis('off')
    plt.title("Dimensionality Reduction Technique Selection Flowchart")
    plt.tight_layout()
    plt.show()

create_decision_flowchart()
```

Lưu sơ đồ này cung cấp hướng dẫn trực quan để chọn kỹ thuật giảm kích thước phù hợp nhất dựa trên đặc điểm và yêu cầu dữ liệu của bạn.

Trang trình bày 16: Số liệu đánh giá về việc giảm kích thước công việc

Để đánh giá kích thước giảm chất lượng, chúng tôi có thể sử dụng nhiều loại dữ liệu khác nhau. Dưới đây là ví dụ về mã hóa có thể hiện hai số liệu đánh giá phổ biến: lỗi tái tạo và độ tin cậy.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import numpy as np

# Load the digits dataset
digits = load_digits()
X = digits.data

# Apply PCA
n_components = 20
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

# Calculate reconstruction error
reconstruction_error = np.mean(np.sum((X - X_reconstructed) ** 2, axis=1))
print(f"Reconstruction Error: {reconstruction_error:.4f}")

# Calculate trustworthiness
trust_score = trustworthiness(X, X_reduced)
print(f"Trustworthiness Score: {trust_score:.4f}")

# Plot cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, n_components + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.show()
```

Số liệu này xác định mức độ biểu tượng theo cấu trúc giảm chiều và thông tin của dữ liệu gốc.

Trang trình bày 17: Tài nguyên bổ sung

Để khám phá thêm về các kỹ thuật giảm kích thước và ứng dụng của chúng, hãy xem xét các tài nguyên sau:

1. "Khảo sát các kỹ thuật giảm kích thước" của L. van der Maaten và cộng sự. (2009) ArXiv: [https://arxiv.org/abs/0904.3664](https://arxiv.org/abs/0904.3664)
2. "Trực quan hóa dữ liệu bằng t-SNE" của L. van der Maaten và G. Hinton (2008) Tạp chí Nghiên cứu Học máy
3. "UMAP: Xấp xỉ tối đa hệ thống và được phép tham chiếu để giảm kích thước" của L. McInnes và cộng đồng. (2018) ArXiv: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
4. " Hướng dẫn phân tích thành phần chính" của J. Shlens (2014) ArXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)

Những tài nguyên này cung cấp các cuộc thảo luận chuyên sâu về các phương pháp giảm kích thước khác nhau, nền tảng lý thuyết và ứng dụng thực tế của chúng.