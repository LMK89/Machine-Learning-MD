## Giảm kích thước màn hình trong Python
Trang trình bày 1: Giảm kích thước không giám sát

Giảm kích thước giám sát là một kỹ thuật được sử dụng để giảm số lượng tính năng trong dữ liệu trong khi vẫn đảm bảo an toàn cấu trúc thiết kế yếu của nó. Quá trình này rất quan trọng để xử lý nhiều dữ liệu, cải thiện hiệu quả tính toán và tạo điều kiện trực tuyến. Trong phần trình bày này, chúng tôi sẽ khám phá các phương pháp khác nhau và cách phát triển chúng bằng Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate a sample dataset
X, _ = make_blobs(n_samples=300, n_features=3, centers=4, random_state=42)

# Visualize the 3D data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.title("3D Dataset")
plt.show()
```

Slide 2: Phân tích thành phần chính (PCA)

PCA là một trong những kỹ thuật giảm kích thước phổ biến nhất. Nó xác định các thành phần chính của dữ liệu, đó là mức tối đa sai lệch định hướng phương pháp. Bằng cách tham chiếu dữ liệu lên các thành phần này, chúng tôi có thể giảm kích thước của dữ liệu trong khi vẫn giữ lại toàn bộ thông tin.

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA-reduced Dataset")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

# Print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Trang trình bày 3: t-SNE (Nhúng hàng xóm ngẫu nhiên phân phối t)

t-SNE là một kỹ thuật giảm tuyến tính kích thước, đặc biệt hiệu quả để hiển thị dữ liệu nhiều chiều. Nó bảo tồn bộ cấu trúc cục bộ, giúp ích cho việc khám phá các cụm và mẫu trong dữ liệu.

```python
from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE-reduced Dataset")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.show()
```

Slide 4: UMAP (Xấp xỉ và cho phép đa tạp đều)

Gần đây UMAP là một kỹ thuật giảm kích thước nhằm mục tiêu duy trì mục tiêu tồn tại ở cả cấu trúc cục bộ và toàn cầu. Nó thường nhanh hơn t-SNE và có thể xử lý dữ liệu lớn hơn bằng một cách hiệu quả hơn.

```python
import umap

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("UMAP-reduced Dataset")
plt.xlabel("UMAP Feature 1")
plt.ylabel("UMAP Feature 2")
plt.show()
```

Trang trình bày 5: Bộ mã hóa tự động để giảm kích thước

Bộ mã hóa tự động là mạng lưới thần kinh có thể được sử dụng để giảm kích thước. Chúng bao gồm một bộ nén mã hóa hóa dữ liệu và một bộ giải mã tái tạo nó. Việc nén diễn đàn ở lớp giữa có thể được sử dụng để biểu diễn dữ liệu theo chiều sâu hơn.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the autoencoder model
input_dim = X.shape[1]
encoding_dim = 2

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = models.Model(input_layer, decoded)
encoder = models.Model(input_layer, encoded)

# Compile and train the model
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=0)

# Use the encoder to get the reduced representation
X_encoded = encoder.predict(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1])
plt.title("Autoencoder-reduced Dataset")
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.show()
```

Trang trình bày 6: PCA hạt nhân

Kernel PCA là phần mở rộng của PCA có thể nắm bắt các mối quan hệ phi tuyến tính trong dữ liệu. Nó sử dụng kernel thủ thuật để xạ dữ liệu tới không có chiều cao hơn trước khi áp dụng PCA.

```python
from sklearn.decomposition import KernelPCA

# Apply Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca = kpca.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1])
plt.title("Kernel PCA-reduced Dataset")
plt.xlabel("KPCA Feature 1")
plt.ylabel("KPCA Feature 2")
plt.show()
```

Trang trình bày 7: SVD bị cắt ngắn (LSA)

SVD cut short, còn được gọi là LSA (phân tích ẩn ẩn) trong quá trình xử lý văn bản, là một kỹ thuật giảm kích thước tuyến tính khác. Nó đặc biệt hữu ích cho các ma trận thưa thớt và có thể nhanh hơn PCA đối với một số loại dữ liệu nhất.

```python
from sklearn.decomposition import TruncatedSVD

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_svd[:, 0], X_svd[:, 1])
plt.title("Truncated SVD-reduced Dataset")
plt.xlabel("SVD Feature 1")
plt.ylabel("SVD Feature 2")
plt.show()

# Print explained variance ratio
print("Explained variance ratio:", svd.explained_variance_ratio_)
```

Trình bày 8: Tỷ lệ chia chiều đa chiều (MDS)

MDS là một kỹ thuật duy trì khoảng cách giữa các dữ liệu trong không có chiều sâu hơn. Nó có thể được sử dụng để giảm kích thước tuyến tính và phi tuyến.

```python
from sklearn.manifold import MDS

# Apply MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_mds[:, 0], X_mds[:, 1])
plt.title("MDS-reduced Dataset")
plt.xlabel("MDS Feature 1")
plt.ylabel("MDS Feature 2")
plt.show()
```

Trang trình bày 9: Isomap

Isomap là một kỹ thuật giảm kích thước phi tuyến nỗ lực duy trì khoảng cách trắc địa giữa các dữ liệu. Nó đặc biệt hữu ích cho dữ liệu nằm trên tầng đa chiều được nhúng trong không gian có chiều cao hơn.

```python
from sklearn.manifold import Isomap

# Apply Isomap
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_isomap[:, 0], X_isomap[:, 1])
plt.title("Isomap-reduced Dataset")
plt.xlabel("Isomap Feature 1")
plt.ylabel("Isomap Feature 2")
plt.show()
```

Slide 10: Phân tích nhân tố

Phân tích nhân tố là một phương pháp thống kê được sử dụng để mô tả sự thay đổi giữa các biến số được quan sát, có tương quan với số lượng biến thể không được gọi là các yếu tố có thể thấp hơn.

```python
from sklearn.decomposition import FactorAnalysis

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_fa[:, 0], X_fa[:, 1])
plt.title("Factor Analysis-reduced Dataset")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.show()
```

Slide 11: Ví dụ thực tế: Nén ảnh

Một ứng dụng thực tế của việc giảm kích thước là nén hình ảnh. Chúng tôi có thể sử dụng PCA để giảm kích thước của hình ảnh trong khi vẫn giữ được các đặc tính chính của nó.

```python
from sklearn.decomposition import PCA
import matplotlib.image as mpimg

# Load and preprocess the image
img = mpimg.imread('example_image.jpg')
img_gray = np.mean(img, axis=2)

# Reshape the image
img_reshaped = img_gray.reshape(-1, img_gray.shape[1])

# Apply PCA
pca = PCA(n_components=50)
img_compressed = pca.fit_transform(img_reshaped)

# Reconstruct the image
img_reconstructed = pca.inverse_transform(img_compressed)
img_reconstructed = img_reconstructed.reshape(img_gray.shape)

# Display original and reconstructed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img_gray, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(img_reconstructed, cmap='gray')
ax2.set_title('Reconstructed Image')
plt.show()

# Print compression ratio
original_size = img_gray.size
compressed_size = img_compressed.size
print(f"Compression ratio: {original_size / compressed_size:.2f}")
```

Trang trình chiếu 12: Ví dụ thực tế: Phân cụm tài liệu văn bản

Một ứng dụng thực tế khác của việc giảm kích thước là trong phân tích văn bản. Chúng tôi có thể sử dụng các kỹ thuật như LSA (SVD rút ngắn) để giảm chiều của văn bản dữ liệu cho các tác vụ như phân cụm tài liệu.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outfoxes a lazy canine",
    "The fast and brown fox jumps over the dog",
    "Pythons are non-venomous snakes found in Asia, Africa and Australia",
    "Anacondas are large, non-venomous snakes found in South America",
    "Python is also a popular programming language"
]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Apply LSA
lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X)

# Cluster the documents
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_lsa)

# Visualize the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_lsa[:, 0], X_lsa[:, 1], c=clusters)
plt.title("Document Clusters")
plt.xlabel("LSA Feature 1")
plt.ylabel("LSA Feature 2")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# Print documents with their cluster assignments
for doc, cluster in zip(documents, clusters):
    print(f"Cluster {cluster}: {doc}")
```

Trang trình bày 13: Chọn kỹ thuật giảm kích thước phù hợp

Việc đơn giản lựa chọn phương pháp giảm kích thước phù hợp phụ thuộc vào nhiều yếu tố khác nhau:

1. Đặc điểm dữ liệu: Mối quan hệ tuyến tính và phi tuyến tính
2. Tài nguyên tính toán: Một số phương pháp Hỏi tính toán chuyên sâu hơn
3. Khả năng giải quyết diễn đàn: Các thành phần PCA thường dễ hiểu hơn so với các phần nhúng t-SNE hoặc UMAP
4. Bảo tồn cấu trúc toàn cầu và cục bộ
5. Khả năng mở rộng sang dữ liệu lớn

Hãy xem xét các yếu tố này và thử nghiệm các kỹ thuật khác nhau để tìm ra cách tiếp cận tốt nhất cho vấn đề cụ thể của bạn.

Trang trình bày 14: Chọn kỹ thuật giảm kích thước phù hợp

```python
import time
from sklearn.datasets import make_swiss_roll

# Generate Swiss Roll dataset
X, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# List of dimensionality reduction techniques
techniques = [
    ('PCA', PCA(n_components=2)),
    ('t-SNE', TSNE(n_components=2, random_state=42)),
    ('UMAP', umap.UMAP(random_state=42)),
    ('Isomap', Isomap(n_components=2)),
    ('MDS', MDS(n_components=2, random_state=42))
]

# Apply each technique and measure time
results = []
for name, technique in techniques:
    start_time = time.time()
    X_reduced = technique.fit_transform(X)
    end_time = time.time()
    results.append((name, X_reduced, end_time - start_time))

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, (name, X_reduced, runtime) in enumerate(results):
    axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1])
    axes[i].set_title(f"{name}\nRuntime: {runtime:.2f}s")

plt.tight_layout()
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến công việc tìm hiểu sâu hơn về việc giảm kích thước không giám sát, thì đây là một số tài nguyên có giá trị:

1. "Giảm kích thước: Đánh giá so sánh" của L.J.P. van der Maaten và cộng sự. ArXiv: [https://arxiv.org/abs/0904.3367](https://arxiv.org/abs/0904.3367)
2. "Cách sử dụng kết quả t-SNE" của Martin Wattenberg và cộng đồng. ArXiv: [https://arxiv.org/abs/1610.02831](https://arxiv.org/abs/1610.02831)
3. "UMAP: Phép tiến trình và cho phép hệ thống tối đa hóa hóa để giảm kích thước" của Leland McInnes và cộng đồng. ArXiv: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về các kỹ thuật giảm kích thước khác nhau, tính chất và ứng dụng của chúng.
