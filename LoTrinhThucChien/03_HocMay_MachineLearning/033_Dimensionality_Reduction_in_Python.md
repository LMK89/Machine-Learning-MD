## Giảm kích thước trong Python
Slide 1: Giới thiệu về Giảm kích thước

Giảm kích thước là một kỹ thuật quan trọng trong khoa học dữ liệu và máy học, được sử dụng để đơn giản hóa các bộ dữ liệu phức tạp trong khi vẫn lưu giữ được thông tin cần thiết. Nó giúp trực quan hóa dữ liệu nhiều chiều, giảm độ phức tạp và giảm thiểu lời khuyên về chiều.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample high-dimensional data
np.random.seed(42)
X = np.random.randn(100, 50)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize the reduced data
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title("2D Representation of 50D Data")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()
```

Slide 2: Phân tích thành phần chính (PCA)

PCA là một trong những kỹ thuật giảm kích thước phổ biến nhất. Nó hoạt động bằng cách xác định các thành phần chính, theo hướng có tối đa phương pháp sai trong dữ liệu. Các thành phần này trực tiếp giao tiếp với nhau và thu được các mẫu quan trọng nhất trong dữ liệu.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))
```

Trang trình bày 3: t-SNE (Nhúng hàng xóm ngẫu nhiên phân phối t)

t-SNE là một kỹ thuật giảm kích thước tuyến tính, đặc biệt hiệu quả để hiển thị dữ liệu nhiều chiều. Nó hoạt động bằng cách giảm thiểu sự khác biệt giữa hai phân bố: một phân tích đo sự tương đồng theo cặp trong không gian nhiều chiều và phân tích kia đo sự tương đồng theo cặp trong không gian nhiều chiều.

```python
from sklearn.manifold import TSNE
import seaborn as sns

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=iris.target, palette='deep')
plt.title("t-SNE visualization of Iris dataset")
plt.show()
```

Slide 4: UMAP (Xấp xỉ và cho phép đa tạp đều)

UMAP là một kỹ thuật giảm kích thước phi mạnh mẽ khác. Nó dựa trên các kỹ thuật học tập đa dạng và phân tích dữ liệu phân tích. UMAP thường cung cấp khả năng bảo trì toàn cấu trúc tốt hơn t-SNE trong khi vẫn duy trì hiệu suất tính toán.

```python
import umap

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Visualize the result
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=iris.target, palette='deep')
plt.title("UMAP visualization of Iris dataset")
plt.show()
```

Trang trình bày 5: Bộ mã hóa tự động để giảm kích thước

Bộ mã hóa tự động là mạng lưới thần kinh có thể được sử dụng để giảm kích thước. Chúng bao gồm một bộ đầu vào nén mã hóa hóa và một bộ giải mã tái tạo nó. Lớp hào cổ chai ở giữa có thể hiện không giảm chiều.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define the autoencoder architecture
input_dim = X.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Use the encoder to get the reduced representation
X_encoded = encoder.predict(X)
```

Trang trình bày 6: Lựa chọn tính năng và trích xuất tính năng

Việc giảm kích thước có thể đạt được thông qua lựa chọn tính năng hoặc trích xuất tính năng. Bấm vào các tính năng bao gồm việc chọn một tập hợp các tính năng ban đầu, trong khi trích xuất tính năng tạo ra các tính năng mới bằng cách kết hợp các tính năng gốc. PCA là một ví dụ về trích xuất đặc điểm, trong khi các phương pháp như Lasso có thể được sử dụng để lựa chọn đặc điểm.

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Feature selection using ANOVA F-value
selector = SelectKBest(f_classif, k=2)
X_selected = selector.fit_transform(X, iris.target)

# Print selected feature indices
print("Selected feature indices:", selector.get_support(indices=True))

# Visualize selected features
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=iris.target)
plt.title("Selected Features")
plt.show()
```

Slide 7: Lời nói của kích thước

Lời nói của chiều không đề cập đến nhiều biểu tượng khác nhau khi phân tích dữ liệu trong không gian nhiều chiều. Khi số chiều tăng lên, có thể phân tích không gian tăng nhanh đến mức dữ liệu có sẵn trở nên thưa thớt, tạo việc phân tích kê trở nên khó khăn.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_random_points(dim, num_points=1000):
    return np.random.random((num_points, dim))

def calculate_pairwise_distances(points):
    return np.linalg.norm(points[:, np.newaxis] - points, axis=2)

dims = range(1, 101, 10)
avg_distances = []

for dim in dims:
    points = generate_random_points(dim)
    distances = calculate_pairwise_distances(points)
    avg_distances.append(np.mean(distances))

plt.plot(dims, avg_distances)
plt.xlabel("Number of Dimensions")
plt.ylabel("Average Pairwise Distance")
plt.title("Effect of Dimensionality on Average Pairwise Distance")
plt.show()
```

Slide 8: Học đa dạng tập tin

Nghiên cứu cơ sở đa dạng của tập tin dựa trên giả định rằng chiều cao của dữ liệu thường nằm trên hoặc dưới dạng chiều tối đa gần nhất. Các kỹ thuật như Isomap và Nhúng tuyến tính địa phương (LLE) cố gắng khám phá cấu trúc đa dạng dạng cơ bản này.

```python
from sklearn.manifold import Isomap, LocallyLinearEmbedding

# Apply Isomap
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X)

# Apply LLE
lle = LocallyLinearEmbedding(n_components=2)
X_lle = lle.fit_transform(X)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X_isomap[:, 0], X_isomap[:, 1], c=iris.target)
ax1.set_title("Isomap")

ax2.scatter(X_lle[:, 0], X_lle[:, 1], c=iris.target)
ax2.set_title("Locally Linear Embedding")

plt.show()
```

Trang trình bày 9: SVD bị cắt ngắn (LSA)

SVD cắt ngắn, còn được gọi là Phân tích ẩn ẩn (LSA) trong xử lý văn bản, là một kỹ thuật giảm kích thước tuyến tính. Nó đặc biệt hữu ích cho các loại ma trận thưa thớt và thường được áp dụng trong khai thác văn bản và xử lý ngôn ngữ tự nhiên.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outfoxes a lazy fox",
    "The lazy fox is quickly outfoxed by the dog"
]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts)

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

# Visualize the result
plt.scatter(X_svd[:, 0], X_svd[:, 1])
plt.title("Truncated SVD on Text Data")
for i, text in enumerate(texts):
    plt.annotate(f"Text {i+1}", (X_svd[i, 0], X_svd[i, 1]))
plt.show()
```

Trang trình bày 10: Ví dụ thực tế: Nén hình ảnh

Có thể sử dụng kích thước nhỏ hơn để nén hình ảnh. Bằng cách áp dụng PCA vào hình ảnh dữ liệu, chúng tôi có thể giữ lại các tính năng quan trọng nhất đồng thời giảm kích thước tệp.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import data

# Load sample image
image = data.camera()

# Reshape image to 2D array
X = image.reshape(-1, image.shape[1])

# Apply PCA with different numbers of components
n_components = [10, 50, 100, 200]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for ax, n in zip(axes.ravel(), n_components):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)

    # Reshape back to image
    img_reconstructed = X_reconstructed.reshape(image.shape)

    ax.imshow(img_reconstructed, cmap='gray')
    ax.set_title(f"{n} components")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 11: Ví dụ thực tế: Phát hiện bất ngờ

Việc giảm kích thước có thể được sử dụng để phát hiện sự bất thường bằng cách xác định các điểm dữ liệu sai lệch đáng kể để biểu hiện đã giảm.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate normal data and anomalies
np.random.seed(42)
normal_data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=1000)
anomalies = np.random.multivariate_normal(mean=[3, 3], cov=[[1, 0.5], [0.5, 1]], size=20)
X = np.vstack((normal_data, anomalies))

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# Reconstruct the data
X_reconstructed = pca.inverse_transform(X_pca)

# Calculate reconstruction error
mse = np.mean(np.square(X_scaled - X_reconstructed), axis=1)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], X[:, 1], c=mse, cmap='viridis')
plt.colorbar(label='Reconstruction Error')
plt.title("Anomaly Detection using PCA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

Trang trình bày 12: Chọn kỹ thuật giảm kích thước phù hợp

Việc lựa chọn phương pháp giảm kích thước phù hợp phụ thuộc vào nhiều yếu tố khác nhau, suy nghĩ hạn chế như bản chất của dữ liệu, kích thước đầu ra mong muốn và các công cụ yêu cầu của nhiệm vụ của bạn. Hãy xem xét các yếu tố như tính tuyến tính và tính phi tuyến, hiệu suất tính toán và khả năng diễn giải khi đưa ra lựa chọn.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Generate sample data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 50)  # 50-dimensional data

# Apply different dimensionality reduction techniques
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=42)
umap_reducer = umap.UMAP(random_state=42)

X_pca = pca.fit_transform(X)
X_tsne = tsne.fit_transform(X)
X_umap = umap_reducer.fit_transform(X)

# Visualize results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.scatter(X_pca[:, 0], X_pca[:, 1])
ax1.set_title("PCA")

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1])
ax2.set_title("t-SNE")

ax3.scatter(X_umap[:, 0], X_umap[:, 1])
ax3.set_title("UMAP")

plt.tight_layout()
plt.show()
```

Trang trình bày 13: Đánh giá kích thước công việc giảm

Đánh giá chất lượng của việc giảm kích thước là rất quan trọng. Số lượng dữ liệu đánh giá phổ biến bao gồm tỷ lệ phương pháp được giải thích thích hợp, lỗi tái tạo và duy trì khoảng cách theo cặp hoặc bộ cấu trúc cục bộ.

```python
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

def evaluate_dim_reduction(X_original, X_reduced):
    # Calculate pairwise distances in original and reduced space
    dist_original = pairwise_distances(X_original)
    dist_reduced = pairwise_distances(X_reduced)

    # Flatten distance matrices
    dist_original_flat = dist_original[np.triu_indices(dist_original.shape[0], k=1)]
    dist_reduced_flat = dist_reduced[np.triu_indices(dist_reduced.shape[0], k=1)]

    # Calculate Spearman correlation
    correlation, _ = spearmanr(dist_original_flat, dist_reduced_flat)

    return correlation

# Evaluate PCA, t-SNE, and UMAP
pca_score = evaluate_dim_reduction(X, X_pca)
tsne_score = evaluate_dim_reduction(X, X_tsne)
umap_score = evaluate_dim_reduction(X, X_umap)

print(f"PCA distance preservation: {pca_score:.4f}")
print(f"t-SNE distance preservation: {tsne_score:.4f}")
print(f"UMAP distance preservation: {umap_score:.4f}")
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về các kỹ thuật giảm kích thước và ứng dụng của chúng, các tài nguyên sau được khuyến nghị:

1. "Giảm kích thước: Đánh giá so sánh" của L.J.P. van der Maaten, E.O. Postma, và H.J. van den Herik (2008) ArXiv: [https://arxiv.org/abs/0904.3841](https://arxiv.org/abs/0904.3841)
2. "Trực quan hóa dữ liệu bằng t-SNE" của L.J.P. van der Maaten và G.E. Hinton (2008) Tạp chí Nghiên cứu Học máy
3. "UMAP: Xấp tĩnh đa thống nhất và phép chiếu để giảm kích thước" của L. McInnes, J. Healy và J. Melville (2018) ArXiv: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)

Bài viết này cung cấp các cuộc thảo luận chuyên sâu về các kỹ thuật giảm kích thước khác nhau, nền tảng toán học và ứng dụng thực tế của chúng.
