## Giảm kích thước trong Machine Learning với Python
Slide 1: Giới thiệu về Giảm kích thước

Giảm kích thước là một kỹ thuật quan trọng trong số lượng máy tiêu chuẩn giảm hoặc biến thể trong dữ liệu trong khi vẫn đảm bảo an toàn cho thông tin cần thiết của nó. Quá trình này giúp giải quyết giới hạn về chiều, cải thiện hiệu quả tính toán và có thể nâng cao hiệu suất của các mô hình học.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate a 3D dataset
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 3)
X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1

# Perform PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize the original and reduced data
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2])
ax1.set_title('Original 3D Data')

ax2 = fig.add_subplot(122)
ax2.scatter(X_reduced[:, 0], X_reduced[:, 1])
ax2.set_title('Reduced 2D Data')

plt.tight_layout()
plt.show()
```

Slide 2: Phân tích thành phần chính (PCA)

PCA là một trong những kỹ thuật giảm kích thước phổ biến nhất. Nó hoạt động bằng cách xác định các thành phần chính, theo hướng có sai số phương pháp tối đa trong dữ liệu. Các thành phần này trực tiếp giao tiếp với nhau và thu được các mẫu quan trọng nhất trong dữ liệu.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the results
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.title('PCA of Iris Dataset')
plt.show()

# Print the explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Trang trình bày 3: t-SNE (Nhúng hàng xóm ngẫu nhiên phân phối t)

t-SNE là một kỹ thuật giảm tuyến tính kích thước, đặc biệt hiệu quả để hiển thị dữ liệu nhiều chiều. Nó tập trung vào bộ bảo mật cấu trúc cục bộ, giúp phát hiện các cụm và mẫu trong dữ liệu tổng hợp bộ đệm.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE visualization of the digits dataset')
plt.show()
```

Trang trình bày 4: Bộ mã hóa tự động để giảm kích thước

Bộ mã hóa tự động là mạng lưới thần kinh có thể được sử dụng để giảm kích thước. Chúng tôi bao gồm một bộ mã hóa hóa đầu vào và một bộ giải mã tái tạo nó. Lớp hào cổ chai ở giữa có thể hiện không giảm chiều.

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 10)

# Define the autoencoder architecture
input_dim = X.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=0)

# Encode the data
encoded_data = encoder.predict(X)

# Visualize the encoded data
plt.figure(figsize=(8, 6))
plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
plt.title('2D representation of the input data')
plt.xlabel('Encoded dimension 1')
plt.ylabel('Encoded dimension 2')
plt.show()
```

Trang trình bày 5: Lựa chọn tính năng và trích xuất tính năng

Việc giảm kích thước có thể đạt được thông qua lựa chọn tính năng hoặc trích xuất tính năng. Bấm vào các bao gồm các tính năng để chọn một tập hợp các tính năng ban đầu, trong khi trích xuất tính năng tạo ra các tính năng mới bằng cách kết hợp các tính năng gốc.

```python
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Feature Selection
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)

# Feature Extraction (PCA)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

print("Original feature names:", boston.feature_names)
print("Selected feature indices:", selector.get_support(indices=True))
print("PCA explained variance ratio:", pca.explained_variance_ratio_)

# Visualize the first two components of PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.colorbar(label='House Price')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Boston Housing Dataset')
plt.show()
```

Trang trình bày 6: SVD bị cắt ngắn (Phân tích giá trị tối thiểu)

SVD cut short, còn được gọi là LSA (phân tích ẩn ẩn) trong quá trình xử lý văn bản, là một kỹ thuật giảm kích thước tuyến tính khác. Nó đặc biệt hữu ích cho các ma trận thưa thớt và có thể mang lại hiệu quả hơn PCA đối với một số loại dữ liệu tốt nhất.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(n_samples=1000, n_features=50, centers=5, random_state=42)

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)

# Visualize the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('Truncated SVD visualization')
plt.xlabel('First SVD component')
plt.ylabel('Second SVD component')
plt.show()

print("Explained variance ratio:", svd.explained_variance_ratio_)
```

Slide 7: UMAP (Xấp xỉ và cho phép đa tạp đều)

UMAP là một kỹ thuật giảm kích thước tương đối mới, thường hoạt động tốt hơn t-SNE về mặt bảo tồn cấu trúc cục bộ và toàn cầu. Nhìn chung, nó cũng nhanh hơn t-SNE, làm cho nó phù hợp với các dữ liệu lớn hơn.

```python
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Visualize the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral')
plt.colorbar(scatter)
plt.title('UMAP projection of the digits dataset')
plt.show()
```

Slide 8: Ví dụ thực tế: Nén ảnh

Có thể sử dụng kích thước nhỏ hơn để nén hình ảnh. Bằng cách áp dụng PCA vào dữ liệu hình ảnh, chúng tôi có thể giảm kích thước trong khi vẫn giữ những đặc điểm quan trọng nhất của hình ảnh.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import data

# Load sample image
image = data.camera()

# Reshape the image
X = image.reshape(-1, image.shape[1])

# Apply PCA with different numbers of components
n_components = [10, 50, 100, 200]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, n in zip(axes.ravel(), n_components):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)

    ax.imshow(X_reconstructed.reshape(image.shape), cmap='gray')
    ax.set_title(f'{n} components')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 9: Ví dụ thực tế: Phân tích văn bản

Giảm kích thước là rất quan trọng trong phân tích văn bản cho các tác vụ như phân cụm tài liệu và lập mô hình chủ đề. Vui lòng sử dụng SVD cut short (LSA) để giảm kích thước của dữ liệu văn bản và kết quả trực tuyến hóa.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Sample text data
documents = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The bird flew over the mat",
    "The fish swam in the bowl",
    "The dog barked at the bird"
]

# Create TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Apply Truncated SVD (LSA)
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)

# Visualize the results
plt.figure(figsize=(10, 8))
plt.scatter(X_svd[:, 0], X_svd[:, 1])
for i, doc in enumerate(documents):
    plt.annotate(f"Doc {i+1}", (X_svd[i, 0], X_svd[i, 1]))
plt.title('LSA of text documents')
plt.xlabel('First SVD component')
plt.ylabel('Second SVD component')
plt.show()

print("Top words for each component:")
feature_names = vectorizer.get_feature_names_out()
for i, comp in enumerate(svd.components_):
    top_words = [feature_names[j] for j in comp.argsort()[:-5 - 1:-1]]
    print(f"Component {i + 1}: {', '.join(top_words)}")
```

Trang trình bày 10: Xử lý lời nói theo kích thước

Lời nói của chiều không có vấn đề với nhiều biểu tượng khác nhau khi phân tích dữ liệu trong không gian nhiều chiều. Giảm kích thước giúp giảm thiểu những vấn đề này bằng cách giảm số lượng tính năng trong khi vẫn duy trì được mức độ của thông tin.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_random_points(dim, num_points):
    return np.random.random((num_points, dim))

def calculate_pairwise_distances(points):
    return np.linalg.norm(points[:, np.newaxis] - points, axis=2)

dimensions = range(1, 101, 10)
num_points = 1000
ratios = []

for dim in dimensions:
    points = generate_random_points(dim, num_points)
    distances = calculate_pairwise_distances(points)
    ratio = (np.max(distances) - np.min(distances)) / np.min(distances)
    ratios.append(ratio)

plt.figure(figsize=(10, 6))
plt.plot(dimensions, ratios, marker='o')
plt.title('Effect of Dimensionality on Distance Ratios')
plt.xlabel('Number of Dimensions')
plt.ylabel('Ratio of Max to Min Distance')
plt.grid(True)
plt.show()

print(f"Ratio for 1D: {ratios[0]:.2f}")
print(f"Ratio for 91D: {ratios[-1]:.2f}")
```

Trang trình bày 11: Chọn số lượng phù hợp kích thước

Xác định mức độ ưu tiên của số kích thước là rất quan trọng trong việc giảm kích thước. Chúng tôi có thể sử dụng các kỹ thuật như phương pháp giảm tay hoặc phương pháp giải thích tích lũy để đưa ra quyết định này.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.data

# Perform PCA
pca = PCA()
pca.fit(X)

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)

# Add a line at 95% explained variance
plt.axhline(y=0.95, color='r', linestyle='--')
plt.text(40, 0.96, '95% explained variance', color='r')

# Find the number of components for 95% variance
n_components_95 = next(i for i, ratio in enumerate(cumulative_variance_ratio) if ratio >= 0.95) + 1
plt.axvline(x=n_components_95, color='g', linestyle='--')
plt.text(n_components_95 + 1, 0.5, f'{n_components_95} components', color='g', rotation=90)

plt.show()

print(f"Number of components for 95% explained variance: {n_components_95}")
```

Trang trình bày 12: Giảm kích thước

Việc giảm kích thước trong máy học có thể cải thiện hiệu suất và hiệu quả của mô hình. Dưới đây là ví dụ sử dụng PCA làm bước xử lý trước trong phân loại tác vụ.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
    ('svm', SVC())
])

# Fit the pipeline and make predictions
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Print the number of components selected by PCA
n_components = pipeline.named_steps['pca'].n_components_
print(f"Number of components selected by PCA: {n_components}")

# Compare with a model without PCA
pipeline_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])
pipeline_no_pca.fit(X_train, y_train)
y_pred_no_pca = pipeline_no_pca.predict(X_test)

print("\nClassification report without PCA:")
print(classification_report(y_test, y_pred_no_pca))
```

Trang trình bày 13: Sai công thức và giới hạn của kích thước công việc giảm

Mặc dù các kỹ thuật giảm kích thước rất mạnh nhưng chúng tôi cũng có những công thức và giới hạn. Hiểu những điều này là rất quan trọng để áp dụng hiệu quả trong dự án máy học.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from sklearn.decomposition import PCA

# Generate an S-curve dataset
X, color = make_s_curve(n_samples=1000, noise=0.1, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the original 3D data and the PCA projection
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis')
ax1.set_title('Original 3D S-curve')

ax2 = fig.add_subplot(122)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis')
ax2.set_title('PCA projection to 2D')

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", sum(pca.explained_variance_ratio_))
```

Trang trình bày 14: Tính chất tuyến tính giảm kích thước: Kernel PCA

Kernel PCA là phần mở rộng của PCA có thể nắm bắt các mối quan hệ phi tuyến tính trong dữ liệu bằng cách sử dụng kỹ thuật kernel. Nó đặc biệt hữu ích khi xử lý các tập tin có cấu trúc phi tuyến, phức tạp.

```python
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# Generate a nonlinear dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Apply Kernel PCA with different kernels
kernels = ['linear', 'rbf', 'poly']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, kernel in zip(axes, kernels):
    kpca = KernelPCA(n_components=2, kernel=kernel)
    X_kpca = kpca.fit_transform(X)

    ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis')
    ax.set_title(f'Kernel PCA with {kernel} kernel')
    ax.set_xlabel('First principal component')
    ax.set_ylabel('Second principal component')

plt.tight_layout()
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến công việc tìm hiểu sâu hơn về các kỹ thuật giảm kích thước kỹ thuật thì đây là một số tài nguyên có giá trị:

1. "Khảo sát các kỹ thuật giảm kích thước" của Laurens van der Maaten và cộng sự. (2009) ArXiv: [https://arxiv.org/abs/0903.5485](https://arxiv.org/abs/0903.5485)
2. "Giảm kích thước: Đánh giá so sánh" của Laurens van der Maaten và cộng sự. (2008) Có tại: [http://www.cs.toronto.edu/~hinton/absps/DRtutorial.pdf](http://www.cs.toronto.edu/~hinton/absps/DRtutorial.pdf)
3. "Trực quan hóa dữ liệu bằng t-SNE" của Laurens van der Maaten và Geoffrey Hinton (2008) Tạp chí Nghiên cứu Học máy
4. "UMAP: Phép tiến trình và cho phép hệ thống tối đa hóa hóa để giảm kích thước" của Leland McInnes và cộng đồng. (2018) ArXiv: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)

Những tài nguyên này cung cấp các giải pháp phân tích sâu và nền tảng về các kỹ thuật giảm kích thước khác nhau, cũng như các ứng dụng của chúng trong máy học và trực quan hóa dữ liệu.
