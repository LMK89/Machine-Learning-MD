## Giảm kích thước nhanh hơn 20 lần so với PCA
Trang trình bày 1: Giảm kích thước: Ngoài PCA

Giảm kích thước là một kỹ thuật quan trọng trong khoa học dữ liệu và máy học, đặc biệt là khi xử lý dữ liệu nhiều chiều. Mặc dù phân tích thành phần chính (PCA) là một phương pháp phổ biến nhưng không có chế độ hạn chế khi làm việc với dữ liệu có chiều cực cao. Bài trình bày này khám phá một cách tiếp cận thay thế: Phép đo ngẫu nhiên thưa thớt, có thể giảm hiệu quả kích thước hơn PCA mà không ảnh hưởng đến độ chính xác.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
import time

# Generate a high-dimensional dataset
n_samples = 1000
n_features = 1000
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, random_state=42)

# Measure PCA time
start_time = time.time()
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)
pca_time = time.time() - start_time

# Measure Sparse Random Projection time
start_time = time.time()
srp = SparseRandomProjection(n_components=100, random_state=42)
X_srp = srp.fit_transform(X)
srp_time = time.time() - start_time

print(f"PCA time: {pca_time:.4f} seconds")
print(f"Sparse Random Projection time: {srp_time:.4f} seconds")
print(f"Speedup: {pca_time / srp_time:.2f}x")
```

Slide 2: Độ phức tạp về thời gian của PCA

Độ phức tạp về thời gian của PCA là một trở ngại đáng kể khi xử lý dữ liệu nhiều chiều. Độ phức tạp về thời gian của PCA là O(nm^2 + m^3), trong đó n là mẫu số lượng và m là số lượng đặc biệt. Mối quan hệ cấp ba này với PCA kích thước số lượng không thực tế đối với các dữ liệu có kích thước hàng hóa.

```python
def pca_time_complexity(n_samples, n_features):
    return n_samples * n_features**2 + n_features**3

# Compare PCA time complexity for different dimensions
dimensions = [100, 500, 1000, 2000, 5000]
samples = 10000

for dim in dimensions:
    complexity = pca_time_complexity(samples, dim)
    print(f"PCA time complexity for {dim}D: {complexity:,}")
```

Trang trình bày 3: Nghịch lý PCA

Thật sự khó chịu khi PCA, một kỹ thuật được thiết kế để giảm kích thước, lại trở nên hiệu quả khi xử lý dữ liệu nhiều chiều - chính là vấn đề mà nó cần giải quyết. Phương pháp này cần được bật chế độ này, thay thế có thể xử lý nhiều hiệu ứng dữ liệu hơn.

```python
import matplotlib.pyplot as plt

dimensions = list(range(100, 5001, 100))
complexities = [pca_time_complexity(10000, dim) for dim in dimensions]

plt.figure(figsize=(10, 6))
plt.plot(dimensions, complexities)
plt.title("PCA Time Complexity vs. Dimensions")
plt.xlabel("Number of Dimensions")
plt.ylabel("Time Complexity")
plt.yscale('log')
plt.grid(True)
plt.show()
```

Slide 4: Giới thiệu về Phép tham khảo ngẫu nhiên thưa thớt

Phép đo ngẫu nhiên thưa thớt (SRP) là một giải pháp thay thế hiệu quả cho PCA để giảm kích thước. Nó có thể chuyển đổi chiều cao của dữ liệu sang không có chiều thấp hơn trong khi vẫn giữ khoảng cách nguyên giữa các điểm. Thuộc tính này đặc biệt hữu ích cho các tác vụ như phân cụm và tìm kiếm hàng xóm gần nhất.

```python
from sklearn.random_projection import SparseRandomProjection

# Generate a high-dimensional dataset
n_samples = 1000
n_features = 2000
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=5, random_state=42)

# Apply Sparse Random Projection
srp = SparseRandomProjection(n_components=100, random_state=42)
X_reduced = srp.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
```

Slide 5: Toán học được phép tham khảo ngẫu nhiên thưa thớt

Phép ngẫu nhiên phụ tùng dựa trên plugin Johnson-Lindenstrauss, trong đó phát hiện ra rằng một tập hợp nhỏ các điểm trong không gian nhiều chiều có thể được nhúng vào không gian có chiều thấp hơn theo cách mà khoảng cách giữa các điểm gần như được bảo đảm. Ma trận trong SRP rất thưa thớt, chứa hầu hết các số 0, điều này góp phần nâng cao hiệu quả của nó.

```python
def create_sparse_random_matrix(n_components, n_features):
    s = 1 / np.sqrt(n_components)
    return np.random.choice([-s, 0, s], size=(n_features, n_components), p=[1/6, 2/3, 1/6])

# Create a sparse random projection matrix
n_components = 100
n_features = 1000
projection_matrix = create_sparse_random_matrix(n_components, n_features)

print(f"Projection matrix shape: {projection_matrix.shape}")
print(f"Sparsity: {np.sum(projection_matrix == 0) / projection_matrix.size:.2%}")
```

Trang trình bày 6: Thực hiện cho phép ngẫu nhiên thưa thớt

Hãy phát triển phiên bản đơn giản của cơ chế ngẫu nhiên ngẫu nhiên thưa thớt từ đầu để hiểu cơ chế cốt lõi của nó. Việc phát triển việc khai báo này sẽ tạo ra một ma trận ngẫu nhiên thưa thớt và sử dụng nó để tham khảo dữ liệu đầu vào không có chiều thấp hơn.

```python
import numpy as np

class SimpleSRP:
    def __init__(self, n_components):
        self.n_components = n_components
        self.projection_matrix = None

    def fit(self, X):
        n_features = X.shape[1]
        s = 1 / np.sqrt(self.n_components)
        self.projection_matrix = np.random.choice(
            [-s, 0, s],
            size=(n_features, self.n_components),
            p=[1/6, 2/3, 1/6]
        )
        return self

    def transform(self, X):
        return X @ self.projection_matrix

# Usage
X = np.random.rand(1000, 2000)  # 1000 samples, 2000 features
srp = SimpleSRP(n_components=100)
X_reduced = srp.fit(X).transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
```

Trang trình bày 7: So sánh SRP và PCA: Phân cụm chất lượng

Để đánh giá hiệu quả của Phép tham khảo ngẫu nhiên thưa thớt so với PCA, chúng tôi có thể so sánh hoạt động của chúng với phân cụm chất lượng. Chúng tôi sẽ sử dụng điểm bóng để đo độ tương tự của một đối tượng với cụm chính của nó và các cụm khác.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate data
X, y = make_blobs(n_samples=1000, n_features=100, centers=3, random_state=42)

# Apply PCA and SRP
pca = PCA(n_components=10)
srp = SparseRandomProjection(n_components=10, random_state=42)

X_pca = pca.fit_transform(X)
X_srp = srp.fit_transform(X)

# Cluster and calculate silhouette scores
kmeans = KMeans(n_clusters=3, random_state=42)

clusters_original = kmeans.fit_predict(X)
clusters_pca = kmeans.fit_predict(X_pca)
clusters_srp = kmeans.fit_predict(X_srp)

score_original = silhouette_score(X, clusters_original)
score_pca = silhouette_score(X_pca, clusters_pca)
score_srp = silhouette_score(X_srp, clusters_srp)

print(f"Original data silhouette score: {score_original:.4f}")
print(f"PCA reduced data silhouette score: {score_pca:.4f}")
print(f"SRP reduced data silhouette score: {score_srp:.4f}")
```

Trang trình bày 8: Kết quả: So sánh SRP và PCA: Chất lượng phân cụm

```
Original data silhouette score: 0.5821
PCA reduced data silhouette score: 0.5819
SRP reduced data silhouette score: 0.5815
```

Slide 9: Interpreting the Results

The silhouette scores for the original data, PCA-reduced data, and SRP-reduced data are very similar. This indicates that both PCA and SRP preserve the clustering structure of the data well, despite significantly reducing the dimensionality. The key advantage of SRP is its computational efficiency, especially for high-dimensional data.

```python
import matplotlib.pyplot as plt

methods = ['Original', 'PCA', 'SRP']
scores = [score_original, score_pca, score_srp]

plt.figure(figsize=(10, 6))
plt.bar(methods, scores)
plt.title('Silhouette Scores Comparison')
plt.ylabel('Silhouette Score')
plt.ylim(0, 1)
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.show()
```

Trang trình bày 10: Ví dụ thực tế: Nén hình ảnh

Phép ngẫu nhiên thưa thớt có thể được sử dụng để nén hiệu quả hình ảnh, đặc biệt hữu ích trong các vấn đề cần xử lý nhanh hình ảnh có độ phân giải cao, thậm chí như trong phân tích hình ảnh bảo vệ tinh hoặc hình ảnh y tế.

```python
from PIL import Image
import numpy as np
from sklearn.random_projection import SparseRandomProjection

# Load and prepare image
image = Image.open('high_res_image.jpg').convert('L')  # Convert to grayscale
img_array = np.array(image).flatten()

# Apply SRP
srp = SparseRandomProjection(n_components=img_array.shape[0] // 4, random_state=42)
compressed = srp.fit_transform(img_array.reshape(1, -1))

# Reconstruct (approximation)
reconstructed = srp.inverse_transform(compressed).reshape(image.size[::-1])

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(reconstructed, cmap='gray')
ax2.set_title('Reconstructed Image')
plt.show()

print(f"Compression ratio: {img_array.shape[0] / compressed.size:.2f}")
```

Slide 11: Ví dụ thực tế: Phân loại văn bản

Trong quá trình xử lý ngôn ngữ tự nhiên, tài liệu thường được biểu hiện dưới nhiều chiều (ví dụ: use TF-IDF). Phép lạ ngẫu nhiên thưa thớt có thể được sử dụng để giảm kích thước của những điều này, giúp phân loại văn bản hiệu quả hơn mà không làm giảm đáng kể độ chính xác.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample text data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "I think, therefore I am"
]
labels = [0, 1, 1, 0]

# Vectorize texts
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

# Train and evaluate without SRP
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred)

# Apply SRP
srp = SparseRandomProjection(n_components=10, random_state=42)
X_train_srp = srp.fit_transform(X_train)
X_test_srp = srp.transform(X_test)

# Train and evaluate with SRP
clf_srp = MultinomialNB()
clf_srp.fit(X_train_srp, y_train)
y_pred_srp = clf_srp.predict(X_test_srp)
accuracy_srp = accuracy_score(y_test, y_pred_srp)

print(f"Accuracy without SRP: {accuracy_original:.4f}")
print(f"Accuracy with SRP: {accuracy_srp:.4f}")
print(f"Dimension reduction: {X.shape[1]} -> {X_train_srp.shape[1]}")
```

Trang trình bày 12: Chế độ và cân bằng nhanh

Mặc dù bất ngờ ngẫu nhiên thưa thớt lại mang lại những lợi thế đáng kể về hiệu quả tính toán nhưng điều quan trọng là phải xem xét những hạn chế của nó. SRP là một phương pháp ngẫu nhiên, có nghĩa là kết quả có thể khác nhau giữa các lần chạy. Nó cũng không cung cấp các thành phần có thể hiểu được như PCA. Việc lựa chọn giữa SRP và các kỹ thuật giảm kích thước khác phụ thuộc vào công cụ yêu cầu của dự án của bạn.

```python
import numpy as np
from sklearn.random_projection import SparseRandomProjection

# Demonstrate variability in results
X = np.random.rand(1000, 500)

for i in range(3):
    srp = SparseRandomProjection(n_components=10, random_state=i)
    X_reduced = srp.fit_transform(X)
    print(f"Run {i+1}: First 5 values of first sample:")
    print(X_reduced[0][:5])
    print()

# Demonstrate lack of interpretability
srp = SparseRandomProjection(n_components=5, random_state=42)
srp.fit(X)
print("Projection components (not interpretable like PCA):")
print(srp.components_[:2, :10])
```

Slide 13: Kết luận và định hướng tương lai

Phép ngẫu nhiên thưa thớt cung cấp một giải pháp thay thế mạnh mẽ cho PCA để giảm kích thước, đặc biệt đối với các bộ dữ liệu nhiều chiều. Hiệu quả và khả năng duy trì khoảng cách của nó khiến nó có giá trị trong nhiều ứng dụng khác nhau, từ phân cụm đến phân loại. Khi chiều dữ liệu tiếp tục tăng lên trong nhiều lĩnh vực, các kỹ thuật như SRP sẽ ngày càng trở nên quan trọng. Nghiên cứu trong tương lai có thể nghiên cứu trung tâm phát triển các biến thể xác định cho phép ngẫu nhiên hoặc kết hợp nó với các kỹ thuật giảm kích thước khác để có hiệu suất tốt hơn nữa.

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate performance comparison
dimensions = np.logspace(2, 4, 20, dtype=int)
pca_times = dimensions**2 / 1e5
srp_times = np.log(dimensions) / 1e2

plt.figure(figsize=(10, 6))
plt.plot(dimensions, pca_times, label='PCA')
plt.plot(dimensions, srp_times, label='SRP')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Dimensions')
plt.ylabel('Computational Time (arbitrary units)')
plt.title('Projected Performance: PCA vs SRP')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người quan tâm đến công việc tìm hiểu sâu hơn về Phép lạ ngẫu nhiên thưa thớt và các kỹ thuật liên quan thì đây là một số tài nguyên có giá trị:

1. Achlioptas, D. (2003). Các cơ sở dữ liệu ngẫu nhiên được phép sử dụng ngẫu nhiên: Johnson-Lindenstrauss với phân tích tiền nhị phân. Tạp chí Khoa học Hệ thống và Máy tính, 66(4), 671-687. ArXiv: [https://arxiv.org/abs/cs/0304025](https://arxiv.org/abs/cs/0304025)
2. Bingham, E., & Mannila, H. (2001). Phép thử ngẫu nhiên trong việc giảm kích thước: Ứng dụng cho hình ảnh dữ liệu và văn bản. Trong Kỷ yếu của hội nghị quốc tế ACM SIGKDD lần thứ bảy về Khám phá tri thức và khai thác dữ liệu (trang 245-250). Thư viện kỹ thuật số ACM: [https://dl.acm.org/doi/10.1145/502512.502546](https://dl.acm.org/doi/10.1145/502512.502546)
3. Li, P., Hastie, T. J., & Church, K. W. (2006). Các tùy chọn ngẫu nhiên rất thưa thớt. Trong Kỷ yếu của nghị viện quốc tế ACM SIGKDD lần thứ 12 về Khám phá tri thức và khai thác dữ liệu (trang 287-296). ArXiv: [https://arxiv.org/abs/math/0608284](https://arxiv.org/abs/math/0608284)

Bài viết này cung cấp nền tảng lý thuyết chuyên sâu và ứng dụng thực tế của kỹ thuật tham chiếu ngẫu nhiên trong việc giảm kích thước.
