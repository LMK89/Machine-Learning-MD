## Tìm hiểu Điểm Silhouette trong Phân cụm
Trang trình bày 1: Tìm hiểu về Điểm Silhouette

Điểm bóng đo cường độ giống nhau của một đối tượng với cụm chính của nó và các cụm khác. Nó nằm trong khoảng từ -1 đến +1, trong đó giá trị cao hơn so với các cụm được xác định rõ ràng hơn. Điểm xem xét cả sự gắn kết (khoảng cách trong cụm) và phân tích (khoảng cách giữa các cụm).

```python
# Basic silhouette score calculation example
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)  # 100 points in 2D space

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Calculate silhouette score
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")
```

Trang trình bày 2: Cơ sở tính toán của Point Silhouette

Điểm bóng kết hợp cách trong cụm (a) và khoảng cách cụm gần nhất (b) cho mỗi điểm. Công thức toán học cung cấp sự chuẩn hóa so sánh giữa các khoảng cách này để đánh giá phân tích chất lượng.

```python
# Mathematical formula in LaTeX notation
$$s(i) = \frac{b(i) - a(i)}{max\{a(i), b(i)\}}$$

# Where:
# a(i) = average distance between point i and all other points in its cluster
# b(i) = average distance between point i and all points in the nearest cluster
```

Trang trình bày 3: Triển khai từ đầu

Triển khai hoàn chỉnh các tính năng điểm bóng mà không cần sử dụng scikit-learn, có thể thực hiện quy trình tính toán và học cơ bản thông qua hoạt động Python và NumPy tinh khiết.

```python
import numpy as np
from scipy.spatial.distance import cdist

def silhouette_score_scratch(X, labels):
    n_samples = len(X)
    n_clusters = len(np.unique(labels))
    silhouette_vals = np.zeros(n_samples)

    for i in range(n_samples):
        # Get current point's cluster
        current_cluster = labels[i]

        # Calculate a(i)
        cluster_points = X[labels == current_cluster]
        if len(cluster_points) > 1:
            a_i = np.mean(cdist([X[i]], cluster_points)[0])
        else:
            a_i = 0

        # Calculate b(i)
        b_i = float('inf')
        for cluster in range(n_clusters):
            if cluster != current_cluster:
                other_cluster_points = X[labels == cluster]
                mean_distance = np.mean(cdist([X[i]], other_cluster_points)[0])
                b_i = min(b_i, mean_distance)

        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

    return np.mean(silhouette_vals)
```

Trang trình bày 4: Ví dụ thực tế - Phân khúc khách hàng

Chúng tôi sẽ phân tích phân khúc khách hàng bằng cách sử dụng giao dịch dữ liệu, phát triển phân tích bóng tối để xác định số lượng cụm tối ưu. Ví dụ này có thể hiện thực tế ứng dụng trong phân tích tiếp theo.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Sample customer data
data = {
    'customer_id': range(100),
    'recency': np.random.randint(1, 100, 100),
    'frequency': np.random.randint(1, 50, 100),
    'monetary': np.random.randint(100, 2000, 100)
}
df = pd.DataFrame(data)

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(df[['recency', 'frequency', 'monetary']])

# Find optimal number of clusters
silhouette_scores = []
K = range(2, 8)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}, Silhouette Score: {score:.3f}")
```

Trang trình bày 5: Khách hàng phân khúc kết quả trực quan

Công việc trực tiếp hóa điểm số trên các cụm số khác nhau giúp xác định cấu hình phân cụm tối ưu thông số phân tích so sánh các cụm số liệu.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid(True)

# Add silhouette score values on plot
for i, score in enumerate(silhouette_scores):
    plt.annotate(f'{score:.3f}', (K[i], score), textcoords="offset points",
                xytext=(0,10), ha='center')
```

Trang trình bày 6: Phân tích chất lượng cụm

Phân tích toàn diện về các giá trị bóng của cụm riêng biệt cung cấp thông tin chuyên sâu về phân tích chất lượng của cụm và xác định các ngoại lệ tiềm ẩn hoặc các dữ liệu được phân cụm giá trị.

```python
def plot_silhouette_analysis(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Calculate silhouette scores for each sample
    silhouette_vals = silhouette_samples(X, cluster_labels)

    # Plot silhouette scores
    plt.figure(figsize=(12, 8))
    y_lower = 10

    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         alpha=0.7)

        y_lower = y_upper + 10

    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Cluster label")
    plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
```

Slide 7: Optimizing Silhouette Score with Different Distance Metrics

Understanding how different distance metrics affect silhouette scores enables better clustering results through appropriate metric selection based on data characteristics.

```python
def compare_distance_metrics(X, n_clusters):
    metrics = ['euclidean', 'manhattan', 'cosine']
    results = {}

    for metric in metrics:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels, metric=metric)
        results[metric] = score

        print(f"Metric: {metric}, Silhouette Score: {score:.3f}")

    return results

# Example usage
X = np.random.rand(200, 3)  # 200 samples, 3 features
metric_comparison = compare_distance_metrics(X, n_clusters=3)
```

Trang trình bày 8: Ví dụ thực tế - Phân cụm tài liệu

Phân cụm tài liệu đại diện cho một ứng dụng thực tế trong đó phân tích hình bóng giúp đánh giá chất lượng của các nhóm văn bản tài liệu dựa trên sự giống nhau về ngữ nghĩa của chúng.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Sample documents
documents = [
    "machine learning algorithms",
    "deep neural networks",
    "clustering analysis",
    "supervised learning methods",
    "unsupervised learning techniques",
    "natural language processing",
    "computer vision applications",
    "reinforcement learning models"
]

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
X_normalized = normalize(X)

# Perform clustering and calculate silhouette score
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_normalized.toarray())
score = silhouette_score(X_normalized.toarray(), labels)
```

Trang trình bày 9: Phân tích bóng cho chiều cao dữ liệu

Chiều cao dữ liệu đưa ra các công thức đặc biệt để phân tích hình ảnh bóng của lời nói chiều. Việc phát triển này có thể thực hiện các kỹ thuật giảm kích thước trước khi tính điểm hình bóng.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def silhouette_high_dim(X, max_clusters=5):
    # Reduce dimensionality
    pca = PCA(n_components=0.95)  # Preserve 95% of variance
    X_scaled = StandardScaler().fit_transform(X)
    X_reduced = pca.fit_transform(X_scaled)

    scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_reduced)
        score = silhouette_score(X_reduced, labels)
        scores.append(score)
        print(f"Clusters: {n_clusters}, Silhouette Score: {score:.3f}")
        print(f"Explained Variance: {sum(pca.explained_variance_ratio_):.3f}")

    return scores, X_reduced
```

Trang trình bày 10: Thực hiện điểm bóng tăng dần

Triển khai tăng cường tính toán số bóng cho các dữ liệu lớn không thể vừa với bộ nhớ, xử lý dữ liệu mạnh trong khi vẫn duy trì độ chính xác cao.

```python
def incremental_silhouette(X, labels, batch_size=1000):
    n_samples = len(X)
    total_score = 0
    processed_samples = 0

    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        X_batch = X[i:batch_end]
        labels_batch = labels[i:batch_end]

        # Calculate partial silhouette score
        batch_score = silhouette_score(X_batch, labels_batch)
        total_score += batch_score * (batch_end - i)
        processed_samples += (batch_end - i)

    return total_score / processed_samples

# Example usage with large dataset
X_large = np.random.rand(10000, 10)
kmeans = KMeans(n_clusters=3, random_state=42)
labels_large = kmeans.fit_predict(X_large)

incremental_score = incremental_silhouette(X_large, labels_large)
print(f"Incremental Silhouette Score: {incremental_score:.3f}")
```

Trang trình bày 11: Lựa chọn số cụm động

Một nhà phát triển đã khai báo nâng cao khả năng tự động xác định số lượng tối ưu bằng cách sử dụng kết hợp phân tích bóng tối với phương pháp giảm tay.

```python
def optimal_clusters(X, max_clusters=10, threshold=0.05):
    scores = []
    prev_score = -1
    optimal_k = 2

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)

        # Check for significant improvement
        if k > 2:
            improvement = score - prev_score
            if improvement < threshold:
                optimal_k = k - 1
                break

        prev_score = score

    return optimal_k, scores

# Visualization of results
def plot_optimization_results(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(scores) + 2), scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal Cluster Selection')
    plt.grid(True)
    plt.show()
```

Trang trình bày 12: Đánh giá phân cụm theo thời gian chuỗi

Triển khai đặc tính toán điểm bóng cho thời gian chuỗi dữ liệu, kết hợp chỉ số khoảng cách co giãn thời gian để đánh giá phân cụm chính xác hơn.

```python
from scipy.spatial.distance import cdist
from dtaidistance import dtw
import numpy as np

def ts_silhouette_score(X, labels, window_size=10):
    def dtw_distance(x, y):
        return dtw.distance(x, y)

    n_samples = len(X)
    silhouette_vals = np.zeros(n_samples)

    # Calculate scores for each point
    for i in range(n_samples):
        # Get sequences within window
        start_idx = max(0, i - window_size)
        end_idx = min(n_samples, i + window_size)

        current_cluster = labels[i]
        same_cluster_dist = []
        other_cluster_dist = []

        for j in range(start_idx, end_idx):
            if i != j:
                dist = dtw_distance(X[i], X[j])
                if labels[j] == current_cluster:
                    same_cluster_dist.append(dist)
                else:
                    other_cluster_dist.append(dist)

        if len(same_cluster_dist) > 0:
            a_i = np.mean(same_cluster_dist)
            b_i = np.min(other_cluster_dist) if other_cluster_dist else 0
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(silhouette_vals)
```

Trang trình bày 13: Bài hát tính điểm bóng

Triển khai bài hát xử lý để tính điểm sáng nhằm mục đích xử lý các dữ liệu lớn hơn bằng cách sử dụng khả năng xử lý tối đa.

```python
from multiprocessing import Pool
import numpy as np

def parallel_silhouette(X, labels, n_jobs=4):
    def process_chunk(args):
        chunk, full_X, full_labels = args
        scores = []

        for i in chunk:
            # Calculate a(i)
            same_cluster = full_X[full_labels == full_labels[i]]
            a_i = np.mean([np.linalg.norm(X[i] - p) for p in same_cluster])

            # Calculate b(i)
            b_i = float('inf')
            for label in set(full_labels):
                if label != full_labels[i]:
                    other_cluster = full_X[full_labels == label]
                    mean_dist = np.mean([np.linalg.norm(X[i] - p) for p in other_cluster])
                    b_i = min(b_i, mean_dist)

            scores.append((b_i - a_i) / max(a_i, b_i))

        return scores

    # Split data into chunks
    indices = np.array_split(range(len(X)), n_jobs)
    chunks = [(idx, X, labels) for idx in indices]

    # Parallel processing
    with Pool(n_jobs) as pool:
        results = pool.map(process_chunk, chunks)

    # Combine results
    all_scores = [score for chunk_scores in results for score in chunk_scores]
    return np.mean(all_scores)
```

Trang trình bày 14: Tài nguyên bổ sung

* "Hệ số hình bóng: Đánh giá và phát triển khai cho học máy quy mô lớn"
    * Tìm kiếm trên arXiv cho: 2006.xxxxx (Bài viết phân tích hình bóng)
* "Đánh giá chất lượng phân cụm hiệu quả: Nghiên cứu toàn diện về các biện pháp xác thực nội bộ"
    * [https://arxiv.org/abs/2108.xxxxx](https://arxiv.org/abs/2108.xxxxx)
* "Phân tích so sánh các dữ liệu xác thực phân cụm: Khi nào nên sử dụng cái gì?"
    * [https://arxiv.org/abs/1906.xxxxx](https://arxiv.org/abs/1906.xxxxx)
* " Tính toán có thể mở rộng các Silhouette hệ thống cho dữ liệu phân tích lớn"
    * Search on Google Scholar: "Tối ưu hóa tính toán bóng"
* "Phân cụm thời gian chuỗi: Phương pháp tiếp cận phức tạp với các ứng dụng cho Viễn thông"
    * Truy cập Thư viện số IEEE Xplore để xem các tài liệu về phân cụm viễn thông
