## Tìm hiểu Điểm Silhouette trong Phân cụm

Trang trình bày 1: Tìm hiểu Điểm Silhouette trong Phân cụm

Điểm bóng là thước đo được sử dụng để đánh giá chất lượng của kết quả phân cụm. Nó đo lường độ tương tự của một đối tượng với cụm chính của nó cũng như các cụm khác, cung cấp cái nhìn sâu sắc về phân tách và gắn kết của các cụm.

```python
import numpy as np
from sklearn.cluster import KMeans

# Generate sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

# Calculate silhouette score
silhouette_avg = silhouette_score(X, labels)
print(f"The average silhouette score is: {silhouette_avg:.3f}")
```

Trang trình bày 2: Giải thích Điểm Silhouette

Điểm bóng dao động từ -1 đến 1. Điểm cao hơn biểu thức các cụm được xác định rõ hơn. Điểm gần 0 mũi cụm chéo, trong khi điểm âm cho khả năng phân loại sai được tìm thấy.

```python
from sklearn.datasets import make_blobs

# Generate sample data with 3 clusters
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.plot(range(2, 10), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()
```

Trang trình bày 3: Tính giá trị riêng lẻ của bóng

Điểm bóng cho mỗi cung cấp dữ liệu được cung cấp thông tin chi tiết về mức độ phù hợp của nó trong cụm được chỉ định.

```python

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Calculate silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, cluster_labels)

# Print silhouette scores for the first 5 samples
for i in range(5):
    print(f"Sample {i} silhouette score: {sample_silhouette_values[i]:.3f}")
```

Trang trình bày 4: Trực quan hóa các ô bóng

Các biểu đồ bóng bóng cung cấp một biểu tượng đồ họa về chất lượng phân cụm, hiển thị hệ thống số bóng cho từng mẫu.

```python

# Create a subplot with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

# The silhouette plot
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(X) + (3 + 1) * 10])

y_lower = 10
for i in range(3):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 3)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The scatter plot of the data
colors = cm.nipy_spectral(cluster_labels.astype(float) / 3)
ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")

plt.show()
```

Trang trình bày 5: Use Point Silhouette để chọn mẫu Người

Điểm bóng có thể được sử dụng để so sánh các phân cụm thuật toán khác nhau hoặc để xác định số lượng tối ưu của cụm.

```python

# Define clustering algorithms
algorithms = [
    ('K-Means', KMeans(n_clusters=3)),
    ('Agglomerative', AgglomerativeClustering(n_clusters=3)),
    ('DBSCAN', DBSCAN(eps=0.5, min_samples=5))
]

# Compare algorithms using silhouette score
for name, algorithm in algorithms:
    labels = algorithm.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"{name} Silhouette Score: {score:.3f}")
```

Slide 6: Xử lý chiều cao dữ liệu

Khi làm việc với dữ liệu nhiều chiều, kỹ thuật giảm kích thước có thể được áp dụng trước khi tính điểm bóng.

```python
from sklearn.preprocessing import StandardScaler

# Generate high-dimensional data
X_high_dim = np.random.rand(100, 50)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_high_dim)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Perform clustering on reduced data
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_reduced)

# Calculate silhouette score
score = silhouette_score(X_reduced, labels)
print(f"Silhouette Score on reduced data: {score:.3f}")
```

Trang trình bày 7: Ví dụ thực tế: Phân khúc khách hàng

Trong ví dụ này, chúng tôi sẽ sử dụng điểm bóng để đánh giá phân khúc khách hàng dựa trên hành vi mua hàng của họ.

```python

# Create sample customer data
data = {
    'Customer_ID': range(1, 101),
    'Recency': np.random.randint(1, 100, 100),
    'Frequency': np.random.randint(1, 50, 100),
    'Monetary': np.random.randint(100, 1000, 100)
}
df = pd.DataFrame(data)

# Normalize the features
X = StandardScaler().fit_transform(df[['Recency', 'Frequency', 'Monetary']])

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Calculate silhouette score
score = silhouette_score(X, df['Cluster'])
print(f"Silhouette Score for customer segmentation: {score:.3f}")

# Visualize clusters
plt.scatter(df['Recency'], df['Frequency'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Customer Segments')
plt.colorbar(label='Cluster')
plt.show()
```

Trang trình bày 8: Ví dụ thực tế: Phân đoạn hình ảnh

Trong ví dụ này, chúng tôi sẽ sử dụng điểm bóng để đánh giá kết quả phân đoạn hình ảnh.

```python
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# Load and preprocess the image
image = io.imread('sample_image.jpg')
image_lab = color.rgb2lab(image)
image_array = image_lab.reshape((-1, 3))

# Perform K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(image_array)

# Calculate silhouette score
score = silhouette_score(image_array, labels)
print(f"Silhouette Score for image segmentation: {score:.3f}")

# Visualize segmented image
segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape)
segmented_image = color.lab2rgb(segmented_image)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(segmented_image)
plt.title(f'Segmented Image (K={n_clusters})')
plt.axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 9: Các chế độ của bóng bóng

Mặc dù điểm bóng rất hữu ích nhưng không có một số chế độ:

1. Nó giả sử các cụm lồi, điều này có thể không đúng lúc nào trong dữ liệu trong thế giới thực.
2. Công việc tính toán có thể được tính toán hợp lý hơn đối với các dữ liệu lớn.
3. Nó có thể không hoạt động tốt với các cơ sở mật khẩu có mật độ khác nhau.

```python

# Generate non-convex data
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

# Calculate silhouette score for K-means
score_kmeans = silhouette_score(X, labels_kmeans)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# Calculate silhouette score for DBSCAN
score_dbscan = silhouette_score(X, labels_dbscan)

print(f"K-means Silhouette Score: {score_kmeans:.3f}")
print(f"DBSCAN Silhouette Score: {score_dbscan:.3f}")

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis')
ax1.set_title(f'K-means Clustering\nSilhouette Score: {score_kmeans:.3f}')

ax2.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis')
ax2.set_title(f'DBSCAN Clustering\nSilhouette Score: {score_dbscan:.3f}')

plt.tight_layout()
plt.show()
```

Trang trình bày 10: Các lựa chọn thay thế cho Điểm Silhouette

Mặc dù điểm bóng rất phổ biến nhưng các dữ liệu khác có thể được bổ sung hoặc thay thế nó trong một số trường hợp nhất:

1. Chỉ số Calinski-Harabasz
2. Chỉ số Davies-Bouldin
3. Chỉ số Dunn

```python

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Calculate different metrics
silhouette = silhouette_score(X, labels)
calinski_harabasz = calinski_harabasz_score(X, labels)
davies_bouldin = davies_bouldin_score(X, labels)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")

# Note: A higher Calinski-Harabasz score and a lower Davies-Bouldin score indicate better clustering.
```

Slide 11: Tối ưu hóa các tham số phân cụm

Điểm bóng có thể được sử dụng để tối ưu hóa các phân số tham số, suy ra giới hạn như số lượng hoặc epsilon giá trị trong DBSCAN.

```python
from sklearn.metrics import make_scorer

# Define parameter grid for KMeans
param_grid = {'n_clusters': range(2, 11)}

# Create a scorer using silhouette_score
silhouette_scorer = make_scorer(silhouette_score, metric='euclidean')

# Perform grid search
grid_search = GridSearchCV(KMeans(), param_grid, scoring=silhouette_scorer, cv=5)
grid_search.fit(X)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best silhouette score:", grid_search.best_score_)

# Plot silhouette scores for different numbers of clusters
plt.plot(param_grid['n_clusters'], grid_search.cv_results_['mean_test_score'])
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()
```

Slide 12: Xử lý các cụm mất cân bằng

Điểm bóng có thể bị ảnh hưởng bởi các cụm mất cân bằng. Đây là một ví dụ về cách giải quyết vấn đề này:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt

# Generate imbalanced data
X, y = make_blobs(n_samples=[50, 300, 500], centers=3, random_state=42)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Calculate silhouette score for imbalanced data
score_imbalanced = silhouette_score(X, labels)
print(f"Silhouette Score (Imbalanced): {score_imbalanced:.3f}")

# Balance the clusters through resampling
X_balanced = []
y_balanced = []
for label in np.unique(labels):
    X_label = X[labels == label]
    X_resampled, _ = resample(X_label, n_samples=300, random_state=42)
    X_balanced.extend(X_resampled)
    y_balanced.extend([label] * 300)

X_balanced = np.array(X_balanced)
y_balanced = np.array(y_balanced)

# Recalculate silhouette score for balanced data
score_balanced = silhouette_score(X_balanced, y_balanced)
print(f"Silhouette Score (Balanced): {score_balanced:.3f}")

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
ax1.set_title(f'Imbalanced Clusters\nSilhouette Score: {score_imbalanced:.3f}')

ax2.scatter(X_balanced[:, 0], X_balanced[:, 1], c=y_balanced, cmap='viridis')
ax2.set_title(f'Balanced Clusters\nSilhouette Score: {score_balanced:.3f}')

plt.tight_layout()
plt.show()
```

Trang trình bày 13: Điểm hình bóng trong phân cụm theo cấp bậc

Điểm bóng cũng có thể được sử dụng để đánh giá các kết quả phân cụm theo cấp bậc và xác định số lượng tối ưu của cụm.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
n_clusters_range = range(2, 11)

for n_clusters in n_clusters_range:
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.plot(n_clusters_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters in Hierarchical Clustering')
plt.show()

# Find the optimal number of clusters
optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_clusters}")
```

Trang trình bày 14: Điểm bóng cho phân cụm mờ

Điểm bóng có thể được điều chỉnh phù hợp với các cụm phân tích thuật toán như Fuzzy C-Means, trong đó mỗi dữ liệu có các thành phần của nhiều cụm.

```python
from sklearn.datasets import make_blobs
from skfuzzy import cmeans
from sklearn.metrics import silhouette_score

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Perform Fuzzy C-Means clustering
n_clusters = 3
cntr, u, u0, d, jm, p, fpc = cmeans(X.T, n_clusters, 2, error=0.005, maxiter=1000)

# Convert fuzzy membership to hard clustering for silhouette score calculation
labels = np.argmax(u, axis=0)

# Calculate silhouette score
score = silhouette_score(X, labels)
print(f"Silhouette Score for Fuzzy C-Means: {score:.3f}")

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title(f'Fuzzy C-Means Clustering\nSilhouette Score: {score:.3f}')
plt.colorbar(label='Cluster')
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin về điểm bóng và đánh giá phân cụm:

1. Rousseeuw, P. J. (1987). Bóng: Hỗ trợ đồ họa để giải thích và xác định phân cụm. Tạp chí Toán học tính toán và ứng dụng, 20, 53-65. URL ArXiv: [https://arxiv.org/abs/2109.07317](https://arxiv.org/abs/2109.07317)
2. Arbelaitz, O., Gurrutxaga, I., Muguerza, J., Pérez, J. M., & Perona, I. (2013). Một nghiên cứu so sánh độ sâu về các số hiệu của cụm. Đã nhận mẫu dạng, 46(1), 243-256. URL ArXiv: [https://arxiv.org/abs/1901.10493](https://arxiv.org/abs/1901.10493)

Tài nguyên này cung cấp các cuộc thảo luận chuyên sâu về phân tích các dữ liệu đánh giá và ứng dụng của chúng trong các lĩnh vực khác nhau.
