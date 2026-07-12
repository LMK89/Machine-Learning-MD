## Độ phức tạp về thời gian của phân cụm K-Means trong Python
Trang trình bày 1: Phân cụm K-nghĩa là: Phân tích phức tạp về thời gian

Phân cụm K-mean là một thuật toán học máy không giám sát phổ biến được sử dụng để phân vùng dữ liệu thành nhóm K hoặc cụm riêng biệt, không chéo. Hiểu được sự phức tạp về thời gian của nó là rất quan trọng để phát triển hiệu quả khai thác và khả năng mở rộng. Hãy cùng khám phá phức tạp về thời gian thuật toán bằng các ví dụ Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Clustering Example')
plt.show()
```

Slide 2: Tổng quan về thuật toán K-mean

Thuật toán K-nghĩa là phân bổ lại các điểm dữ liệu cho các cụm và cập nhật cụm. Các bước khởi động chính bao gồm việc tạo, phân công và cập nhật. Hãy phát triển một phiên bản đơn giản của K-mean để hiểu các thành phần cốt lõi của nó.

```python
def kmeans(X, k, max_iters=100):
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

# Example usage
X = np.random.rand(100, 2)
labels, centroids = kmeans(X, k=3)
```

Trang trình bày 3: Độ phức tạp về thời gian: Khởi tạo

Bước khởi động tạo bao gồm các công việc chọn K điểm ngẫu nhiên làm tâm ban đầu. Quá trình này có phức tạp về thời gian là O(K), trong đó K là cụm số.

```python
def initialize_centroids(X, k):
    n_samples = X.shape[0]
    centroid_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[centroid_indices]
    return centroids

# Example usage
X = np.random.rand(1000, 2)
k = 5
initial_centroids = initialize_centroids(X, k)
print(f"Shape of initial centroids: {initial_centroids.shape}")
```

Trang trình bày 4: Độ phức tạp về thời gian: Bước phân công

Bước phân công tính toán khoảng cách giữa mỗi điểm dữ liệu và tất cả các tâm, sau đó phân bổ từng điểm cho tâm gần nhất. Bước này có tốc độ phức tạp về thời gian là O(n \* K \* d), trong đó n là dữ liệu số, K là cụm số và d là số nguyên.

```python
def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    return labels

# Example usage
X = np.random.rand(1000, 2)
centroids = np.random.rand(5, 2)
labels = assign_clusters(X, centroids)
print(f"Number of points in each cluster: {np.bincount(labels)}")
```

Trang trình bày 5: Độ phức tạp về thời gian: Bước cập nhật

Bước cập nhật sẽ tính toán lại trọng tâm dựa trên giá trị trung bình của tất cả các điểm được phân bổ cho mỗi cụm. Bước này có tốc độ phức tạp về thời gian là O(n \* d), trong đó n là dữ liệu số và d là số nguyên.

```python
def update_centroids(X, labels, k):
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

# Example usage
X = np.random.rand(1000, 2)
labels = np.random.randint(0, 5, 1000)
k = 5
new_centroids = update_centroids(X, labels, k)
print(f"Shape of updated centroids: {new_centroids.shape}")
```

Trang trình bày 6: Độ phức tạp về tổng thể thời gian

Độ phức tạp về tổng thể thời gian của K-means là O(n \* K \* d \* I), trong đó:

*n: data point number
* K: số cụm
*d: số chiều
* I: số lần lặp

Công việc phức tạp này được phát ra từ việc lặp lại các bước được chỉ định và cập nhật cho vòng lặp I. Hãy hình dung thời gian thực hiện thay đổi như thế nào với các tham số khác nhau.

```python
import time

def measure_kmeans_time(n, k, d, max_iters):
    X = np.random.rand(n, d)
    start_time = time.time()
    kmeans(X, k, max_iters)
    end_time = time.time()
    return end_time - start_time

n_values = [1000, 2000, 4000, 8000]
times = [measure_kmeans_time(n, k=5, d=2, max_iters=100) for n in n_values]

plt.plot(n_values, times, marker='o')
plt.xlabel('Number of data points (n)')
plt.ylabel('Execution time (seconds)')
plt.title('K-means Execution Time vs. Number of Data Points')
plt.show()
```

Slide 7: Tác động của số lượng (K)

Số lượng cụm (K) ảnh đáng kể đến mức độ phức tạp về thời gian. Vui lòng xem xét việc tăng cường ảnh hưởng như thế nào đến thời điểm thực hiện trong khi vẫn giữ các tham số khác không thay đổi.

```python
k_values = [2, 4, 8, 16, 32]
times = [measure_kmeans_time(n=5000, k=k, d=2, max_iters=100) for k in k_values]

plt.plot(k_values, times, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Execution time (seconds)')
plt.title('K-means Execution Time vs. Number of Clusters')
plt.show()
```

Slide 8: Tác động của số lượng kích thước (d)

Số chiều (d) cũng ảnh hưởng đến độ phức tạp về thời gian. Vui lòng hiển thị các tác động tăng cường của ảnh như thế nào đến thời gian thực hiện trong khi vẫn giữ các tham số khác không thay đổi.

```python
d_values = [2, 4, 8, 16, 32]
times = [measure_kmeans_time(n=5000, k=5, d=d, max_iters=100) for d in d_values]

plt.plot(d_values, times, marker='o')
plt.xlabel('Number of dimensions (d)')
plt.ylabel('Execution time (seconds)')
plt.title('K-means Execution Time vs. Number of Dimensions')
plt.show()
```

Trang trình bày 9: Tối ưu hóa K-nghĩa: Thuật toán Elkan

Thuật toán Elkan là phiên bản được tối ưu hóa của K-mean giúp giảm số lượng phép tính khoảng cách, có khả năng cải thiện phức tạp về thời gian. Nó sử dụng bất đẳng thức tam giác để tránh các cách tính toán không cần thiết.

```python
from sklearn.cluster import KMeans

def elkan_kmeans(X, k, max_iters=100):
    kmeans = KMeans(n_clusters=k, algorithm='elkan', max_iter=max_iters, n_init=1)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

# Compare standard K-means with Elkan K-means
X = np.random.rand(10000, 10)
k = 5

start_time = time.time()
kmeans(X, k, max_iters=100)
standard_time = time.time() - start_time

start_time = time.time()
elkan_kmeans(X, k, max_iters=100)
elkan_time = time.time() - start_time

print(f"Standard K-means time: {standard_time:.4f} seconds")
print(f"Elkan K-means time: {elkan_time:.4f} seconds")
print(f"Speedup: {standard_time / elkan_time:.2f}x")
```

Trang trình bày 10: Ví dụ thực tế: Nén hình ảnh

Phân cụm K-mean có thể được sử dụng để nén hình ảnh bằng cách giảm số lượng màu trong hình ảnh. Hãy phát triển một trình nén hình ảnh đơn giản bằng K-mean.

```python
from PIL import Image

def compress_image(image_path, k):
    # Load image and convert to numpy array
    img = Image.open(image_path)
    img_array = np.array(img)

    # Reshape the image to 2D array of pixels
    pixels = img_array.reshape(-1, 3)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Replace each pixel with its nearest centroid
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape back to original image shape
    compressed_img_array = compressed_pixels.reshape(img_array.shape)

    # Convert to uint8 and create new image
    compressed_img = Image.fromarray(compressed_img_array.astype('uint8'))
    return compressed_img

# Example usage
original_image_path = 'path_to_your_image.jpg'
compressed_image = compress_image(original_image_path, k=16)
compressed_image.save('compressed_image.jpg')

# Display original and compressed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(Image.open(original_image_path))
ax1.set_title('Original Image')
ax2.imshow(compressed_image)
ax2.set_title('Compressed Image (16 colors)')
plt.show()
```

Trang trình bày 11: Ví dụ thực tế: Phân khúc khách hàng

Phân cụm K-mean được sử dụng rộng rãi trong phân khúc khách hàng để phân nhóm khách hàng dựa trên hành vi hoặc đặc điểm của họ. Hãy thực hiện một ví dụ phân khúc khách hàng đơn giản.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate sample customer data
np.random.seed(42)
n_customers = 1000
age = np.random.randint(18, 70, n_customers)
income = np.random.randint(20000, 200000, n_customers)
spending_score = np.random.randint(1, 100, n_customers)

df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'SpendingScore': spending_score
})

# Normalize the features
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_normalized)

# Visualize the results
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Age'], df['Income'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Spending Score')
plt.title('Customer Segmentation using K-means')
plt.colorbar(scatter)
plt.show()

# Print cluster statistics
print(df.groupby('Cluster').mean())
```

Slide 12: Các công thức và chế độ giới hạn

Mặc dù K-means được sử dụng rộng rãi nhưng không có một số chế độ:

1. Độ nhạy cảm của tâm trí ban đầu: Thuật toán có thể hội tụ về điểm tối ưu cục bộ.
2. Số cụm được xác định trước: Công cụ xác định tối ưu K có thể là một phương thức.
3. Giả sử các cụm hình cầu: K-mean có thể hoạt động nguy hiểm trên các cụm có kích thước không cầu hoặc không đồng đều.

Vui lòng giải thích những giới hạn này bằng một ví dụ đơn giản.

```python
from sklearn.datasets import make_blobs, make_moons

# Generate datasets
n_samples = 1000
blob_centers = [(0, 0), (5, 5), (0, 5)]
X_blobs, _ = make_blobs(n_samples=n_samples, centers=blob_centers, cluster_std=0.7)
X_moons, _ = make_moons(n_samples=n_samples, noise=0.1)

# Perform K-means clustering
kmeans_blobs = KMeans(n_clusters=3, random_state=42)
kmeans_moons = KMeans(n_clusters=2, random_state=42)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_blobs[:, 0], X_blobs[:, 1], c=kmeans_blobs.fit_predict(X_blobs), cmap='viridis')
ax1.set_title('K-means on Blob Dataset')

ax2.scatter(X_moons[:, 0], X_moons[:, 1], c=kmeans_moons.fit_predict(X_moons), cmap='viridis')
ax2.set_title('K-means on Moon Dataset')

plt.show()
```

Trang trình bày 13: Cải thiện K-mean: Khởi tạo K-mean++

K-means++ là một phương pháp khởi động nhằm mục tiêu chọn các tâm trí tốt hơn, có khả năng dẫn đến hội tụ nhanh hơn và kết quả phân cụm tốt hơn. Vui lòng so sánh tiêu chuẩn K-mean với K-means++.

```python
from sklearn.cluster import KMeans

# Generate sample data
X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

# Standard K-means
kmeans_standard = KMeans(n_clusters=5, init='random', n_init=10, random_state=42)
kmeans_standard.fit(X)

# K-means++
kmeans_plus_plus = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
kmeans_plus_plus.fit(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=kmeans_standard.labels_, cmap='viridis')
ax1.scatter(kmeans_standard.cluster_centers_[:, 0], kmeans_standard.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
ax1.set_title('Standard K-means')

ax2.scatter(X[:, 0], X[:, 1], c=kmeans_plus_plus.labels_, cmap='viridis')
ax2.scatter(kmeans_plus_plus.cluster_centers_[:, 0], kmeans_plus_plus.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
ax2.set_title('K-means++')

plt.show()

print(f"Standard K-means inertia: {kmeans_standard.inertia_:.2f}")
print(f"K-means++ inertia: {kmeans_plus_plus.inertia_:.2f}")
```

Trang trình bày 14: Kết luận và các phương pháp hay nhất

Để tối ưu hóa phân cụm K-mean:

1. Sử dụng khởi động K-means++ để có tâm trí tốt hơn.
2. Chuẩn hóa các tính năng để đảm bảo chất lượng cho nhau.
3. Chạy nhiều lần khởi động để tránh cục bộ tối ưu.
4. Sử dụng phương pháp thu nhỏ hoặc phân tích bóng bóng để xác định mức độ tối ưu.
5. Cân nhắc sử dụng K-means theo dõi thu nhỏ cho các dữ liệu lớn.

Dưới đây là ví dụ phát triển các phương pháp hay nhất này:

Trang trình bày 15: Kết luận và các phương pháp hay nhất

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Generate sample data
X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Find optimal K using silhouette analysis
silhouette_scores = []
K_range = range(2, 11)

for K in K_range:
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_normalized)
    silhouette_scores.append(silhouette_score(X_normalized, kmeans.labels_))

optimal_K = K_range[silhouette_scores.index(max(silhouette_scores))]

# Fit final model with optimal K
final_kmeans = KMeans(n_clusters=optimal_K, init='k-means++', n_init=10, random_state=42)
final_kmeans.fit(X_normalized)

print(f"Optimal number of clusters: {optimal_K}")
print(f"Final model inertia: {final_kmeans.inertia_:.2f}")
```

Trang trình bày 16: Tài nguyên bổ sung

Để khám phá thêm về phân cụm K-mean và phức tạp về thời gian của nó:

1. Bài viết ArXiv về K-means++: URL "k-means++: Ưu điểm của việc gieo hạt nguy hiểm": [https://arxiv.org/abs/0609164](https://arxiv.org/abs/0609164)
2. Tài liệu ArXiv về K-means theo lô nhỏ: URL "Phân cụm K-Means quy mô web": [https://arxiv.org/abs/1006.4757](https://arxiv.org/abs/1006.4757)
3. Bài viết ArXiv về thuật toán Elkan: URL "Dùng bất đẳng thức tam giác để tăng tốc k-Means": [https://arxiv.org/abs/1203.1898](https://arxiv.org/abs/1203.1898)

Các tài nguyên này cung cấp phân tích chuyên sâu và cải tiến thuật toán K-mean, tập trung vào phức tạp về thời gian và hiệu suất tối ưu hóa.
