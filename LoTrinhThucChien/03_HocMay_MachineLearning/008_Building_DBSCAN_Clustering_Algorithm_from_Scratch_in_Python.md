## Xây dựng DBSCAN phân tích thuật toán từ đầu bằng Python
Slide 1: Giới thiệu về DBSCAN

DBSCAN (Phân cụm ứng dụng không dựa trên mật độ nhiễu) là một biến phổ phân tích kỹ thuật được sử dụng trong khai thác dữ liệu và học máy. Nhóm các điểm được đóng gói chặt chẽ lại với nhau, đánh dấu các điểm của mình trong các vùng có mật độ thấp là các ngoại lệ. Hãy khám phá cách xây dựng thuật toán này từ đầu bằng Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2) * 0.5
X = np.r_[X, X + [2, 2], X + [-2, 2]]

plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Sample Data for DBSCAN")
plt.show()
```

Slide 2: Tìm hiểu DBSCAN các thông số

DBSCAN yêu cầu hai tham số chính: epsilon (eps) và điểm tối thiểu (min\_pts). Epsilon xác định khoảng cách cận cảnh, trong khi min\_pts đặt số điểm tối thiểu cần thiết để tạo thành một vùng dày đặc. Các tham số này có hại đáng kể đến kết quả phân cụm.

```python
def plot_circles(X, eps):
    for point in X:
        circle = plt.Circle(point, eps, fill=False, linestyle='--')
        plt.gca().add_artist(circle)

eps = 0.5
min_pts = 5

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plot_circles(X[:5], eps)
plt.title(f"Epsilon Neighborhoods (eps={eps})")

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.scatter(X[0], X[1], s=100, c='red')
plt.title(f"Core Point (min_pts={min_pts})")
plt.show()
```

Slide 3: Thực hiện tính toán khoảng cách

Bước đầu tiên trong DBSCAN là tính khoảng cách giữa các điểm. Chúng tôi sẽ sử dụng khoảng cách Euclide cho ví dụ này, nhưng các cách khoảng cách số liệu khác có thể được sử dụng tùy thuộc vào ứng dụng.

```python
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def get_neighbors(X, point_idx, eps):
    distances = [euclidean_distance(X[point_idx], other_point) for other_point in X]
    return [i for i, dist in enumerate(distances) if dist <= eps]

# Example usage
point_idx = 0
neighbors = get_neighbors(X, point_idx, eps)
print(f"Number of neighbors for point {point_idx}: {len(neighbors)}")
```

Slide 4: Xác định cốt lõi định nghĩa

Điểm cốt lõi là những điểm có ít nhất min\_pts lân cận trong khoảng cách epsilon. Chúng tôi tạo cơ sở dữ liệu cho các cụm trong DBSCAN.

```python
def find_core_points(X, eps, min_pts):
    core_points = []
    for i in range(len(X)):
        if len(get_neighbors(X, i, eps)) >= min_pts:
            core_points.append(i)
    return core_points

core_points = find_core_points(X, eps, min_pts)
print(f"Number of core points: {len(core_points)}")

plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.scatter(X[core_points, 0], X[core_points, 1], c='red', s=50)
plt.title("Core Points Identified")
plt.show()
```

Trang trình bày 5: Mở rộng cụm

Sau khi xác định các cốt lõi, chúng tôi sẽ mở rộng các cụm bằng cách bao gồm các cụm lân cận của chúng và các cụm lân cận của cụm lân cận đó theo cách quy phục.

```python
def expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_pts):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor = neighbors[i]
        if labels[neighbor] == -1:  # Noise becomes border point
            labels[neighbor] = cluster_id
        elif labels[neighbor] == 0:  # Unvisited
            labels[neighbor] = cluster_id
            new_neighbors = get_neighbors(X, neighbor, eps)
            if len(new_neighbors) >= min_pts:
                neighbors.extend(new_neighbors)
        i += 1
    return labels

# This function will be used in the main DBSCAN algorithm
```

Slide 6: Triển khai DBSCAN thuật toán

Bây giờ, hãy tập hợp mọi thứ lại với nhau để phát triển việc hoàn thiện khai báo thuật toán DBSCAN.

```python
def dbscan(X, eps, min_pts):
    labels = [0] * len(X)  # 0: unvisited, -1: noise
    cluster_id = 0
    core_points = find_core_points(X, eps, min_pts)

    for point_idx in range(len(X)):
        if labels[point_idx] != 0:
            continue
        if point_idx in core_points:
            cluster_id += 1
            neighbors = get_neighbors(X, point_idx, eps)
            labels = expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_pts)
        else:
            labels[point_idx] = -1  # Noise

    return labels

# Run DBSCAN
labels = dbscan(X, eps, min_pts)
```

Slide 7: DBSCAN kết quả trực quan

Vui lòng trực tiếp hóa kết quả phân cụm để xem DBSCAN đã hoạt động như thế nào trên mẫu dữ liệu của chúng tôi.

```python
unique_labels = set(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

plt.figure(figsize=(10, 7))
for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'gray'  # Use gray for noise points
    class_member_mask = (np.array(labels) == label)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[color], alpha=0.7, label=f'Cluster {label}')

plt.title("DBSCAN Clustering Results")
plt.legend()
plt.show()
```

Trang trình bày 8: Xử lý các phân phối dữ liệu khác nhau

DBSCAN hoạt động tốt trên các dữ liệu có mật độ khác nhau và dạng hình không cần thiết. Hãy thử nghiệm nó trên một tập dữ liệu phức tạp hơn.

```python
from sklearn.datasets import make_moons

# Generate a more complex dataset
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Run DBSCAN with adjusted parameters
eps_moons = 0.2
min_pts_moons = 5
labels_moons = dbscan(X_moons, eps_moons, min_pts_moons)

# Visualize results
plt.figure(figsize=(10, 7))
unique_labels = set(labels_moons)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'gray'
    class_member_mask = (np.array(labels_moons) == label)
    xy = X_moons[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[color], alpha=0.7, label=f'Cluster {label}')

plt.title("DBSCAN on Complex Dataset")
plt.legend()
plt.show()
```

Trang trình bày 9: Thông số độ nhạy

Hiệu suất của DBSCAN phụ thuộc vào các thông số của nó. Hãy cùng khám phá việc thay đổi ảnh hưởng của eps để tìm ra kết quả phân cụm.

```python
def plot_dbscan_results(X, eps, min_pts):
    labels = dbscan(X, eps, min_pts)
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'gray'
        class_member_mask = (np.array(labels) == label)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[color], alpha=0.7)
    plt.title(f"DBSCAN: eps={eps}, min_pts={min_pts}")

plt.figure(figsize=(15, 5))
eps_values = [0.1, 0.3, 0.5]

for i, eps in enumerate(eps_values):
    plt.subplot(1, 3, i+1)
    plot_dbscan_results(X_moons, eps, min_pts_moons)

plt.tight_layout()
plt.show()
```

Trang trình bày 10: Ví dụ thực tế: Phân cụm địa lý

DBSCAN đặc biệt hữu ích cho địa lý dữ liệu. Vui lòng sử dụng nó để phân tích các thành phố dựa trên chế độ của chúng.

```python
# Sample city data (longitude, latitude)
cities = np.array([
    [-122.4194, 37.7749],  # San Francisco
    [-122.2711, 37.8044],  # Berkeley
    [-122.0839, 37.3861],  # San Jose
    [-118.2437, 34.0522],  # Los Angeles
    [-117.1611, 32.7157],  # San Diego
    [-74.0060, 40.7128],   # New York City
    [-73.9442, 40.6782],   # Brooklyn
    [-73.7845, 40.9115],   # White Plains
    [-87.6298, 41.8781],   # Chicago
    [-87.9065, 41.9742],   # O'Hare Airport
])

# Run DBSCAN
eps_cities = 1  # Approximately 111 km
min_pts_cities = 2
labels_cities = dbscan(cities, eps_cities, min_pts_cities)

# Visualize results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(cities[:, 0], cities[:, 1], c=labels_cities, cmap='viridis')
plt.colorbar(scatter)
plt.title("City Clusters based on Geographical Proximity")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
```

Slide 11: DBSCAN hiệu suất tối ưu

Đối với các dữ liệu lớn, chúng tôi có thể tối ưu hóa DBSCAN bằng cách sử dụng không gian chỉ mục cài đặt cấu trúc như cây KD để tìm kiếm cận cảnh nhanh hơn.

```python
from scipy.spatial import cKDTree

def get_neighbors_kdtree(tree, point, eps):
    return tree.query_ball_point(point, eps)

def dbscan_optimized(X, eps, min_pts):
    tree = cKDTree(X)
    labels = [0] * len(X)
    cluster_id = 0

    for point_idx in range(len(X)):
        if labels[point_idx] != 0:
            continue
        neighbors = get_neighbors_kdtree(tree, X[point_idx], eps)
        if len(neighbors) < min_pts:
            labels[point_idx] = -1  # Noise
        else:
            cluster_id += 1
            labels = expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_pts)

    return labels

# Compare performance
import time

start_time = time.time()
labels_original = dbscan(X, eps, min_pts)
original_time = time.time() - start_time

start_time = time.time()
labels_optimized = dbscan_optimized(X, eps, min_pts)
optimized_time = time.time() - start_time

print(f"Original DBSCAN time: {original_time:.4f} seconds")
print(f"Optimized DBSCAN time: {optimized_time:.4f} seconds")
print(f"Speed improvement: {original_time / optimized_time:.2f}x")
```

Slide 12: Xử lý chiều cao dữ liệu

DBSCAN có thể gặp khó khăn với dữ liệu nhiều chiều làm lời nói về chiều. Hãy khám phá một kỹ thuật để giải quyết vấn đề này: giảm kích thước bằng PCA.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Generate high-dimensional data
X_high_dim, _ = make_blobs(n_samples=300, n_features=20, centers=3, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_high_dim)

# Run DBSCAN on reduced data
eps_reduced = 2
min_pts_reduced = 5
labels_reduced = dbscan(X_reduced, eps_reduced, min_pts_reduced)

# Visualize results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_reduced, cmap='viridis')
plt.colorbar(scatter)
plt.title("DBSCAN on PCA-reduced High-Dimensional Data")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()
```

Trang trình bày 13: Ví dụ thực tế: Phân đoạn hình ảnh

DBSCAN can be apply for image phân đoạn dịch vụ. Vui lòng sử dụng nó để phân tích hình ảnh dựa trên cường độ và vị trí pixel.

```python
from skimage import io
from skimage.color import rgb2gray

# Load and preprocess image
image = io.imread('https://raw.githubusercontent.com/scikit-image/scikit-image/master/skimage/data/astronaut.png')
gray_image = rgb2gray(image)

# Create feature matrix
h, w = gray_image.shape
X_image = np.column_stack([np.repeat(np.arange(h), w),
                           np.tile(np.arange(w), h),
                           gray_image.ravel()])

# Run DBSCAN
eps_image = 5
min_pts_image = 50
labels_image = dbscan(X_image, eps_image, min_pts_image)

# Visualize results
segmented_image = labels_image.reshape(gray_image.shape)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(gray_image, cmap='gray')
plt.title("Original Grayscale Image")
plt.subplot(122)
plt.imshow(segmented_image, cmap='viridis')
plt.title("DBSCAN Segmented Image")
plt.show()
```

Slide 14: Các công thức và chế độ giới hạn

Mặc dù DBSCAN mạnh mẽ nhưng nó cũng có những chế độ hạn chế. Nó phải lộn xộn với các mật độ khác nhau và nhiều dữ liệu. Đơn vị lựa chọn eps và min\_pts có thể là một công thức. Đối với các mật khẩu khác nhau, hãy xem xét tính toán OPTICS hoặc HDBSCAN. Đối với kích thước cao, hãy sử dụng kỹ thuật giảm kích thước hoặc điều chỉnh khoảng cách dữ liệu.

```python
# Varying density example
X_varied = np.vstack([
    np.random.randn(100, 2) * 0.3,
    np.random.randn(50, 2) * 0.1 + [1, 1]
])

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_varied[:, 0], X_varied[:, 1])
plt.title("Varying Density Data")

labels_varied = dbscan(X_varied, eps=0.1, min_pts=5)
plt.subplot(122)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=labels_varied, cmap='viridis')
plt.title("DBSCAN Result")
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến công việc tìm hiểu sâu hơn về DBSCAN và các thuật toán phân tích có liên quan, hãy xem xét khám phá các tài nguyên có giá trị sau:

1. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). Thuật toán dựa trên mật khẩu để khám phá các cụm trong cơ sở dữ liệu không có nhiễu lớn. Trong KDD (Tập 96, số 34, trang 226-231). Có tại: [https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
2. Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017). Xem lại, xem lại DBSCAN: Tại sao và như thế nào bạn nên (vẫn) sử dụng DBSCAN. ACM giao dịch trên cơ sở dữ liệu hệ thống (TODS), 42(3), 1-21. ArXiv: [https://arxiv.org/abs/1706.06778](https://arxiv.org/abs/1706.06778)
3. Campello, R. J., Moulavi, D., & Sander, J. (2013). Phân cụm dựa trên mật khẩu dựa trên tính toán phân cấp mật khẩu. Trong hội nghị Châu Á Thái Bình Dương về khám phá tri thức và khai thác dữ liệu (trang 160-172). Springer, Berlin, Heidelberg. ArXiv: [https://arxiv.org/abs/1507.07021](https://arxiv.org/abs/1507.07021)

Bài viết này cung cấp những giải pháp sâu sắc về nền tảng lý thuyết, ứng dụng thực tế và phần mở rộng của DBSCAN để giải quyết các công thức dữ liệu khác nhau. Chúng tôi cung cấp những hiểu biết sâu sắc có giá trị để hiểu và phát triển các cụm phân tích thuật toán dựa trên mật độ.
