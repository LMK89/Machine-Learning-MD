## Trình diễn trực quan về phân cụm DBSCAN
Slide 1: Giới thiệu về phân cụm DBSCAN

DBSCAN (Phân cụm ứng dụng không gian dựa trên mật độ có nhiễu) là một thuật toán phân cụm mạnh mẽ giúp nhóm các điểm dữ liệu dựa trên mật độ. Không giống như các phương pháp phân cụm truyền thống, DBSCAN có thể xác định các cụm có hình dạng tùy ý và xử lý nhiễu một cách hiệu quả. Bài trình bày này sẽ khám phá các khái niệm cốt lõi, cách triển khai và lợi thế của DBSCAN so với các thuật toán phân cụm khác.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2) * 0.5
X[100:200] += [2, 2]
X[200:] += [-2, 2]

plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Sample Data for DBSCAN Clustering")
plt.show()
```

Slide 2: Các khái niệm cốt lõi của DBSCAN

DBSCAN dựa vào hai tham số chính: epsilon (ε) và minPts. Epsilon xác định khoảng cách tối đa giữa hai điểm được coi là lân cận, trong khi minPts là số điểm tối thiểu cần thiết để tạo thành một vùng dày đặc. Thuật toán phân loại điểm thành ba loại: điểm cốt lõi, điểm biên và điểm nhiễu.

```python
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def get_neighbors(data, point_idx, epsilon):
    return [idx for idx, point in enumerate(data) if euclidean_distance(data[point_idx], point) <= epsilon]

# Example usage
epsilon = 0.5
minPts = 5
point_idx = 0
neighbors = get_neighbors(X, point_idx, epsilon)
print(f"Number of neighbors for point {point_idx}: {len(neighbors)}")
```

Slide 3: Mã nguồn cho các khái niệm cốt lõi của DBSCAN

```python
def classify_points(data, epsilon, minPts):
    classifications = ['Noise'] * len(data)
    for idx in range(len(data)):
        neighbors = get_neighbors(data, idx, epsilon)
        if len(neighbors) >= minPts:
            classifications[idx] = 'Core'
        elif len(neighbors) > 0:
            classifications[idx] = 'Border'
    return classifications

# Example usage
classifications = classify_points(X, epsilon, minPts)
print(f"Core points: {classifications.count('Core')}")
print(f"Border points: {classifications.count('Border')}")
print(f"Noise points: {classifications.count('Noise')}")
```

Slide 4: Triển khai thuật toán DBSCAN

Thuật toán DBSCAN bắt đầu bằng cách chọn một điểm tùy ý chưa được thăm và tìm tất cả các điểm lân cận của nó trong khoảng cách epsilon. Nếu số lượng hàng xóm ít nhất là minPts thì một cụm mới sẽ được hình thành. Sau đó, thuật toán sẽ mở rộng cụm một cách đệ quy bằng cách thêm các điểm lõi lân cận và các điểm lân cận của chúng.

Slide 5: Mã nguồn triển khai thuật toán DBSCAN

```python
def dbscan(data, epsilon, minPts):
    labels = [0] * len(data)  # 0 represents unvisited points
    cluster_id = 0

    for point_idx in range(len(data)):
        if labels[point_idx] != 0:
            continue

        neighbors = get_neighbors(data, point_idx, epsilon)

        if len(neighbors) < minPts:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(data, labels, point_idx, neighbors, cluster_id, epsilon, minPts)

    return labels

def expand_cluster(data, labels, point_idx, neighbors, cluster_id, epsilon, minPts):
    labels[point_idx] = cluster_id

    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:
            labels[neighbor_idx] = cluster_id
            new_neighbors = get_neighbors(data, neighbor_idx, epsilon)

            if len(new_neighbors) >= minPts:
                neighbors.extend(new_neighbors)

        i += 1

# Example usage
epsilon = 0.5
minPts = 5
cluster_labels = dbscan(X, epsilon, minPts)
```

Slide 6: Trực quan hóa kết quả DBSCAN

Sau khi áp dụng DBSCAN cho dữ liệu mẫu, chúng tôi có thể hình dung kết quả để hiểu rõ hơn cách thuật toán xác định các cụm và xử lý các điểm nhiễu. Hình ảnh trực quan này giúp chứng minh khả năng của thuật toán trong việc phát hiện các cụm có hình dạng tùy ý.

Trang trình bày 7: Mã nguồn để hiển thị kết quả DBSCAN

```python
def plot_dbscan_results(data, labels):
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'black'

        class_member_mask = (labels == label)
        xy = data[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[color], alpha=0.7, label=f'Cluster {label}')

    plt.title("DBSCAN Clustering Results")
    plt.legend()
    plt.show()

# Example usage
plot_dbscan_results(X, cluster_labels)
```

Slide 8: Ưu điểm của DBSCAN so với KMeans

DBSCAN cung cấp một số lợi thế so với các thuật toán phân cụm truyền thống như KMeans:

1. Nó có thể xác định các cụm có hình dạng tùy ý, không chỉ các cụm hình cầu.
2. Nó tự động phát hiện và xử lý các điểm nhiễu.
3. Không cần chỉ định trước số lượng cụm.
4. Nó có thể xử lý các cụm có mật độ khác nhau.

Những ưu điểm này làm cho DBSCAN đặc biệt hữu ích cho các bộ dữ liệu phức tạp có hình dạng và mật độ cụm không đồng nhất.

Trang trình bày 9: Mã nguồn để so sánh DBSCAN và KMeans

```python
from sklearn.cluster import KMeans

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Plot KMeans results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plot_dbscan_results(X, cluster_labels)
plt.title("DBSCAN Clustering")

plt.subplot(122)
plot_dbscan_results(X, kmeans_labels)
plt.title("KMeans Clustering")

plt.tight_layout()
plt.show()
```

Trang trình bày 10: Ví dụ thực tế: Phân cụm dữ liệu địa lý

DBSCAN đặc biệt hữu ích cho việc phân cụm dữ liệu địa lý, chẳng hạn như xác định các khu vực đô thị hoặc các điểm ưa thích. Hãy xem xét tập dữ liệu tọa độ GPS đại diện cho các vị trí khác nhau trong thành phố. DBSCAN có thể nhóm các điểm này thành các cụm một cách hiệu quả, đại diện cho các vùng lân cận hoặc khu vực hoạt động riêng biệt.

Trang trình bày 11: Mã nguồn cho phân cụm dữ liệu địa lý

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample GPS coordinates
np.random.seed(42)
gps_data = np.random.randn(500, 2) * 0.1
gps_data[:100] += [0.5, 0.5]  # Downtown area
gps_data[100:200] += [-0.5, 0.5]  # Residential area
gps_data[200:300] += [0, -0.5]  # Industrial area

# Apply DBSCAN
epsilon = 0.05
minPts = 10
gps_labels = dbscan(gps_data, epsilon, minPts)

# Visualize results
plt.figure(figsize=(10, 8))
plot_dbscan_results(gps_data, gps_labels)
plt.title("DBSCAN Clustering of GPS Coordinates")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
```

Trang trình bày 12: Ví dụ thực tế: Phân đoạn hình ảnh

Một ứng dụng thực tế khác của DBSCAN là phân đoạn hình ảnh. Bằng cách coi cường độ và vị trí điểm ảnh là các đối tượng, DBSCAN có thể nhóm các điểm ảnh tương tự lại với nhau, phân chia hình ảnh thành các vùng riêng biệt một cách hiệu quả. Kỹ thuật này hữu ích trong nhiều lĩnh vực khác nhau, bao gồm hình ảnh y tế và thị giác máy tính.

Slide 13: Mã nguồn phân đoạn ảnh bằng DBSCAN

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and preprocess the image
image = Image.open("sample_image.jpg").convert("L")  # Convert to grayscale
image_array = np.array(image)
height, width = image_array.shape

# Create feature matrix (x, y, intensity)
features = np.column_stack([np.repeat(np.arange(height), width),
                            np.tile(np.arange(width), height),
                            image_array.flatten()])

# Apply DBSCAN
epsilon = 10
minPts = 50
segment_labels = dbscan(features, epsilon, minPts)

# Reshape labels to image dimensions
segmented_image = segment_labels.reshape(height, width)

# Visualize results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(122)
plt.imshow(segmented_image, cmap='nipy_spectral')
plt.title("DBSCAN Segmentation")

plt.tight_layout()
plt.show()
```

Trang trình bày 14: Hạn chế và cân nhắc

Mặc dù DBSCAN mạnh mẽ nhưng nó có một số hạn chế:

1. Độ nhạy đối với việc lựa chọn tham số (epsilon và minPts).
2. Khó khăn trong việc xử lý các cụm có mật độ khác nhau.
3. Độ phức tạp tính toán O(n^2) trong trường hợp xấu nhất.

Để giải quyết những vấn đề này, các biến thể như OPTICS và HDBSCAN đã được phát triển, mang lại hiệu suất được cải thiện và khả năng thích ứng với các bộ dữ liệu khác nhau.

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin về DBSCAN và các thuật toán phân cụm có liên quan, hãy xem xét các tài nguyên sau:

1. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). Thuật toán dựa trên mật độ để khám phá các cụm trong cơ sở dữ liệu không gian lớn có nhiễu. Trong KDD (Tập 96, số 34, trang 226-231). ArXiv: [https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
2. Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017). Xem lại DBSCAN, xem lại: tại sao và làm thế nào bạn nên (vẫn) sử dụng DBSCAN. Giao dịch ACM trên Hệ thống Cơ sở dữ liệu (TODS), 42(3), 1-21. ArXiv: [https://arxiv.org/abs/1706.06778](https://arxiv.org/abs/1706.06778)
3. Campello, R. J., Moulavi, D., & Sander, J. (2013). Phân cụm dựa trên mật độ dựa trên ước tính mật độ phân cấp. Trong hội nghị Châu Á Thái Bình Dương về khám phá tri thức và khai thác dữ liệu (trang 160-172). Springer, Berlin, Heidelberg. ArXiv: [https://arxiv.org/abs/1507.07212](https://arxiv.org/abs/1507.07212)
