## So sánh Phân tích hình bóng và đường cong giảm tay để phân cụm KMeans trong Python
Trang trình bày 1: Giới thiệu về Phân cụm KMeans

KMeans là một máy tính toán thuật toán không giám sát phổ biến được sử dụng để phân tích các dữ liệu thành các nhóm riêng biệt. Nó nhắm vào mục tiêu phân chia n quan sát thành k cụm, trong đó mỗi cuộc khảo sát thuộc về cụm có giá trị trung bình gần nhất. Trong phần trình bày này, chúng tôi sẽ khám phá hai kỹ thuật quan trọng để đánh giá phân cụm KMeans: Phân tích đường cong cong tay và bóng bóng.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Sample Data for KMeans Clustering")
plt.show()
```

Slide 2: Phương pháp đường cong xẹp tay

Đường cong giảm tay là một phương pháp đồ họa được sử dụng để xác định số lượng cụm tối ưu trong KMeans. Nó vẽ tổng bình phương trong cụm (WCSS) theo số cụm. “Khuỷu tay” trong đường cong ý số cụm tối ưu.

```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

Slide 3: thích đường cong xốc tay Giải thích

Đường công Elbow giúp xác định điểm mà việc bổ sung nhiều cụm không làm giảm đáng kể WCSS. Điểm này giống như một sự giảm bớt trong biểu đồ, biểu thị số cụm tối ưu. Tuy nhiên, giảm tay không phải lúc nào cũng được xác định rõ ràng, điều này có thể tạo ra việc giải thích trở nên khó khăn.

```python
# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    return np.degrees(np.math.atan2(np.linalg.det([v1,v2]),np.dot(v1,v2)))

# Find the point with the maximum angle
angles = [calculate_angle((1, wcss[0]), (i+1, wcss[i]), (10, wcss[-1])) for i in range(1, 9)]
elbow = angles.index(max(angles)) + 2

plt.plot(range(1, 11), wcss, marker='o')
plt.plot(elbow, wcss[elbow-1], marker='o', markersize=12, markeredgecolor="red", markerfacecolor="none")
plt.title('Elbow Curve with Detected Elbow')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.annotate(f'Elbow at k={elbow}', xy=(elbow, wcss[elbow-1]), xytext=(elbow+1, wcss[elbow-1]+500),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

Slide 4: Các chế độ của đường cong tay

Mặc dù Đường cong thu gọn mang tính trực quan nhưng cũng không có chế độ hạn chế. Có thể không phải lúc nào đó nó cũng cung cấp một định nghĩa rõ ràng, đặc biệt là đối với các bộ dữ liệu phức tạp. Ngoài ra, nó không xem xét dạng hình hoặc mật khẩu của các cụm, điều này có thể dẫn đến kết quả dưới mức tối ưu trong một số trường hợp.

```python
# Generate a more complex dataset
X_complex, _ = make_blobs(n_samples=500, centers=6, cluster_std=[1.0, 2.0, 0.5, 3.0, 1.5, 1.0], random_state=42)

wcss_complex = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_complex)
    wcss_complex.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_complex, marker='o')
plt.title('Elbow Curve for Complex Dataset')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

Trang trình bày 5: Giới thiệu về Phân tích hình bóng

Phân tích bóng là một kỹ thuật khác để đánh giá hiệu suất phân cụm. Nó đo cường độ tương tự của một đối tượng với cụm chính và các cụm khác. Điểm bóng dao động từ -1 đến 1, trong đó giá trị cao cho biết đối tượng được kết hợp tốt với cụm của chính nó và tương đối phù hợp với các cụm lân cận.

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for i in range(2, 11):  # Start from 2 clusters as silhouette score is not defined for 1 cluster
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, cluster_labels))

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()
```

Trang trình bày 6: Giải thích điểm bóng

Điểm bóng cao hơn cho các cụm được xác định rõ hơn. Số tối ưu thường là cụm tối đa hóa bóng tối. Tuy nhiên, điều quan trọng là phải xem xét không chỉ điểm bóng trung bình mà còn có cả phân tích bổ sung trên tất cả các dữ liệu điểm.

```python
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

n_clusters = 4  # Let's assume we've chosen 4 clusters
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(X)

silhouette_vals = silhouette_samples(X, cluster_labels)

y_lower, y_upper = 0, 0
yticks = []
for i in range(n_clusters):
    cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    color = cm.nipy_spectral(float(i) / n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)
    yticks.append((y_lower + y_upper) / 2)
    y_lower = y_upper

plt.yticks(yticks, range(1, n_clusters + 1))
plt.ylabel("Cluster")
plt.xlabel("Silhouette coefficient")
plt.title("Silhouette Plot for KMeans Clustering")
plt.show()
```

Trang trình bày 7: Ưu điểm của phân tích hình bóng

Phân tích Silhouette cung cấp cái nhìn toàn diện về chất lượng cụm. Nó xem xét cả sự gắn kết (các điểm gần nhau trong một cụm như thế nào) và sự phân tách (các cụm được phân tích tốt như thế nào với nhau). Điều này làm cho nó đặc biệt hữu ích cho các tập dữ liệu trong đó các cụm có thể không có dạng hình cầu hoặc có kích thước khác nhau.

```python
# Function to plot clusters with silhouette scores
def plot_clusters_with_silhouette(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_vals = silhouette_samples(X, cluster_labels)

    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='*', s=250, c='red', label='Centroids')

    for i, txt in enumerate(silhouette_vals):
        plt.annotate(f'{txt:.2f}', (X[i, 0], X[i, 1]), fontsize=8)

    plt.title(f'Clusters with Silhouette Scores (n_clusters={n_clusters})')
    plt.legend()
    plt.show()

plot_clusters_with_silhouette(X, 4)
```

Trang trình bày 8: Kết quả phân tích đường cong xẹp tay và hình bóng

Mặc dù cả hai phương pháp đều có điểm mạnh, nhưng việc kết hợp phân tích đường cong Elbow và Silhouette có thể mang lại một cách mạnh mẽ hơn để xác định số lượng cụm tối ưu. Sự hợp lý này giúp giảm thiểu những hạn chế của từng phương pháp và cung cấp cái nhìn toàn diện hơn về hiệu suất phân cụm.

```python
# Function to plot both Elbow Curve and Silhouette Scores
def plot_elbow_and_silhouette(X, max_clusters=10):
    wcss = []
    silhouette_scores = []

    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

        cluster_labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, cluster_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(2, max_clusters + 1), wcss, marker='o')
    ax1.set_title('Elbow Curve')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('WCSS')

    ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    ax2.set_title('Silhouette Scores')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

plot_elbow_and_silhouette(X)
```

Trang trình bày 9: Ví dụ thực tế: Phân khúc khách hàng

Hãy xem xét vấn đề trong đó một công ty thương mại điện tử mong muốn phân khúc khách hàng dựa trên hành vi mua hàng của họ. Công ty có dữ liệu về hai số liệu chính: giá trị đơn hàng trung bình và tần suất mua hàng. Vui lòng áp dụng phân cụm KMeans và đánh giá nó bằng cách sử dụng cả Phân tích đường cong và hình bóng.

```python
# Generate sample customer data
np.random.seed(42)
order_value = np.random.normal(100, 50, 1000)
purchase_frequency = np.random.normal(5, 2, 1000)
customer_data = np.column_stack((order_value, purchase_frequency))

# Apply Elbow Curve and Silhouette Analysis
plot_elbow_and_silhouette(customer_data)

# Choose optimal number of clusters (let's say 3 based on the results)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(customer_data)

# Plot the results
plt.scatter(customer_data[:, 0], customer_data[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=250, c='red', label='Centroids')
plt.xlabel('Average Order Value')
plt.ylabel('Purchase Frequency')
plt.title('Customer Segmentation')
plt.legend()
plt.show()
```

Slide 10: Diễn giải kết quả phân khúc khách hàng

Kết quả phân cụm cho thấy các phân khúc khách hàng riêng biệt:

1. Giá trị cao, người mua thường xuyên
2. Người mua có giá trị trung bình, tần suất vừa phải
3. Giá trị thấp, người mua không thường xuyên

Việc phân đoạn này cho phép công ty điều chỉnh các chiến lược tiếp theo và cá nhân trải nghiệm khách hàng cho từng nhóm, có khả năng làm tăng sự hài lòng của khách hàng và tăng doanh thu.

```python
# Calculate segment characteristics
for i in range(optimal_clusters):
    segment = customer_data[cluster_labels == i]
    print(f"Segment {i+1}:")
    print(f"  Average Order Value: ${segment[:, 0].mean():.2f}")
    print(f"  Average Purchase Frequency: {segment[:, 1].mean():.2f}")
    print()

# Visualize segment sizes
segment_sizes = [sum(cluster_labels == i) for i in range(optimal_clusters)]
plt.pie(segment_sizes, labels=[f'Segment {i+1}' for i in range(optimal_clusters)], autopct='%1.1f%%')
plt.title('Customer Segment Sizes')
plt.show()
```

Trang trình bày 11: Ví dụ thực tế: Nén hình ảnh

Một ứng dụng thực tế khác của phân cụm KMeans là nén hình ảnh. Bằng cách giảm số lượng màu trong hình ảnh, chúng tôi có thể giảm đáng kể kích thước tệp của nó trong khi vẫn duy trì chất lượng hình ảnh. Vui lòng áp dụng KMeans để nén hình ảnh và sử dụng Elbow Curve để xác định mức độ ưu tiên của màu sắc.

```python
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load and prepare the image
image = Image.open("sample_image.jpg")
image_array = np.array(image)
pixels = image_array.reshape(-1, 3)

# Apply Elbow Curve
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(pixels)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Curve for Image Compression')
plt.xlabel('Number of Colors')
plt.ylabel('WCSS')
plt.show()

# Compress the image with optimal number of colors (let's say 5)
optimal_colors = 5
kmeans = KMeans(n_clusters=optimal_colors, init='k-means++', max_iter=300, n_init=10, random_state=0)
labels = kmeans.fit_predict(pixels)
compressed_pixels = kmeans.cluster_centers_[labels]
compressed_image = compressed_pixels.reshape(image_array.shape).astype(np.uint8)

# Display original and compressed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image_array)
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(compressed_image)
ax2.set_title(f'Compressed Image ({optimal_colors} colors)')
ax2.axis('off')
plt.show()
```

Slide 12: Các công thức và cân nhắc

Mặc dù phân tích đường cong giảm tay và hình bóng là các công cụ mạnh mẽ để đánh giá phân cụm KMeans nhưng chúng có những hạn chế:

1. Độ nhạy cảm với các ngoại lệ giá trị: Cả hai phương pháp đều có thể bị ảnh hưởng bởi các ngoại lệ giá trị trong dữ liệu.
2. Giả định về cụm hình cầu: KMeans giả định các cụm có dạng hình cầu, điều này có thể không đúng lúc nào đó trong dữ liệu trong thế giới thực.
3. Độ tính toán phức tạp: Đối với các dữ liệu lớn, việc tính toán các dữ liệu này có thể có giá trị về mặt tính toán.
4. Tính chủ quan trong cách giải thích: “Khuỷu tay” trong Đường cong tay đôi khi có thể mơ hồ và dễ bị giải thích.

Để giải quyết những vấn đề này, hãy xem xét:

* Sử dụng các kỹ thuật chia tỷ lệ mạnh mẽ để xử lý các ngoại lệ
* Khám phá các phân cụm thuật toán khác cho các cụm không hình cầu
* Triển khai các thuật toán hoặc kỹ thuật lấy kết quả mẫu cho các dữ liệu lớn
* Kết hợp nhiều giá trị dữ liệu để phân tích toàn diện hơn

```python
# Demonstration of the impact of outliers on KMeans clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

# Generate sample data with outliers
np.random.seed(42)
X = np.random.randn(100, 2)
X = np.vstack((X, [10, 10], [-10, -10]))  # Add outliers

# Perform KMeans clustering without scaling
kmeans_no_scale = KMeans(n_clusters=3, random_state=42)
labels_no_scale = kmeans_no_scale.fit_predict(X)

# Perform KMeans clustering with RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
kmeans_scaled = KMeans(n_clusters=3, random_state=42)
labels_scaled = kmeans_scaled.fit_predict(X_scaled)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=labels_no_scale, cmap='viridis')
ax1.set_title('KMeans without Scaling')

ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_scaled, cmap='viridis')
ax2.set_title('KMeans with RobustScaler')

plt.show()
```

Trang trình bày 13: Lời khuyên thiết thực khi sử dụng Phân tích đường cong xuống tay và hình bóng

1. Chuẩn bị dữ liệu: Luôn bắt đầu làm việc sạch sẽ và chuẩn hóa dữ liệu thích hợp.
2. Lựa chọn tính năng: Chọn các tính năng có liên quan đến các phần tạo nên các cụm có ý nghĩa.
3. Chạy nhiều lần: Khởi động ngẫu nhiên trong KMeans, nên chạy thuật toán nhiều lần và lấy kết quả trung bình.
4. Trực quan hóa: Sử dụng trực quan hóa để bổ sung cho các số liệu để hiểu rõ hơn.
5. Kiến thức về lĩnh vực: Kết hợp kiến ​​thức chuyên môn về lĩnh vực khi diễn giải kết quả.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def kmeans_analysis(X, max_clusters=10, n_runs=5):
    wcss = []
    silhouette_avg = []

    for n_clusters in range(2, max_clusters + 1):
        run_wcss = []
        run_silhouette = []

        for _ in range(n_runs):
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=None)
            labels = kmeans.fit_predict(X)
            run_wcss.append(kmeans.inertia_)
            run_silhouette.append(silhouette_score(X, labels))

        wcss.append(np.mean(run_wcss))
        silhouette_avg.append(np.mean(run_silhouette))

    return range(2, max_clusters + 1), wcss, silhouette_avg

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform analysis
n_clusters, wcss, silhouette_avg = kmeans_analysis(X_scaled)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(n_clusters, wcss, marker='o')
ax1.set_title('Elbow Curve')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('WCSS')

ax2.plot(n_clusters, silhouette_avg, marker='o')
ax2.set_title('Average Silhouette Score')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()
```

Trang trình bày 14: Kết luận và các phương pháp hay nhất

Phân tích đường cong thu gọn và hình bóng là các kỹ thuật bổ sung để đánh giá phân cụm KMeans. Trong khi Đường cong Elbow giúp xác định lợi nhuận giảm dần về mặt giải thích phương sai, Phân tích Silhouette cung cấp thông tin chi tiết về chất lượng và phân tích của cụm.

Thực tiễn tốt nhất:

1. Sử dụng kết hợp cả hai phương pháp để phân tích hiệu quả hơn
2. Xem xét bản chất dữ liệu của bạn và vấn đề hiện tại
3. Đừng chỉ dựa vào những số liệu này; kết quả xác thực với chuyên môn về tên miền
4. Nhận được các chế độ giới hạn và giả định của KMeans phân cụm
5. Thử nghiệm các kỹ thuật xử lý khác nhau và kết hợp các tính năng
6. Đối với các dữ liệu lớn, hãy cân nhắc sử dụng các kỹ thuật lấy mẫu để giảm thời gian tính toán

Bằng cách làm theo những hướng dẫn này và hiểu rõ điểm mạnh cũng như hạn chế của từng phương pháp, bạn có thể đưa ra quyết định sáng suốt hơn về chất lượng cụm và số lượng tối ưu cho trường hợp sử dụng công cụ của mình.

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến công việc tìm hiểu sâu hơn về kỹ thuật đánh giá cụm và KMeans thuật toán, thì đây là một số tài nguyên có giá trị:

1. Rousseeuw, P. J. (1987). Bóng: Hỗ trợ đồ họa để giải thích và xác định phân cụm. Tạp chí Toán học tính toán và ứng dụng, 20, 53-65. ArXiv: [https://arxiv.org/abs/2304.10149](https://arxiv.org/abs/2304.10149) (Lưu ý: Đây là bài viết gần đây thảo luận về những tiến bộ trong phân tích hình bóng)
2. Arthur, D., & Vassilvitskii, S. (2007). k-means++: Ưu điểm của việc gieo hạt cẩn thận. Kỷ yếu nghị luận chuyên đề ACM-SIAM thường niên lần thứ 18 về các thuật toán rời rạc. ArXiv: [https://arxiv.org/abs/0606068](https://arxiv.org/abs/0606068)
3. Tibshirani, R., Walther, G., & Hastie, T. (2001). Ước tính số lượng trong một khoảng thống kê dữ liệu tập tin. Tạp chí của Hiệp hội Thống kê Hoàng gia: Series B (Phương pháp thống kê), 63(2), 411-423. ArXiv: [https://arxiv.org/abs/math/0102185](https://arxiv.org/abs/math/0102185)

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về các kỹ thuật đánh giá phân cụm và cải tiến KMeans thuật toán. Họ cung cấp những hiểu biết có giá trị cho cả sự hiểu biết lý thuyết và thực tế.
