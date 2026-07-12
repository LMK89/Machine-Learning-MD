##Đánh giá phân cụm bằng điểm Silhouette

Trang trình bày 1: Tìm hiểu về phân cụm và điểm bóng

Phân cụm là một kỹ thuật máy học không giám sát được sử dụng để nhóm các dữ liệu tương tự lại với nhau. Điểm Silhouette là thước đo đánh giá chất lượng của các cụm này. Nó đo mức độ phù hợp của từng dữ liệu trong cụm được chỉ định của nó so với các cụm khác. Điểm này dao động từ -1 đến 1, trong đó các giá trị gần bằng 1 biểu thị các cụm được xác định rõ, các giá trị xung quanh 0 mẹo các cụm chéo và các âm giá trị có thể biểu thị phân công cụm không chính xác.

```python
import random

def generate_cluster_data(n_points, n_clusters):
    data = []
    for _ in range(n_clusters):
        center = (random.uniform(-10, 10), random.uniform(-10, 10))
        cluster = [(center[0] + random.gauss(0, 1), center[1] + random.gauss(0, 1))
                   for _ in range(n_points // n_clusters)]
        data.extend(cluster)
    return data

# Generate sample clustered data
sample_data = generate_cluster_data(300, 3)

# Print first 5 data points
print("Sample data points:")
for point in sample_data[:5]:
    print(f"({point[0]:.2f}, {point[1]:.2f})")
```

Trang trình bày 2: Triển khai phân cụm K-mean từ đầu

K-means là một biến phổ phân tích thuật toán. Nó nhắm vào mục tiêu phân chia n quan sát thành cụm, trong đó mỗi quan sát thuộc về cụm có giá trị trung bình gần nhất (trong tâm). Chúng tôi sẽ phát triển các thuật toán này từ đầu bằng cách sử dụng các hàm Python tích hợp sẵn.

```python
import random
import math

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def kmeans(data, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = random.sample(data, k)

    for _ in range(max_iterations):
        # Assign points to nearest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            nearest_centroid = min(range(k), key=lambda i: euclidean_distance(point, centroids[i]))
            clusters[nearest_centroid].append(point)

        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = tuple(sum(coord) / len(cluster) for coord in zip(*cluster))
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(data))  # Reinitialize empty clusters

        # Check for convergence
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return clusters, centroids

# Use the sample data from the previous slide
clusters, centroids = kmeans(sample_data, 3)

print("Number of points in each cluster:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {len(cluster)} points")

print("\nFinal centroids:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
```

Trình bày 3: Triển khai điểm Silhouette từ đầu

Điểm Silhouette định lượng chất lượng của phân cụm công việc. Đối với mỗi dữ liệu, không so sánh khoảng cách trung bình đến các điểm trong cụm của chính nó (a) với khoảng cách trung bình đến các điểm trong cụm lân cận gần nhất (b). Sau đó, Điểm Silhouette được tính là (b - a) / max(a, b).

```python
def silhouette_score(data, clusters):
    def avg_distance(point, cluster):
        return sum(euclidean_distance(point, other) for other in cluster) / len(cluster)

    silhouette_values = []

    for i, cluster in enumerate(clusters):
        for point in cluster:
            a = avg_distance(point, cluster)

            b = float('inf')
            for j, other_cluster in enumerate(clusters):
                if i != j:
                    avg_dist = avg_distance(point, other_cluster)
                    b = min(b, avg_dist)

            silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0
            silhouette_values.append(silhouette)

    return sum(silhouette_values) / len(silhouette_values)

# Calculate Silhouette Score for our clustering
score = silhouette_score(sample_data, clusters)
print(f"Silhouette Score: {score:.4f}")
```

Trang trình bày 4: Giải thích Điểm Silhouette

Điểm Silhouette dao động từ -1 đến 1. Điểm gần 1 hơn cho biết các điểm dữ liệu được kết hợp tốt với các cụm của chính chúng và không phù hợp với các cụm lân cận. Khoảng điểm 0 mũi nhọn các cụm chéo, trong khi điểm âm có thể chỉ ra rằng các dữ liệu được phân bổ cho các cụm sai. Trong thực tế, điểm trên 0,5 thường được coi là tốt, trong khi điểm dưới 0,3 có thể chọn chất lượng phân cụm gần.

```python
def interpret_silhouette_score(score):
    if score > 0.7:
        return "Excellent clustering"
    elif score > 0.5:
        return "Good clustering"
    elif score > 0.3:
        return "Fair clustering, consider adjusting parameters"
    else:
        return "Poor clustering, reevaluate your approach"

# Interpret our Silhouette Score
interpretation = interpret_silhouette_score(score)
print(f"Interpretation: {interpretation}")

# Generate scores for different numbers of clusters
for k in range(2, 6):
    clusters, _ = kmeans(sample_data, k)
    score = silhouette_score(sample_data, clusters)
    print(f"K = {k}, Silhouette Score: {score:.4f}")
```

Trang trình bày 5: Hiệu chỉnh bóng

Để làm cho Điểm Silhouette trực quan hơn, đặc biệt đối với các bên liên quan không liên quan đến kỹ thuật, chúng tôi có thể hiệu chỉnh chúng theo thang đo \[0, 1\]. Phép biến đổi này duy trì thứ tự tương thích của số đồng thời để chúng dễ hiểu hơn dưới dạng phần trăm hoặc hiệu suất.

```python
def calibrate_silhouette_score(score):
    # Transform score from [-1, 1] to [0, 1]
    return (score + 1) / 2

# Original and calibrated scores for different clustering scenarios
scenarios = [
    ("Well-separated clusters", 0.8),
    ("Overlapping clusters", 0.3),
    ("Poorly defined clusters", -0.2)
]

print("Scenario | Original Score | Calibrated Score")
print("---------+----------------+-----------------")
for scenario, orig_score in scenarios:
    calibrated = calibrate_silhouette_score(orig_score)
    print(f"{scenario:<9} | {orig_score:14.2f} | {calibrated:15.2f}")

# Calibrate our actual score
calibrated_score = calibrate_silhouette_score(score)
print(f"\nOur clustering - Original: {score:.4f}, Calibrated: {calibrated_score:.4f}")
```

Trang trình bày 6: Use point Silhouette in production

Trong môi trường sản xuất, Điểm Silhouette có thể đóng vai trò là thước đo độ tin cậy để phân tích kết quả. Nó có thể được sử dụng để giám sát chất lượng phân cụm theo thời gian, kích hoạt cảnh báo về những thay đổi không mong muốn hoặc điều chỉnh các tham số phân cụm. Đây là một ví dụ đơn giản về cách phát triển khai điều này trong cài đặt giống như sản phẩm:

```python
import time

def simulate_production_data(n_points, n_clusters, noise_level):
    base_data = generate_cluster_data(n_points, n_clusters)
    noisy_data = [(x + random.gauss(0, noise_level), y + random.gauss(0, noise_level))
                  for x, y in base_data]
    return noisy_data

def monitor_clustering_quality(data, k, threshold=0.5):
    clusters, _ = kmeans(data, k)
    score = silhouette_score(data, clusters)
    calibrated_score = calibrate_silhouette_score(score)

    if calibrated_score < threshold:
        print(f"Alert: Low clustering quality detected! Score: {calibrated_score:.4f}")
    else:
        print(f"Clustering quality acceptable. Score: {calibrated_score:.4f}")

    return calibrated_score

# Simulate a production environment
for i in range(5):
    print(f"\nIteration {i+1}")
    production_data = simulate_production_data(300, 3, noise_level=0.5 * i)
    score = monitor_clustering_quality(production_data, k=3)
    time.sleep(1)  # Simulate time passing between checks
```

Trang trình bày 7: Ví dụ thực tế: Phân khúc khách hàng

Phân khúc khách hàng là một ứng dụng phổ biến của phân cụm trong kinh doanh. Vui lòng xem xét một nền tảng thương mại điện tử muốn phân tích khách hàng dựa trên hành vi mua hàng của họ. Chúng tôi sẽ sử dụng hai tính năng: giá trị đơn hàng trung bình và tần suất mua hàng.

```python
def generate_customer_data(n_customers):
    data = []
    # Loyal high-value customers
    data.extend([(random.gauss(200, 30), random.gauss(10, 2)) for _ in range(n_customers // 3)])
    # Regular mid-value customers
    data.extend([(random.gauss(100, 20), random.gauss(5, 1)) for _ in range(n_customers // 3)])
    # Occasional low-value customers
    data.extend([(random.gauss(50, 10), random.gauss(2, 0.5)) for _ in range(n_customers // 3)])
    return data

customer_data = generate_customer_data(300)

# Perform clustering
clusters, centroids = kmeans(customer_data, 3)

# Calculate and interpret Silhouette Score
score = silhouette_score(customer_data, clusters)
calibrated_score = calibrate_silhouette_score(score)

print(f"Customer Segmentation Silhouette Score: {score:.4f}")
print(f"Calibrated Score: {calibrated_score:.4f}")
print(f"Interpretation: {interpret_silhouette_score(score)}")

# Print cluster centroids
for i, centroid in enumerate(centroids):
    print(f"Segment {i+1} centroid: Avg Order Value: ${centroid[0]:.2f}, "
          f"Purchase Frequency: {centroid[1]:.2f} times/month")
```

Slide 8: Ví dụ thực tế: Phân cụm tài liệu

Phân cụm tài liệu rất hữu ích trong nhiều ứng dụng khác nhau, được giới hạn như tổ chức các bộ sưu tập văn bản lớn hơn hoặc cải thiện kết quả tìm kiếm. Hãy mô phỏng một kịch bản phân cụm tài liệu đơn giản bằng cách sử dụng Tần số từ làm đặc điểm.

```python
import string

def preprocess_text(text):
    return ''.join(c.lower() for c in text if c not in string.punctuation)

def text_to_vector(text, vocabulary):
    words = preprocess_text(text).split()
    return [words.count(word) for word in vocabulary]

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing deals with text and speech",
    "Computer vision focuses on image and video analysis",
    "Reinforcement learning involves agents and environments",
    "Data science combines statistics and programming",
    "Big data requires distributed computing systems",
    "Cloud computing provides scalable infrastructure"
]

# Create vocabulary and document vectors
vocabulary = list(set(word for doc in documents for word in preprocess_text(doc).split()))
doc_vectors = [text_to_vector(doc, vocabulary) for doc in documents]

# Perform clustering
clusters, _ = kmeans(doc_vectors, 3)

# Calculate Silhouette Score
score = silhouette_score(doc_vectors, clusters)
calibrated_score = calibrate_silhouette_score(score)

print(f"Document Clustering Silhouette Score: {score:.4f}")
print(f"Calibrated Score: {calibrated_score:.4f}")
print(f"Interpretation: {interpret_silhouette_score(score)}")

# Print cluster contents
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i+1}:")
    for j in cluster:
        print(f"- {documents[j][:50]}...")
```

Trang trình bày 9: Cụm tối ưu hóa

Một cách sử dụng phổ biến của Point Silhouette là một số lượng tối ưu được xác định cụ thể. Bằng cách tính điểm cho các cụm số khác nhau, chúng tôi có thể tìm ra cấu hình để tạo ra các cụm được xác định tốt nhất.

```python
def optimize_clusters(data, max_clusters):
    scores = []
    for k in range(2, max_clusters + 1):
        clusters, _ = kmeans(data, k)
        score = silhouette_score(data, clusters)
        scores.append((k, score))

    return max(scores, key=lambda x: x[1])

# Optimize clusters for customer data
customer_data = generate_customer_data(300)
optimal_k, best_score = optimize_clusters(customer_data, 10)

print(f"Optimal number of clusters: {optimal_k}")
print(f"Best Silhouette Score: {best_score:.4f}")

# Plot Silhouette Scores
print("\nSilhouette Scores for different numbers of clusters:")
for k in range(2, 11):
    clusters, _ = kmeans(customer_data, k)
    score = silhouette_score(customer_data, clusters)
    print(f"K = {k}: {'*' * int(score * 50)} {score:.4f}")
```

Slide 10: Xử lý chiều cao dữ liệu

Khi xử lý dữ liệu nhiều chiều, việc tính toán khoảng cách trở nên đắt tiền về mặt tính toán và ít ý nghĩa hơn làm "lời nói của chiều". Trong những trường hợp như vậy, kỹ thuật giảm kích thước có thể được áp dụng trước khi phân cụm. Đây là một ví dụ đơn giản sử dụng Phân tích thành phần chính (PCA) được phát triển từ đầu:

```python
def pca(data, n_components):
    # Center the data
    mean = [sum(col) / len(col) for col in zip(*data)]
    centered = [[x - m for x, m in zip(point, mean)] for point in data]

    # Compute covariance matrix
    cov_matrix = [[sum(a * b for a, b in zip(col1, col2)) / (len(data) - 1)
                   for col2 in zip(*centered)] for col1 in zip(*centered)]

    # Compute eigenvectors and eigenvalues
    def power_iteration(matrix, num_iterations=100):
        b_k = [random.random() for _ in range(len(matrix))]
        for _ in range(num_iterations):
            b_k1 = [sum(matrix[i][j] * b_k[j] for j in range(len(matrix))) for i in range(len(matrix))]
            b_k1_norm = math.sqrt(sum(x**2 for x in b_k1))
            b_k = [x / b_k1_norm for x in b_k1]
        return b_k

    eigenvectors = [power_iteration(cov_matrix) for _ in range(n_components)]

    # Project data
    return [[sum(a * b for a, b in zip(point, evec)) for evec in eigenvectors] for point in centered]

# Generate high-dimensional data
high_dim_data = [[random.gauss(0, 1) for _ in range(20)] for _ in range(100)]

# Reduce dimensionality
reduced_data = pca(high_dim_data, 2)

# Cluster reduced data
clusters, _ = kmeans(reduced_data, 3)
score = silhouette_score(reduced_data, clusters)

print(f"Silhouette Score after dimensionality reduction: {score:.4f}")
print(f"Interpretation: {interpret_silhouette_score(score)}")
```

Slide 11: Xử lý các ngoại lệ

Các ngoại lệ có thể gây ảnh hưởng đáng kể đến kết quả phân cụm và điểm Silhouette. Một cách tiếp cận khác để giảm thiểu điều này là sử dụng phân tích kỹ thuật mạnh mẽ hoặc xử lý trước dữ liệu để loại bỏ hoặc giảm hoạt động của các ngoại lệ. Dưới đây là ví dụ về cách phát triển kỹ thuật phát hiện và loại bỏ ngoại lệ đơn giản bằng phương pháp pháp Phạm vi liên tứ phân vị (IQR):

```python
def remove_outliers(data, k=1.5):
    def iqr_boundaries(values):
        sorted_values = sorted(values)
        q1, q3 = sorted_values[len(sorted_values)//4], sorted_values[3*len(sorted_values)//4]
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        return lower_bound, upper_bound

    dimensions = list(zip(*data))
    bounds = [iqr_boundaries(dim) for dim in dimensions]

    cleaned_data = []
    for point in data:
        if all(bound[0] <= value <= bound[1] for value, bound in zip(point, bounds)):
            cleaned_data.append(point)

    return cleaned_data

# Generate data with outliers
data_with_outliers = generate_cluster_data(300, 3)
data_with_outliers.extend([(100, 100), (-100, -100)])  # Add outliers

# Remove outliers
cleaned_data = remove_outliers(data_with_outliers)

print(f"Original data points: {len(data_with_outliers)}")
print(f"Cleaned data points: {len(cleaned_data)}")

# Compare clustering results
original_clusters, _ = kmeans(data_with_outliers, 3)
original_score = silhouette_score(data_with_outliers, original_clusters)

cleaned_clusters, _ = kmeans(cleaned_data, 3)
cleaned_score = silhouette_score(cleaned_data, cleaned_clusters)

print(f"Original Silhouette Score: {original_score:.4f}")
print(f"Cleaned Silhouette Score: {cleaned_score:.4f}")
```

Trang trình bày 12: So sánh điểm Silhouette với các số liệu khác

Mặc dù Điểm Silhouette rất hữu ích nhưng việc so sánh nó với các số liệu đánh giá cụm cụm thường có ích. Ở đây, chúng tôi sẽ phát triển và so sánh Điểm Silhouette với Chỉ số Davies-Bouldin, một chỉ số đánh giá phân cụm nội bộ khác không yêu cầu nhãn thực tế cơ bản.

```python
def davies_bouldin_index(data, clusters):
    def cluster_diameter(cluster):
        return max(euclidean_distance(p1, p2) for p1 in cluster for p2 in cluster)

    def cluster_centroid(cluster):
        return tuple(sum(coord) / len(cluster) for coord in zip(*cluster))

    n = len(clusters)
    centroids = [cluster_centroid(cluster) for cluster in clusters]
    diameters = [cluster_diameter(cluster) for cluster in clusters]

    db_index = 0
    for i in range(n):
        max_ratio = 0
        for j in range(n):
            if i != j:
                ratio = (diameters[i] + diameters[j]) / euclidean_distance(centroids[i], centroids[j])
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio

    return db_index / n

# Generate clustered data
data = generate_cluster_data(300, 3)

# Perform clustering
clusters, _ = kmeans(data, 3)

# Calculate metrics
silhouette = silhouette_score(data, clusters)
db_index = davies_bouldin_index(data, clusters)

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")
print("Note: For Silhouette Score, higher is better. For Davies-Bouldin Index, lower is better.")
```

Trang trình bày 13: Phân cụm kết quả trực quan

Trực quan hóa là rất quan trọng để hiểu kết quả phân cụm. Mặc dù không thể sử dụng các thư viện bên ngoài nhưng chúng tôi có thể tạo một biểu đồ ASCII đơn giản để trực quan hóa kết quả phân cụm 2D cùng với Điểm Silhouette.

```python
def ascii_plot(data, clusters, width=60, height=20):
    min_x = min(p[0] for p in data)
    max_x = max(p[0] for p in data)
    min_y = min(p[1] for p in data)
    max_y = max(p[1] for p in data)

    def scale(value, min_val, max_val, size):
        return int((value - min_val) / (max_val - min_val) * (size - 1))

    plot = [[' ' for _ in range(width)] for _ in range(height)]

    for cluster_idx, cluster in enumerate(clusters):
        for point in cluster:
            x = scale(point[0], min_x, max_x, width)
            y = height - 1 - scale(point[1], min_y, max_y, height)
            plot[y][x] = str(cluster_idx)

    return '\n'.join(''.join(row) for row in plot)

# Generate and cluster 2D data
data = generate_cluster_data(300, 3)
clusters, _ = kmeans(data, 3)

# Calculate Silhouette Score
score = silhouette_score(data, clusters)

print(f"Clustering Visualization (Silhouette Score: {score:.4f})")
print(ascii_plot(data, clusters))
```

Trang trình bày 14: Kết luận và các phương pháp hay nhất

Điểm Silhouette là một công cụ có giá trị để đánh giá chất lượng phân cụm, đặc biệt là trong môi trường sản xuất nơi không có cơ sở thực sự nhãn. Dưới đây là một số phương pháp hay nhất:

1. Sử dụng Điểm Silhouette cùng với các số liệu khác để đánh giá toàn diện.
2. Hiệu chỉnh số theo phạm vi \[0, 1\] để các bên liên quan không chuyên về kỹ thuật giải quyết dễ dàng hơn.
3. Theo dõi Điểm Silhouette theo thời gian để phát hiện những thay đổi về chất phân phối hoặc phân cụm dữ liệu.
4. Sử dụng Điểm Silhouette để tối ưu hóa số lượng cụm.
5. Hãy biết những giới hạn, suy nghĩ như độ nhạy cảm với mật khẩu và "lời nói của chiều".

Bằng cách thực hiện những thực tiễn này, bạn có thể nâng cao độ tin cậy về kết quả phân cụm kết quả của mình và đưa ra quyết định sáng suốt hơn dựa trên dữ liệu nhóm của mình.

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về đánh giá cụm và điểm Silhouette, đây là một số tài nguyên có giá trị:

1. Rousseeuw, P. J. (1987). &quot;Hình bóng: hỗ trợ đồ họa để giải thích và xác định phân tích cụm&quot;. Tạp chí Toán học tính toán và ứng dụng. 20: 53–65. ArXiv: [https://arxiv.org/abs/2107.10874](https://arxiv.org/abs/2107.10874)
2. Arbelaitz, O., Gurrutxaga, I., Muguerza, J., Pérez, J. M., & Perona, I. (2013). &quot;Một nghiên cứu so sánh chiều rộng sâu về các chỉ số hiệu lực của cụm&quot;. Nhận mẫu dạng. 46(1): 243-256. ArXiv: [https://arxiv.org/abs/1110.3174](https://arxiv.org/abs/1110.3174)
3. Bholowalia, P., & Kumar, A. (2014). "EBK-có nghĩa là: Một kỹ thuật phân tích dựa trên phương pháp giảm tay và k-mean trong WSN". Tạp chí quốc tế về ứng dụng máy tính. 105(9). ArXiv: [https://arxiv.org/abs/1410.5545](https://arxiv.org/abs/1410.5545)

Các bài viết này cung cấp phân tích chuyên sâu về các kỹ thuật đánh giá phân cụm, bao gồm Điểm Silhouette, đồng thời đưa ra những hiểu biết sâu sắc về điểm mạnh và hạn chế của chúng trong các bối cảnh khác nhau.
