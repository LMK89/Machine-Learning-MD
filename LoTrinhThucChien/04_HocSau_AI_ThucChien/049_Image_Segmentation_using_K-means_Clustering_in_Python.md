## Phân đoạn hình ảnh bằng cách sử dụng K-mean Clustering trong Python
Trang trình bày 1: Giới thiệu về Phân đoạn hình ảnh với phân cụm K-mean

Phân đoạn hình ảnh là một nhiệm vụ quan trọng trong thị giác máy tính, liên quan đến việc phân vùng hình ảnh thành nhiều phân đoạn hoặc vùng. Phân cụm K-mean là một thuật toán học không giám sát phổ biến có thể được áp dụng để phân đoạn hình ảnh. Kỹ thuật này nhóm các pixel có đặc điểm tương tự thành các cụm, phân tách hiệu quả các đối tượng hoặc vùng khác nhau trong ảnh.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# Load and display an example image
image = io.imread('example_image.jpg')
plt.imshow(image)
plt.title('Original Image')
plt.show()
```

Trang trình bày 2: Tổng quan về thuật toán phân cụm K-mean

Phân cụm K-mean nhằm mục đích phân chia n quan sát thành k cụm, trong đó mỗi quan sát thuộc về cụm có giá trị trung bình gần nhất. Trong bối cảnh phân đoạn hình ảnh, các pixel được coi là các quan sát và các giá trị màu của chúng (thường là trong không gian RGB) đóng vai trò là các đặc điểm để phân cụm.

```python
def kmeans_clustering(image, n_clusters):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)

    # Get the labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return labels, centers
```

Slide 3: Preprocessing the Image

Before applying K-means clustering, we need to preprocess the image. This involves reading the image, converting it to a suitable format, and normalizing the pixel values if necessary.

```python
def preprocess_image(image_path):
    # Read the image
    image = io.imread(image_path)

    # Convert to float32 and normalize
    image = image.astype(np.float32) / 255.0

    return image

# Example usage
image = preprocess_image('example_image.jpg')
plt.imshow(image)
plt.title('Preprocessed Image')
plt.show()
```

Trang trình bày 4: Áp dụng phương tiện K cho phân đoạn hình ảnh

Bây giờ chúng tôi sẽ áp dụng thuật toán K-means để phân đoạn hình ảnh được xử lý trước của chúng tôi. Số cụm (k) xác định số lượng phân đoạn trong ảnh đầu ra.

```python
# Apply K-means clustering
n_clusters = 5
labels, centers = kmeans_clustering(image, n_clusters)

# Reshape labels to the image shape
segmented = centers[labels].reshape(image.shape)

# Display the segmented image
plt.imshow(segmented)
plt.title(f'Segmented Image (k={n_clusters})')
plt.show()
```

Trang trình bày 5: Trực quan hóa các trung tâm cụm

Các trung tâm cụm đại diện cho màu trung bình của từng phân đoạn. Hình dung các trung tâm này có thể cung cấp cái nhìn sâu sắc về màu sắc chủ đạo trong hình ảnh được phân đoạn.

```python
def plot_color_centers(centers):
    # Create a bar plot of cluster centers
    plt.figure(figsize=(10, 2))
    plt.imshow([centers], aspect='auto')
    plt.title('Cluster Color Centers')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Visualize cluster centers
plot_color_centers(centers)
```

Trang trình bày 6: Hiệu ứng của việc thay đổi số lượng cụm

Việc lựa chọn k (số cụm) tác động đáng kể đến kết quả phân đoạn. Hãy khám phá các giá trị khác nhau của k ảnh hưởng đến đầu ra như thế nào.

```python
def segment_and_plot(image, k):
    labels, centers = kmeans_clustering(image, k)
    segmented = centers[labels].reshape(image.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented)
    plt.title(f'Segmented (k={k})')
    plt.show()

# Experiment with different k values
for k in [3, 5, 7, 10]:
    segment_and_plot(image, k)
```

Slide 7: Xử lý các không gian màu khác nhau

Phân cụm K-means có thể được áp dụng cho nhiều không gian màu khác nhau. Mặc dù RGB là phổ biến nhưng các không gian khác như LAB hoặc HSV có thể mang lại kết quả tốt hơn cho một số hình ảnh nhất định.

```python
from skimage import color

def segment_in_color_space(image, color_space, n_clusters):
    if color_space == 'RGB':
        transformed = image
    elif color_space == 'LAB':
        transformed = color.rgb2lab(image)
    elif color_space == 'HSV':
        transformed = color.rgb2hsv(image)

    labels, _ = kmeans_clustering(transformed, n_clusters)
    segmented = labels.reshape(image.shape[:2])

    plt.imshow(segmented, cmap='viridis')
    plt.title(f'Segmented in {color_space} space')
    plt.show()

# Segment in different color spaces
for space in ['RGB', 'LAB', 'HSV']:
    segment_in_color_space(image, space, n_clusters=5)
```

Trang trình bày 8: Ví dụ thực tế: Phân đoạn hình ảnh vệ tinh

Phân cụm K-means có thể được áp dụng cho ảnh vệ tinh để phân loại lớp phủ mặt đất. Kỹ thuật này giúp xác định các loại địa hình khác nhau, chẳng hạn như vùng nước, rừng, khu đô thị và đất nông nghiệp.

```python
# Load a satellite image
satellite_image = io.imread('satellite_image.jpg')

# Segment the image
n_clusters = 6
labels, centers = kmeans_clustering(satellite_image, n_clusters)
segmented = centers[labels].reshape(satellite_image.shape)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(satellite_image)
plt.title('Original Satellite Image')
plt.subplot(1, 2, 2)
plt.imshow(segmented)
plt.title('Segmented Satellite Image')
plt.show()
```

Trang trình bày 9: Ví dụ thực tế: Phân đoạn hình ảnh y tế

Phân cụm K-mean cũng hữu ích trong phân tích hình ảnh y tế, chẳng hạn như phân đoạn quét não MRI để xác định các mô khác nhau hoặc các bất thường tiềm ẩn.

```python
# Load an MRI brain scan
mri_image = io.imread('brain_mri.jpg', as_gray=True)

# Segment the image
n_clusters = 4
labels, centers = kmeans_clustering(mri_image.reshape(-1, 1), n_clusters)
segmented = centers[labels].reshape(mri_image.shape)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mri_image, cmap='gray')
plt.title('Original MRI Scan')
plt.subplot(1, 2, 2)
plt.imshow(segmented, cmap='viridis')
plt.title('Segmented MRI Scan')
plt.show()
```

Slide 10: Những thách thức và hạn chế

Mặc dù phân cụm K-mean rất mạnh nhưng nó có một số hạn chế đối với phân đoạn hình ảnh:

1. Độ nhạy khởi tạo: Kết quả có thể thay đổi tùy theo vị trí trung tâm ban đầu.
2. Cần chỉ định k: Số lượng cụm tối ưu không phải lúc nào cũng được biết trước.
3. Giả định về cụm hình cầu: K-means giả định các cụm có hình cầu và có kích thước bằng nhau, điều này có thể không phải lúc nào cũng đúng đối với dữ liệu hình ảnh.

```python
def demonstrate_initialization_sensitivity(image, k, n_runs=5):
    plt.figure(figsize=(15, 3))
    for i in range(n_runs):
        labels, _ = kmeans_clustering(image, k)
        segmented = labels.reshape(image.shape[:2])
        plt.subplot(1, n_runs, i+1)
        plt.imshow(segmented, cmap='viridis')
        plt.title(f'Run {i+1}')
    plt.show()

# Demonstrate sensitivity to initialization
demonstrate_initialization_sensitivity(image, k=5)
```

Trang trình bày 11: Cải thiện phân đoạn K-mean

Một số kỹ thuật có thể nâng cao kết quả phân đoạn K-mean:

1. Khởi tạo nhiều lần: Chạy K-means nhiều lần và chọn kết quả tốt nhất.
2. Phương pháp khuỷu tay: Xác định k tối ưu bằng cách vẽ tổng bình phương trong cụm theo k.
3. Thông tin không gian: Kết hợp tọa độ pixel làm đặc điểm để xem xét các mối quan hệ không gian.

```python
def kmeans_with_spatial_info(image, n_clusters):
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    spatial_features = np.dstack((image, x/w, y/h))

    labels, _ = kmeans_clustering(spatial_features.reshape((-1, 5)), n_clusters)
    return labels.reshape((h, w))

# Segment with spatial information
spatial_segmentation = kmeans_with_spatial_info(image, n_clusters=5)
plt.imshow(spatial_segmentation, cmap='viridis')
plt.title('Segmentation with Spatial Information')
plt.show()
```

Trang trình bày 12: Kết quả phân đoạn sau xử lý

Sau khi áp dụng K-mean, quá trình xử lý hậu kỳ có thể tinh chỉnh phân đoạn:

1. Các thao tác hình thái: Loại bỏ các vùng nhỏ hoặc lấp đầy các lỗ hổng.
2. Phân tích thành phần được kết nối: Xác định và gắn nhãn các vùng riêng biệt.
3. Làm mịn ranh giới: Tinh chỉnh ranh giới phân đoạn để có diện mạo tự nhiên hơn.

```python
from scipy import ndimage

def post_process_segmentation(segmentation):
    # Apply morphological opening to remove small regions
    opened = ndimage.binary_opening(segmentation)

    # Fill holes in the segments
    filled = ndimage.binary_fill_holes(opened)

    # Label connected components
    labeled, _ = ndimage.label(filled)

    return labeled

# Post-process the segmentation
processed = post_process_segmentation(spatial_segmentation)
plt.imshow(processed, cmap='viridis')
plt.title('Post-processed Segmentation')
plt.show()
```

Trang trình bày 13: Đánh giá chất lượng phân khúc

Đánh giá chất lượng phân đoạn hình ảnh là rất quan trọng. Mặc dù thông tin cơ bản là lý tưởng nhưng bạn có thể sử dụng các số liệu không được giám sát khi không có sẵn:

1. Quán tính: Tổng bình phương khoảng cách của các mẫu tới tâm cụm gần nhất của chúng.
2. Điểm hình bóng: Đo mức độ giống nhau của một đối tượng với cụm của chính nó so với các cụm khác.
3. Chỉ số Calinski-Harabasz: Tỷ lệ phân tán giữa cụm và phân tán trong cụm.

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate_segmentation(image, labels):
    pixels = image.reshape((-1, 3))
    inertia = KMeans(n_clusters=len(np.unique(labels))).fit(pixels).inertia_
    silhouette = silhouette_score(pixels, labels)
    calinski = calinski_harabasz_score(pixels, labels)

    print(f"Inertia: {inertia:.2f}")
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Calinski-Harabasz Index: {calinski:.2f}")

# Evaluate the segmentation
evaluate_segmentation(image, labels)
```

Slide 14: Kết luận và định hướng tương lai

Phân cụm K-mean cung cấp một cách tiếp cận đơn giản nhưng hiệu quả để phân đoạn hình ảnh. Mặc dù có những hạn chế nhưng nó đóng vai trò là nền tảng cho các kỹ thuật nâng cao hơn. Các hướng đi trong tương lai bao gồm:

1. Khám phá các thuật toán phân cụm khác (ví dụ: DBSCAN, dịch chuyển trung bình)
2. Kết hợp deep learning để trích xuất đặc trưng trước khi phân cụm
3. Phát triển các phương pháp thích ứng để tự động chọn số cụm tối ưu

```python
# Placeholder for future improvements
def advanced_segmentation(image):
    # TODO: Implement more sophisticated segmentation techniques
    pass

# Placeholder for automatic cluster number selection
def optimal_cluster_number(image):
    # TODO: Implement method to determine optimal k
    pass
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về phân đoạn hình ảnh và phân cụm K-mean, hãy xem xét các tài nguyên sau:

1. Bài viết của ArXiv: "Khảo sát về những tiến bộ gần đây trong việc ước tính mật độ và đếm đám đông bằng hình ảnh đơn dựa trên CNN" (arXiv:1707.01202)
2. Bài viết ArXiv: "Phân đoạn hình ảnh bằng cách sử dụng Deep Learning: Khảo sát" (arXiv:2001.05566)
3. Bài viết ArXiv: "Đánh giá về các kỹ thuật học sâu hiện đại để phân loại hình ảnh" (arXiv:2101.01169)

Các bài viết này cung cấp cái nhìn tổng quan toàn diện về các kỹ thuật tiên tiến trong phân tích và phân đoạn hình ảnh, xây dựng dựa trên nền tảng của các thuật toán cổ điển như phân cụm K-mean.
