## Kiến trúc cơ bản về PCA trong Machine Learning với Python
Slide 1: Giới thiệu về PCA trong Machine Learning

Phân tích thành phần chính (PCA) là một kỹ thuật cơ bản trong máy học để giảm kích thước và trực quan hóa dữ liệu. Nó giúp xác định các mẫu trong nhiều dữ liệu chiều bằng cách chuyển đổi nó thành một hệ thống mới bao gồm các biến không tương thích được gọi là các thành phần chính.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] = 3 * X[:, 0] + np.random.randn(100) * 0.5

# Plot original data
plt.scatter(X[:, 0], X[:, 1])
plt.title('Original Data')
plt.show()

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot transformed data
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA Transformed Data')
plt.show()
```

Slide 2: Khái niệm về các thành phần chính

Các thành phần chính là các hướng dẫn không đặc biệt nhưng dữ liệu thay đổi nhiều nhất. Chúng tôi trực tiếp giao tiếp với nhau và được sắp xếp theo các phương pháp sai mà chúng thích hợp trong dữ liệu.

```python
# Compute and plot principal components
pca = PCA()
pca.fit(X)

plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation
    plt.arrow(pca.mean_[0], pca.mean_[1], comp[0], comp[1],
              color=f'C{i+2}', width=0.05, head_width=0.2)
    plt.text(pca.mean_[0] + comp[0], pca.mean_[1] + comp[1], f'PC{i+1}')

plt.title('Principal Components')
plt.axis('equal')
plt.show()
```

Slide 3: Phương thức được giải thích bởi các thành phần chính

Phương thức đã được giải quyết không thích hợp bởi mỗi thành phần chính cho phạm vi quan trọng có thể được tìm thấy trong biểu thức gốc của dữ liệu. Thông tin này giúp xác định số lượng thành phần cần giữ lại.

```python
# Calculate and plot explained variance ratio
pca = PCA()
pca.fit(X)

plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()

print("Cumulative explained variance ratio:")
print(np.cumsum(pca.explained_variance_ratio_))
```

Trang trình bày 4: Giảm kích thước bằng PCA

PCA có thể được sử dụng để giảm kích thước của dữ liệu bằng cách chọn một tập hợp các thành phần chính thích hợp cho sai sót phương tiện trong dữ liệu.

```python
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot reduced data
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('Digits Dataset Reduced to 2 Dimensions')
plt.show()

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")
```

Slide 5: Chọn số lượng sự kiện

Việc lựa chọn đúng số lượng thành phần là rất quan trọng. Một cách tiếp cận phổ biến là chọn đủ các thành phần để giải quyết tỷ lệ xác định mức tối thiểu xác định của sai sót phương pháp tổng hợp.

```python
# Compute cumulative explained variance ratio
pca = PCA()
pca.fit(X)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance ratio
plt.plot(range(1, len(cumulative_variance_ratio) + 1),
         cumulative_variance_ratio, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs. Number of Components')
plt.show()

# Find number of components for 95% variance explained
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components for 95% variance explained: {n_components}")
```

Trang trình bày 6: PCA để trực tiếp hóa dữ liệu

PCA thường được sử dụng để trực tiếp hóa dữ liệu nhiều chiều trong không gian 2D hoặc 3D, giúp xác định các mẫu và cụm dễ dàng hơn.

```python
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')
plt.colorbar(scatter)
plt.title('Iris Dataset in 3D PCA Space')
plt.show()
```

Trang trình bày 11: So sánh phân tích tổng hợp dữ liệu

PCA có thể được sử dụng để giảm nhiễu dữ liệu bằng cách xây dựng lại dữ liệu chỉ bằng cách sử dụng chính hàng đầu của các thành phần, lọc ra các biến thể ít quan trọng hơn một cách hiệu quả.

```python
# Generate noisy sine wave
t = np.linspace(0, 10, 1000)
x = np.sin(t)
x_noisy = x + 0.5 * np.random.randn(1000)

# Apply PCA for denoising
X = x_noisy.reshape(-1, 1)
pca = PCA(n_components=1)
X_denoised = pca.inverse_transform(pca.fit_transform(X))

# Plot results
plt.figure(figsize=(12, 4))
plt.plot(t, x, label='Original')
plt.plot(t, x_noisy, label='Noisy')
plt.plot(t, X_denoised, label='Denoised')
plt.legend()
plt.title('PCA for Noise Reduction')
plt.show()
```

Slide 8: PCA để trích xuất tính năng

PCA có thể được sử dụng để trích xuất các tính năng mới thu được các khía cạnh quan trọng nhất của dữ liệu. Sau đó, những tính năng mới này có thể được sử dụng để phân tích độ sâu hơn hoặc đưa vào các máy tính toán thuật toán khác.

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load face dataset
faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train SVM classifier
svm = SVC()
svm.fit(X_train_pca, y_train)

# Predict and calculate accuracy
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy using PCA features: {accuracy:.2f}")
```

Trang trình bày 9: PCA và mối tương quan

PCA hoạt động bằng cách tìm ra các sai lệch cực đại đa hướng theo hướng, thường tương thích với các hướng có tương quan cao trong các lệnh cấm đặc biệt. Biết được mối quan hệ này có thể giúp giải quyết kết quả PCA.

```python
import seaborn as sns

# Generate correlated data
np.random.seed(42)
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5
z = 3*x - 2*y + np.random.randn(100)*0.1
data = np.column_stack((x, y, z))

# Compute correlation matrix
corr_matrix = np.corrcoef(data.T)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Apply PCA
pca = PCA()
pca.fit(data)

# Print explained variance ratio
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
```

Trang trình bày 10: PCA và Tiêu chuẩn hóa

Việc chuẩn hóa các tính năng thường là điều cần thiết trước khi áp dụng PCA, đặc biệt khi các tính năng ở các mô hình khác nhau. Điều này đảm bảo rằng PCA nắm bắt được phương sai thực sự thay vì phương sai nhân tạo ra các thang đo khác nhau.

```python
from sklearn.preprocessing import StandardScaler

# Generate data with different scales
np.random.seed(42)
X = np.column_stack((np.random.randn(100)*10, np.random.randn(100)*0.1))

# Apply PCA without standardization
pca_no_scale = PCA()
X_pca_no_scale = pca_no_scale.fit_transform(X)

# Apply PCA with standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_scale = PCA()
X_pca_scale = pca_scale.fit_transform(X_scaled)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X_pca_no_scale[:, 0], X_pca_no_scale[:, 1])
ax1.set_title('PCA without Standardization')
ax2.scatter(X_pca_scale[:, 0], X_pca_scale[:, 1])
ax2.set_title('PCA with Standardization')
plt.show()

print("Explained variance ratio without standardization:")
print(pca_no_scale.explained_variance_ratio_)
print("\nExplained variance ratio with standardization:")
print(pca_scale.explained_variance_ratio_)
```

Trang trình bày 11: PCA tăng dần

Đối với các dữ liệu lớn không vừa với bộ nhớ, Scikit-learn cung cấp PCA tăng dần, có thể xử lý dữ liệu theo thời gian.

```python
from sklearn.decomposition import IncrementalPCA

# Generate large dataset
np.random.seed(42)
X_large = np.random.randn(10000, 100)

# Apply Incremental PCA
batch_size = 500
ipca = IncrementalPCA(n_components=10, batch_size=batch_size)

# Process data in batches
for i in range(0, X_large.shape[0], batch_size):
    ipca.partial_fit(X_large[i:i+batch_size])

# Transform data
X_ipca = ipca.transform(X_large)

print(f"Original shape: {X_large.shape}")
print(f"Reduced shape: {X_ipca.shape}")
print("\nExplained variance ratio:")
print(ipca.explained_variance_ratio_)
```

Slide 12: Ví dụ thực tế: Nén ảnh

PCA có thể được sử dụng để nén hình ảnh bằng cách giảm kích thước của dữ liệu hình ảnh. Ví dụ này bằng chứng PCA có thể nén và tái tạo hình ảnh thang độ xám.

```python
from sklearn.datasets import load_sample_image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and convert image to grayscale
image = load_sample_image("china.jpg")
gray_image = np.mean(image, axis=2).astype(np.float64)

# Reshape image to 2D array
X = gray_image.reshape(-1, gray_image.shape[1])

# Apply PCA with different number of components
n_components_list = [5, 20, 50, 100]
fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(20, 4))

axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for i, n_components in enumerate(n_components_list, 1):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)

    image_reconstructed = X_reconstructed.reshape(gray_image.shape)

    axes[i].imshow(image_reconstructed, cmap='gray')
    axes[i].set_title(f'{n_components} components')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 13: Ví dụ thực tế: Phát hiện bất ngờ

PCA có thể được sử dụng để phát hiện sự cố bất ngờ bằng cách xác định dữ liệu không phù hợp với PCA chiều rộng không lớn hơn. Ví dụ này trình bày cách sử dụng PCA để phát hiện những điểm bất ngờ trong dữ liệu về thông số cảm xúc.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate normal data and anomalies
np.random.seed(42)
n_samples = 1000
n_features = 10

# Normal data
X_normal = np.random.randn(n_samples, n_features)

# Anomalies
n_anomalies = 50
X_anomalies = np.random.randn(n_anomalies, n_features) * 2 + 5

# Combine normal data and anomalies
X = np.vstack((X_normal, X_anomalies))

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Calculate reconstruction error
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

# Set threshold for anomaly detection (e.g., 95th percentile)
threshold = np.percentile(reconstruction_error, 95)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:n_samples, 0], X_pca[:n_samples, 1], c='blue', label='Normal')
plt.scatter(X_pca[n_samples:, 0], X_pca[n_samples:, 1], c='red', label='Anomaly')
plt.scatter(X_pca[reconstruction_error > threshold, 0],
            X_pca[reconstruction_error > threshold, 1],
            c='green', s=100, alpha=0.5, label='Detected Anomaly')
plt.legend()
plt.title('PCA for Anomaly Detection')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

print(f"Number of detected anomalies: {np.sum(reconstruction_error > threshold)}")
```

Trang trình bày 14: Tài nguyên bổ sung

Để hiểu sâu hơn về PCA và các ứng dụng của nó trong máy học, hãy xem xét khám phá các tài nguyên sau:

1. "Hướng dẫn phân tích thành phần chính" của Jonathon Shlens ArXiv URL: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100) Bài viết này giới thiệu toàn diện về PCA, bao gồm các nền tảng học toán và ứng dụng thực tế của nó.
2. "Phân tích thành phần chính: Đánh giá và những phát triển gần đây" của Ian T. Jolliffe và Jorge Cadima ArXiv URL: [https://arxiv.org/abs/1511.03677](https://arxiv.org/abs/1511.03677) Bài đánh giá này thảo luận về tiến bộ gần đây trong PCA, bao gồm các biến thể mạnh mẽ và rụng tóc của kỹ thuật.
3. Tài liệu Scikit-learn về PCA Tài liệu chính thức này cung cấp các ví dụ thực tế và giải pháp chi tiết về việc phát triển PCA trong scikit-learn.
4. "Nhận dạng mẫu và học máy" của Christopher M. Bishop Sách giáo khoa này bao gồm công việc xử lý kỹ thuật lưỡng PCA trong bối cảnh máy học và nhận thống kê kê mẫu.
5. "Các yếu tố của việc học thống kê" của Trevor Hastie, Robert Tibshirani và Jerome Friedman Cuốn sách toàn diện này đề cập đến PCA và các mối quan hệ của nó với các kỹ thuật học thống kê khác.

Tài nguyên này bao gồm các từ hướng dẫn giới thiệu đến thảo luận nâng cao, cung cấp hiểu biết toàn diện về PCA cũng như vai trò của nó trong máy học và phân tích dữ liệu.
