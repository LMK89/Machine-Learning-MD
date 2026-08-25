## Phân tích thành phần chính (PCA) trong Python
Slide 1: Giới thiệu về Phân tích thành phần chính (PCA)

Phân tích thành phần chính (PCA) là một kỹ thuật giảm kích thước mạnh mẽ được sử dụng trong phân tích dữ liệu và máy học. Nó giúp xác định các mẫu trong nhiều dữ liệu chiều bằng cách chuyển đổi nó thành một hệ thống mới trong đó biểu hiện trục trặc của sai sót đa phương tiện tối đa.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
X = np.dot(X, [[2, 1], [1, 2]])  # Introduce correlation

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot original data and principal components
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # Scale component by its variance explanation power
    plt.arrow(0, 0, comp[0], comp[1], color=f'C{i+1}', alpha=0.8, width=0.05)
    plt.text(comp[0], comp[1], f'PC{i+1}', color=f'C{i+1}')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA: Original Data and Principal Components')
plt.axis('equal')
plt.show()
```

Slide 2: Cơ sở toán học của PCA

PCA dựa trên các khái niệm riêng biệt và giá trị riêng. Nó tìm kiếm các hướng dẫn (vector riêng) trong đó dữ liệu thay đổi nhiều nhất và hướng này trở thành thành phần chính. phương pháp không thích hợp được giải quyết bởi mỗi thành phần được chọn bởi giá trị riêng tương ứng của nó.

```python
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)

# Calculate covariance matrix
cov_matrix = np.cov(X.T)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvectors by decreasing eigenvalues
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Covariance Matrix:")
print(cov_matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
```

Trang trình bày 3: Triển khai PCA từ đầu

Vui lòng khai báo PCA từng bước để hiểu hoạt động bên trong của nó. Chúng tôi sẽ tạo một tập dữ liệu đơn giản, căn giữa, tính toán ma trận Hiệp phương sai và sau đó tìm các thành phần chính.

```python
import numpy as np

def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    top_eigenvectors = eigenvectors[:, :n_components]

    # Project data onto principal components
    X_pca = X_centered.dot(top_eigenvectors)

    return X_pca, top_eigenvectors, eigenvalues

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)

# Apply PCA
X_pca, components, explained_variance = pca_from_scratch(X, n_components=2)

print("Transformed data shape:", X_pca.shape)
print("Principal components shape:", components.shape)
print("Explained variance:", explained_variance[:2])
```

Slide 4: sai phương pháp và chọn một số thành phần Giải thích

Một khía cạnh quan trọng của PCA được xác định có bao nhiêu thành phần chính cần giữ lại. Quyết định này thường dựa trên tỷ lệ giải nén tích lũy của phương pháp, tỷ lệ này cho phép họ biết tổng phương pháp sai trong dữ liệu được ghi lại bởi một số tối đa định nghĩa cụ thể.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 10)

# Perform PCA
pca = PCA()
pca.fit(X)

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance ratio
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

# Find number of components for 95% variance explained
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components for 95% variance explained: {n_components_95}")
```

Trang trình bày 5: PCA để giảm kích thước

Một trong những ứng dụng chính của PCA là giảm kích thước. Bằng cách tham chiếu chiều cao của dữ liệu không có chiều thấp hơn, chúng tôi có thể giảm mức độ phức tạp của dữ liệu trong khi vẫn giữ được phần quan trọng của thông tin đó.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Digits Dataset Projected onto First Two Principal Components')
plt.show()

print(f"Original data shape: {X.shape}")
print(f"Reduced data shape: {X_pca.shape}")
```

Trang trình bày 6: PCA để trực tiếp hóa dữ liệu

PCA là một công cụ tuyệt vời để hiển thị dữ liệu nhiều chiều theo hai chiều hoặc ba chiều. Điều này có thể giúp chúng tôi xác định các mẫu, cụm hoặc ngoại lệ có thể không rõ ràng trong khoảng thời gian cấm đầu nhiều chiều.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Perform PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.7)
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')
plt.title('Iris Dataset Projected onto First Three Principal Components')
plt.colorbar(scatter)
plt.show()
```

Trang trình bày 7: PCA để giảm tiếng ồn

PCA có thể được sử dụng để giảm nhiễu trong dữ liệu bằng cách giả sử rằng các thành phần chính có phương pháp nhỏ nhất tương ứng với nhiễu. Bằng cách xây dựng lại dữ liệu chỉ bằng cách sử dụng chính hàng đầu của các thành phần, chúng tôi có thể loại bỏ một số nhiễu.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate noisy sinusoidal data
np.random.seed(42)
t = np.linspace(0, 10, 1000)
x = np.sin(t) + 0.1 * np.random.randn(1000)

# Reshape data for PCA
X = np.column_stack((t, x))

# Perform PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_pca)

# Plot results
plt.figure(figsize=(12, 4))
plt.plot(t, x, 'b.', alpha=0.3, label='Noisy data')
plt.plot(X_reconstructed[:, 0], X_reconstructed[:, 1], 'r-', label='PCA reconstruction')
plt.legend()
plt.title('PCA for Noise Reduction')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
```

Trang trình bày 8: PCA và tầm quan trọng của các tính năng

PCA có thể giúp chúng tôi hiểu những tính năng gốc đóng góp nhiều nhất cho các thành phần chính. Thông tin này có thể có giá trị khi lựa chọn các tính năng và giải pháp thích hợp cho dữ liệu của chúng tôi.

```python
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Load Boston Housing dataset
boston = load_boston()
X, feature_names = boston.data, boston.feature_names

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
pca.fit(X_scaled)

# Create a DataFrame of feature importances
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(pca.components_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Absolute Importance in First Principal Component')
plt.title('Feature Importance in Boston Housing Dataset')
plt.tight_layout()
plt.show()
```

Trang trình bày 9: PCA để nén hình ảnh

PCA có thể được sử dụng để nén hình ảnh bằng cách giảm kích thước của dữ liệu hình ảnh. Kỹ thuật này có thể đặc biệt hữu ích cho các hình ảnh thang độ xám, trong đó mỗi pixel được biểu thị bằng một giá trị duy nhất.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load sample image (replace with your own image path)
image = plt.imread('sample_image.jpg')
gray_image = np.mean(image, axis=2)  # Convert to grayscale

# Perform PCA
pca = PCA(0.95)  # Keep 95% of variance
image_pca = pca.fit_transform(gray_image)
image_reconstructed = pca.inverse_transform(image_pca)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(gray_image, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(image_reconstructed, cmap='gray')
ax2.set_title(f'Reconstructed Image\n{pca.n_components_} components')
ax2.axis('off')
plt.tight_layout()
plt.show()

print(f"Original image shape: {gray_image.shape}")
print(f"Compressed image shape: {image_pca.shape}")
print(f"Compression ratio: {gray_image.size / image_pca.size:.2f}")
```

Trang trình bày 10: PCA và phát hiện ngoại lệ

PCA có thể được sử dụng để phát hiện các ngoại lệ trong dữ liệu đa biến. Bằng cách tham chiếu dữ liệu lên các thành phần chính và kiểm tra lỗi tái tạo, chúng tôi có thể xác định các điểm không phù hợp với cấu trúc tổng hợp của dữ liệu.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

# Generate sample data with outliers
np.random.seed(42)
X = np.random.randn(100, 2)
X = np.dot(X, [[2, 1], [1, 2]])
outliers = np.random.uniform(low=-10, high=10, size=(5, 2))
X = np.vstack([X, outliers])

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Calculate reconstruction error
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)

# Use Elliptic Envelope for comparison
ee = EllipticEnvelope(contamination=0.1, random_state=42)
outlier_labels = ee.fit_predict(X)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=reconstruction_error, cmap='viridis')
plt.colorbar(label='Reconstruction Error')
plt.title('PCA Reconstruction Error')

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=outlier_labels, cmap='viridis')
plt.title('Elliptic Envelope Outlier Detection')

plt.tight_layout()
plt.show()
```

Trang trình bày 11: PCA cho phân tích chuỗi thời gian

PCA có thể được áp dụng cho dữ liệu chuỗi thời gian để xác định các mô hình hoặc hướng dẫn cơ sở. Kỹ thuật này đặc biệt hữu ích khi xử lý nhiều chuỗi thời gian liên kết.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
trend = np.linspace(0, 10, len(dates))
seasonality = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
noise = np.random.randn(len(dates))

ts1 = trend + seasonality + noise
ts2 = 0.5 * trend + 2 * seasonality + noise
ts3 = -0.3 * trend + 0.5 * seasonality + noise

# Combine time series into a DataFrame
df = pd.DataFrame({'TS1': ts1, 'TS2': ts2, 'TS3': ts3}, index=dates)

# Perform PCA
pca = PCA()
components = pca.fit_transform(df)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

df.plot(ax=ax1)
ax1.set_title('Original Time Series')

pd.DataFrame(components, index=dates, columns=['PC1', 'PC2', 'PC3']).plot(ax=ax2)
ax2.set_title('Principal Components')

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Trang trình bày 12: PCA và Phân tích tương quan

PCA có thể tiết lộ mối tương quan giữa các biến trong dữ liệu. Bằng cách kiểm tra mức tải của các thành phần chính, chúng tôi có thể xác định các nhóm biến có xu hướng thay đổi giống nhau.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Generate correlated data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 5)
X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.5
X[:, 2] = X[:, 1] + np.random.randn(n_samples) * 0.5
X[:, 3] = np.random.randn(n_samples)
X[:, 4] = X[:, 3] + np.random.randn(n_samples) * 0.5

# Perform PCA
pca = PCA()
pca.fit(X)

# Create a DataFrame of the PCA loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(5)],
    index=[f'Var{i+1}' for i in range(5)]
)

# Plot the heatmap of loadings
plt.figure(figsize=(10, 8))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title('PCA Loadings Heatmap')
plt.tight_layout()
plt.show()

# Print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Trang trình bày 13: PCA để phát hiện sự bất thường trong dữ liệu biến

PCA có thể được sử dụng để phát hiện những điểm bất thường trong đa biến dữ liệu bằng cách xác định các điểm dữ liệu sai lệch đáng kể đối với các thành phần chính.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate synthetic sensor data
np.random.seed(42)
n_samples = 1000
n_sensors = 5

normal_data = np.random.randn(n_samples, n_sensors)
anomalies = np.random.uniform(low=-10, high=10, size=(10, n_sensors))
data = np.vstack([normal_data, anomalies])

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# Calculate reconstruction error
reconstructed = pca.inverse_transform(pca_result)
mse = np.mean(np.square(data - reconstructed), axis=1)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=mse, cmap='viridis')
plt.colorbar(label='Reconstruction Error')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA for Anomaly Detection in Sensor Data')
plt.tight_layout()
plt.show()

# Identify potential anomalies
threshold = np.percentile(mse, 99)
anomalies = np.where(mse > threshold)[0]
print(f"Potential anomalies: {anomalies}")
```

Trang trình bày 14: PCA trong các hình ảnh được nhận dạng: Khuôn mặt riêng

PCA được sử dụng trong hệ thống nhận dạng khuôn mặt nhận dạng khuôn mặt để tạo ra một khuôn mặt khuôn mặt riêng biệt, là thành phần chính của một tập hợp hình ảnh khuôn mặt khuôn mặt. Những mặt hàng này có thể được sử dụng để biểu diễn diễn đàn và nhận dạng khuôn mặt một cách hiệu quả.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

# Load face dataset
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data
y = faces.target

# Perform PCA
n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X)

# Plot the first few eigenfaces
fig, axes = plt.subplots(3, 5, figsize=(15, 9),
                         subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(faces.images[0].shape),
              cmap='gray')
    ax.set_title(f'Eigenface {i+1}')
plt.tight_layout()
plt.show()

# Print explained variance ratio
print("Cumulative explained variance ratio:",
      np.sum(pca.explained_variance_ratio_))
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về Phân tích thành phần chính và các ứng dụng của nó, thì đây là một số tài nguyên có giá trị:

1. Bài viết ArXiv: " Hướng dẫn về phân tích thành phần chính" của Jonathon Shlens URL: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
2. Bài viết ArXiv: "Phân tích thành phần chính:Đánh giá và những phát triển gần đây" của Hervé Abdi và Lynne J. Williams URL: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
3. Tài liệu Scikit-learn về PCA: [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
4. Khóa học Coursera: "Machine Learning" của Andrew Ng, trình bày chuyên sâu về PCA
5. Sách: "Nhận dạng mẫu và học máy" của Christopher M. Bishop, cung cấp cách xử lý toán học kỹ thuật lưỡng về PCA

Những tài nguyên này cung cấp sự hợp lý giữa nền tảng lý thuyết và ứng dụng thực tế của PCA, phù hợp với người học ở nhiều cấp độ chuyên môn khác nhau.
