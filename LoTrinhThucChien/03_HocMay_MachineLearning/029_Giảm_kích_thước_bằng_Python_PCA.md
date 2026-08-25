## Giảm kích thước bằng Python (PCA)
Slide 1: Giới thiệu về Giảm kích thước

Kích thước giảm quá trình giảm số lượng hoặc biến trong dữ liệu trong khi vẫn giữ nguyên thông tin liên kết. Điều đặc biệt hữu ích này trong máy học khi xử lý dữ liệu nhiều chiều có thể dẫn đến các công thức về chiều và tính toán.

Mã số:

```python
# No code for the introduction
```

Slide 2: Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a popular dimensionality reduction technique that transforms the data into a new coordinate system, where the new axes (principal components) are orthogonal and ordered by the amount of variance they explain in the data.

Code:

```python
from sklearn.decomposition import PCA

# Create a PCA object
pca = PCA(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = pca.fit_transform(X)
```

Slide 3: Tiêu chuẩn hóa và định tâm trung bình

Trước khi áp dụng PCA, điều kiện cần thiết là chuẩn hóa dữ liệu bằng cách trừ giá trị trung bình và chia cho độ lệch. Điều này đảm bảo rằng tất cả các đặc điểm đều có cùng tỷ lệ và các đặc điểm có phương pháp sai lớn hơn việc sử dụng ưu tiên trong phân tích.

Mã số:

```python
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA on the scaled data
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X_scaled)
```

Slide 4: thích sai tỷ lệ phương pháp giải thích

PCA cung cấp phương pháp tỷ lệ không được giải quyết thích hợp, biểu thị phương pháp tỷ lệ sai trong dữ liệu gốc được lưu giữ bởi từng thành phần chính. Thông tin này có thể được sử dụng để xác định số lượng thành phần cần giữ lại để giảm kích thước.

Mã số:

```python
# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Sum of the explained variance ratios (should be close to 1)
print(sum(explained_variance_ratio))
```

Slide 5: Trực quan hóa các thành phần chính

Các thành phần chính có thể được hiển thị trực quan để hiểu mối liên hệ giữa các tính năng và phân tích bổ sung dữ liệu khi không thể chuyển đổi.

Mã số:

```python
import matplotlib.pyplot as plt

# Plot the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```

Trang trình bày 6: Giảm kích thước bằng PCA

Sau khi xác định số lượng thành phần chính cần giữ lại, PCA có thể được sử dụng để chuyển đổi dữ liệu thành không có chiều sâu hơn, giúp giảm số lượng tính năng một cách hiệu quả.

Mã số:

```python
# Reduce to 3 dimensions
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_scaled)

# Shape of the reduced data
print(X_reduced.shape)
```

Slide 7: PCA cho trực quan hóa

PCA có thể được sử dụng để trực tiếp hóa dữ liệu nhiều chiều bằng cách chiếu nó lên không có chiều thấp hơn, thường là 2D hoặc 3D, để hiểu và khám phá tốt hơn.

Mã số:

```python
# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_scaled)

# Plot the data in 2D
plt.scatter(X_vis[:, 0], X_vis[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```

Slide 8: PCA để trích xuất tính năng

PCA có thể được sử dụng như một kỹ thuật trích xuất đặc trưng, ​​​​trong đó bản thân các thành phần chính sẽ trở thành các đặc điểm mới. Điều này có thể hữu ích để giảm chiều của dữ liệu trong khi vẫn đảm bảo an toàn cho các thông tin liên kết.

Mã số:

```python
# Reduce to 5 principal components
pca = PCA(n_components=5)
X_features = pca.fit_transform(X_scaled)

# Use the transformed data (X_features) as input to a machine learning model
```

Trang trình bày 9: PCA tăng dần

PCA tăng dần là một biến thể của PCA cho phép cập nhật hiệu quả khi dữ liệu mới được thêm vào dữ liệu mà không cần phải tính toán lại các thành phần chính từ đầu.

Mã số:

```python
from sklearn.decomposition import IncrementalPCA

# Initialize the Incremental PCA object
ipca = IncrementalPCA(n_components=2, batch_size=100)

# Partial fit on the first batch of data
ipca.partial_fit(X[:100])

# Partial fit on the next batch of data
ipca.partial_fit(X[100:200])

# Transform the data
X_transformed = ipca.transform(X)
```

Trình bày 10: PCA hạt nhân

Kernel PCA là một phần mở rộng tính năng phi tuyến của PCA, có thể nắm bắt các mối quan hệ phi tuyến tính tính trong dữ liệu bằng cách sử dụng cơ sở dữ liệu ánh xạ vào không gian đặc biệt có chiều cao hơn và sau đó áp dụng tính năng tuyến tính PCA trong không gian.

Mã số:

```python
from sklearn.decomposition import KernelPCA

# Initialize the Kernel PCA object
kpca = KernelPCA(n_components=2, kernel='rbf')

# Fit and transform the data
X_transformed = kpca.fit_transform(X)
```

Trang trình bày 11: PCA để giảm tiếng ồn

PCA có thể được sử dụng để giảm nhiễu bằng cách tham chiếu dữ liệu lên không có chiều thấp hơn được hiển thị thành các thành phần chính, loại bỏ nhiễu do các thành phần bị loại bỏ.

Mã số:

```python
# Reduce to 10 principal components
pca = PCA(n_components=10)
X_denoised = pca.fit_transform(X_noisy)

# Reconstruct the denoised data
X_reconstructed = pca.inverse_transform(X_denoised)
```

Trang trình bày 12: PCA để phát hiện ngoại lệ

PCA có thể được sử dụng để phát hiện ngoại lệ bằng cách xác định các phương pháp tạo lỗi tái tạo dữ liệu lớn khi tham chiếu không gian với thành phần chính.

Mã số:

```python
from sklearn.decomposition import PCA

# Initialize the PCA object
pca = PCA(n_components=5)

# Fit and transform the data
X_transformed = pca.fit_transform(X)

# Reconstruct the data
X_reconstructed = pca.inverse_transform(X_transformed)

# Calculate the squared reconstruction error
squared_errors = ((X - X_reconstructed) ** 2).sum(axis=1)

# Identify outliers based on a threshold
threshold = np.percentile(squared_errors, 95)
outliers = np.where(squared_errors > threshold)[0]
```

Trang trình bày 13: PCA để nén dữ liệu

PCA có thể được sử dụng để nén dữ liệu bằng cách giữ lại các phần chính được thu thập khác biệt trong dữ liệu, giảm thiểu hiệu quả của các yêu cầu lưu trữ hoặc truyền tải.

Mã số:

```python
# Reduce to 10 principal components
pca = PCA(n_components=10)
X_compressed = pca.fit_transform(X)

# Reconstruct the compressed data
X_reconstructed = pca.inverse_transform(X_compressed)
```

Trang trình bày 14 (Tài nguyên bổ sung): Tài nguyên bổ sung

Để khám phá và tìm hiểu thêm, đây là một số tài nguyên được xuất bản từ ​​arXiv.org:

1. " Hướng dẫn phân tích thành phần chính" của Jonathon Shlens ([https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100))
2. "Phân tích thành phần chính trong thiết kế sơ đồ tích hợp tuyến tính" của Robert Dworkin, Dan Knebel ([https://arxiv.org/abs/1904.02120](https://arxiv.org/abs/1904.02120))
3. "Phân tích thành phần chính: Kỹ thuật nhận dạng mẫu hình ảnh mạnh mẽ" của Kazi A. Kalpoma ([https://arxiv.org/abs/1704.04392](https://arxiv.org/abs/1704.04392))
