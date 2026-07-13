## Động lực sử dụng KernelPCA thay vì PCA
Slide 1: Giới thiệu về KernelPCA và PCA

Phân tích thành phần chính (PCA) và Kernel PCA là các kỹ thuật giảm kích thước được sử dụng trong máy học. Trong khi PCA là một phương pháp tuyến tính tính, KernelPCA khái niệm mở rộng này đã hát các mối quan hệ phi tuyến tính. Bài trình bày này sẽ khám phá động lực đằng sau việc sử dụng KernelPCA thay vì PCA, cung cấp các ví dụ về mã hóa hóa và ứng dụng thực tế.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA

# Generate sample data
np.random.seed(42)
X = np.random.randn(200, 2)
X[:100, 0] = 2 + 0.5 * X[:100, 0]
X[100:, 0] = -2 + 0.5 * X[100:, 0]

# Plot the original data
plt.scatter(X[:, 0], X[:, 1], c='b', alpha=0.5)
plt.title("Original Data")
plt.show()
```

Trang trình bày 2: Tìm hiểu PCA

PCA tìm kiếm mức tối đa theo hướng của các phương pháp sai lệch trong dữ liệu có chiều cao và chiếu nó vào không gian có chiều thấp hơn. Nó hoạt động tốt với dữ liệu có thể phân tách tuyến tính nhưng lại gặp khó khăn với các mối quan hệ phi tuyến tính.

```python
# Apply PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# Plot PCA results
plt.scatter(X[:, 0], X[:, 1], c='b', alpha=0.5)
plt.plot(X_pca, np.zeros_like(X_pca), 'ro', alpha=0.5)
plt.title("PCA Projection")
plt.show()
```

Slide 3: Các chế độ của PCA

PCA giả định mối quan hệ tuyến tính giữa các tính năng. Khi xử lý dữ liệu tuyến tính năng, PCA không thể nhận được kết quả đầu ra về cấu trúc. Chế độ này được cung cấp bằng cách sử dụng KernelPCA cho bộ đệm dữ liệu phức tạp hơn.

```python
# Generate non-linear data
theta = np.linspace(0, 2*np.pi, 200)
X_nonlinear = np.column_stack([np.cos(theta) + 0.1*np.random.randn(200),
                               np.sin(theta) + 0.1*np.random.randn(200)])

# Apply PCA to non-linear data
pca_nonlinear = PCA(n_components=1)
X_pca_nonlinear = pca_nonlinear.fit_transform(X_nonlinear)

# Plot results
plt.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c='b', alpha=0.5)
plt.plot(X_pca_nonlinear, np.zeros_like(X_pca_nonlinear), 'ro', alpha=0.5)
plt.title("PCA on Non-linear Data")
plt.show()
```

Slide 4: Giới thiệu về KernelPCA

KernelPCA giải quyết các giới hạn của PCA bằng cách sử dụng "thủ thuật hạt nhân". Điều này cho phép nó thực hiện giảm kích thước phi tuyến tính, nắm bắt các mối quan hệ phức tạp trong dữ liệu mà PCA có thể bỏ lỡ.

```python
# Apply KernelPCA with RBF kernel
kpca = KernelPCA(n_components=1, kernel='rbf')
X_kpca = kpca.fit_transform(X_nonlinear)

# Plot KernelPCA results
plt.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c='b', alpha=0.5)
plt.scatter(X_kpca, np.zeros_like(X_kpca), c='r', alpha=0.5)
plt.title("KernelPCA Projection")
plt.show()
```

Trang trình bày 5: Thủ thuật hạt nhân

KernelPCA cho phép kernel thủ thuật chèn đầu dữ liệu vào không có chiều cao hơn mà không cần phải tính toán rõ ràng việc chuyển đổi. Điều này cho phép nắm bắt các mối quan hệ phi tuyến tính một cách hiệu quả.

```python
def rbf_kernel(X, Y, gamma=1):
    """Compute the RBF (Gaussian) kernel between X and Y"""
    X_norm = np.sum(X**2, axis=1)
    Y_norm = np.sum(Y**2, axis=1)
    K = np.exp(-gamma * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)))
    return K

# Compute and visualize the kernel matrix
K = rbf_kernel(X_nonlinear, X_nonlinear)
plt.imshow(K, cmap='viridis')
plt.colorbar()
plt.title("RBF Kernel Matrix")
plt.show()
```

Trang trình bày 6: Chọn hạt nhân phù hợp

Hiệu suất của KernelPCA phụ thuộc vào việc lựa chọn hợp lý các chức năng của kernel. Các hạt nhân phổ biến bao gồm RBF (Gaussian), đa thức và sigmoid. Việc lựa chọn hạt nhân ảnh hưởng đến kỹ thuật nắm bắt các mối quan hệ phi tuyến tính.

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': np.logspace(-3, 3, 7),
    'degree': [2, 3, 4]  # Only used by poly kernel
}

# Perform grid search
grid_search = GridSearchCV(KernelPCA(n_components=1), param_grid, cv=5)
grid_search.fit(X_nonlinear)

print("Best parameters:", grid_search.best_params_)
```

Slide 7: Độ phức tạp tính toán

Mặc dù KernelPCA có thể nắm bắt được các quan hệ phi tuyến tính nhưng không có chi phí tính toán cao hơn PCA. Độ phức tạp về thời gian của KernelPCA là O(n^3), trong đó n là lượng mẫu, gây khó khăn cho dữ liệu lớn.

```python
import time

def compare_runtime(X, n_components=2):
    start_time = time.time()
    PCA(n_components=n_components).fit(X)
    pca_time = time.time() - start_time

    start_time = time.time()
    KernelPCA(n_components=n_components, kernel='rbf').fit(X)
    kpca_time = time.time() - start_time

    print(f"PCA runtime: {pca_time:.4f} seconds")
    print(f"KernelPCA runtime: {kpca_time:.4f} seconds")

# Generate larger dataset
X_large = np.random.randn(1000, 50)
compare_runtime(X_large)
```

Slide 8: Xử lý tiền cho KernelPCA

Công việc xử lý trước dữ liệu phù hợp là rất quan trọng đối với KernelPCA. Việc chia tỷ lệ các tính năng đầu tiên để đảm bảo rằng tất cả các nguyên tử đều đóng góp như nhau trong quá trình tính toán hạt nhân, không cho bất kỳ tính năng đơn lẻ nào phân phối quá trình phân tích.

```python
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_nonlinear)

# Apply KernelPCA to scaled data
kpca_scaled = KernelPCA(n_components=1, kernel='rbf')
X_kpca_scaled = kpca_scaled.fit_transform(X_scaled)

# Plot results
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='b', alpha=0.5)
plt.scatter(X_kpca_scaled, np.zeros_like(X_kpca_scaled), c='r', alpha=0.5)
plt.title("KernelPCA on Scaled Data")
plt.show()
```

Trang trình bày 9: KernelPCA giải thích kết quả

Công việc giải quyết kết quả KernelPCA có thể gặp khó khăn do tính chất phi tuyến tính của biến đổi được phép. Các kỹ thuật trực quan hóa như biểu tượng phân tích dữ liệu được chuyển đổi hoặc sai tỷ lệ phân tích phương pháp có thể giúp hiểu được kết quả.

```python
# Compute explained variance ratio
kpca_multi = KernelPCA(n_components=2, kernel='rbf')
X_kpca_multi = kpca_multi.fit_transform(X_scaled)

explained_variance_ratio = kpca_multi.eigenvalues_ / np.sum(kpca_multi.eigenvalues_)

plt.bar(range(1, 3), explained_variance_ratio)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio of KernelPCA Components")
plt.show()
```

Slide 10: Ví dụ thực tế: Nhận dạng chữ viết tay

KernelPCA có thể đặc biệt hữu ích trong các tác vụ xử lý hình ảnh, nghĩ ra giới hạn như nhận dạng chữ viết tay. Nó có thể chụp các mẫu phi tuyến tính ở cường độ pixel mà PCA tuyến tính có thể bỏ lỡ.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load digits dataset
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)

# Apply KernelPCA
kpca_digits = KernelPCA(n_components=50, kernel='rbf')
X_train_kpca = kpca_digits.fit_transform(X_train)
X_test_kpca = kpca_digits.transform(X_test)

# Train and evaluate SVM classifier
svm = SVC()
svm.fit(X_train_kpca, y_train)
y_pred = svm.predict(X_test_kpca)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

Trang trình bày 11: Ví dụ thực tế: Nhận rõ ràng dạng dạng

KernelPCA có thể mang lại hiệu quả trong các nhiệm vụ nhận dạng biểu cảm khuôn mặt, trong đó mối liên hệ giữa các khuôn mặt đặc điểm và biểu thức thường là tính chất phi tuyến. Nó có thể giúp trích xuất các đặc điểm có ý nghĩa từ hình ảnh khuôn mặt.

```python
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Olivetti faces dataset
faces = fetch_olivetti_faces()
X_faces, y_faces = faces.data, faces.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_faces, y_faces, test_size=0.2, random_state=42)

# Apply KernelPCA
kpca_faces = KernelPCA(n_components=100, kernel='rbf')
X_train_kpca = kpca_faces.fit_transform(X_train)
X_test_kpca = kpca_faces.transform(X_test)

# Train and evaluate SVM classifier
svm = SVC()
svm.fit(X_train_kpca, y_train)
y_pred = svm.predict(X_test_kpca)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Visualize original and transformed faces
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(X_test[i].reshape(64, 64), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(X_test_kpca[i].reshape(10, 10), cmap='gray')
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()
```

Slide 12: Công thức và cân nhắc

Mặc dù KernelPCA mang lại những lợi ích như vậy cho PCA đối với dữ liệu phi tuyến tính nhưng nó cũng thiết lập các công thức nhỏ. Chúng bao gồm phức tạp tính toán ngày càng tăng, khó khăn trong công việc đơn giản hạt nhân và tham số phù hợp cũng như khả năng trang bị xử lý trên các dữ liệu nhỏ.

```python
# Demonstrate overfitting with small dataset
X_small = X_nonlinear[:20]
y_small = np.array([0]*10 + [1]*10)

kpca_small = KernelPCA(n_components=2, kernel='rbf')
X_small_kpca = kpca_small.fit_transform(X_small)

plt.scatter(X_small_kpca[:10, 0], X_small_kpca[:10, 1], c='r', label='Class 0')
plt.scatter(X_small_kpca[10:, 0], X_small_kpca[10:, 1], c='b', label='Class 1')
plt.legend()
plt.title("KernelPCA on Small Dataset (Potential Overfitting)")
plt.show()
```

Trang trình bày 13: Khi nào nên sử dụng KernelPCA

KernelPCA đặc biệt hữu ích khi xử lý các dữ liệu tuyến tính phi tuyến tính, phức tạp trong đó tính năng tuyến tính PCA không thể được cấu hình cơ sở dữ liệu. Nó có giá trị trong các lĩnh vực như xử lý hình ảnh, tin sinh học và bất kỳ lĩnh vực nào mà mối liên hệ liên hệ quan hệ dữ liệu vốn dĩ là phi tuyến tính.

```python
# Generate and visualize a complex dataset
t = np.linspace(0, 4*np.pi, 500)
X_complex = np.column_stack([
    t*np.cos(t) + 0.5*np.random.randn(500),
    t*np.sin(t) + 0.5*np.random.randn(500)
])

plt.scatter(X_complex[:, 0], X_complex[:, 1], c=t, cmap='viridis')
plt.title("Complex Non-linear Dataset")
plt.colorbar(label='t')
plt.show()

# Apply PCA and KernelPCA
pca_complex = PCA(n_components=1)
kpca_complex = KernelPCA(n_components=1, kernel='rbf')

X_pca_complex = pca_complex.fit_transform(X_complex)
X_kpca_complex = kpca_complex.fit_transform(X_complex)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(X_complex[:, 0], X_complex[:, 1], c=X_pca_complex, cmap='viridis')
ax1.set_title("PCA Projection")
ax2.scatter(X_complex[:, 0], X_complex[:, 1], c=X_kpca_complex, cmap='viridis')
ax2.set_title("KernelPCA Projection")
plt.tight_layout()
plt.show()
```

Slide 14: Kết luận và định hướng tương lai

Khả năng mở rộng KernelPCA của PCA để xử lý các tính năng phi dữ liệu tuyến tính, nó trở thành một công cụ mạnh mẽ trong máy học và phân tích dữ liệu. Khi các bộ dữ liệu ngày càng phức tạp, các kỹ thuật như KernelPCA ngày càng trở nên có giá trị. Nghiên cứu trong tương lai có thể nghiên cứu công việc tối ưu hóa KernelPCA cho mô-đun dữ liệu lớn và phát triển nhân viên mới cho các công cụ ứng dụng.

```python
# Demonstrate KernelPCA with custom kernel
def custom_kernel(X, Y):
    return np.tanh(np.dot(X, Y.T) + 1)

kpca_custom = KernelPCA(n_components=2, kernel=custom_kernel)
X_kpca_custom = kpca_custom.fit_transform(X_complex)

plt.scatter(X_kpca_custom[:, 0], X_kpca_custom[:, 1], c=t, cmap='viridis')
plt.title("KernelPCA with Custom Kernel")
plt.colorbar(label='t')
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về KernelPCA và các ứng dụng của nó, nên sử dụng các tài nguyên sau:

1. “Phân tích thành phần chính hạt nhân” của Bernhard Schölkopf, Alexander Smola và Klaus-Robert Müller (1998). Có tại: [https://arxiv.org/abs/1207.3538](https://arxiv.org/abs/1207.3538)
2. "Hướng dẫn về hỗ trợ máy chủ để nhận dạng mẫu" của Christopher J.C. Burges (1998). Có tại: [https://www.microsoft.com/en-us/research/publication/a-tutorial-on-support-vector-machines-for-pattern-recognition/](https://www.microsoft.com/en-us/research/publication/a-tutorial-on-support-vector-machines-for-pattern-recognition/)
3. "Phân tích thành phần phi tuyến như một vấn đề về giá trị riêng của nhân" của Bernhard Schölkopf, Alexander Smola
