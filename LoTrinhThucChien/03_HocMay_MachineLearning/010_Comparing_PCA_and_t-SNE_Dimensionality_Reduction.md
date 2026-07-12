## So sánh PCA và t-SNE giảm kích thước

Trang trình bày 1: Sự khác biệt chính giữa PCA và t-SNE

Phân tích thành phần chính (PCA) và Nhúng hàng ngẫu nhiên phân phối t (t-SNE) là hai kỹ thuật giảm kích thước phổ biến được sử dụng trong khoa học dữ liệu và học máy. Mặc dù cả hai đều hướng tới mục tiêu giảm chiều của dữ liệu nhiều chiều nhưng chúng khác nhau đáng kể về cách tiếp cận và ứng dụng. Bài trình bày này sẽ khám phá những điểm khác biệt chính giữa PCA và t-SNE, cung cấp thông tin chi tiết về thời điểm sử dụng từng phương pháp.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate sample data
np.random.seed(42)
data = np.random.randn(1000, 50)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
ax1.set_title('PCA')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
ax2.set_title('t-SNE')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Trang trình bày 2: Tuyến tính và phi tuyến tính

PCA là một kỹ thuật giảm kích thước tuyến tính, giả định mối liên hệ giữa các biến là tuyến tính. Nó hoạt động bằng cách tìm kiếm mức tối đa sai lệch phương tiện theo hướng trong dữ liệu và tham chiếu dữ liệu lên các hướng này. Ngược lại, t-SNE là một kỹ thuật phi tuyến tính có thể nắm bắt các mối quan hệ phi tuyến tính phức tạp trong dữ liệu. Điều này làm cho t-SNE phù hợp hơn để tiết lộ các cấu trúc hỗn hợp dữ liệu mà các phương pháp tuyến tính như PCA có thể bỏ qua.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate non-linear data (Swiss roll dataset)
n_points = 1000
X = np.zeros((n_points, 3))
t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_points))
X[:, 0] = t * np.cos(t)
X[:, 1] = 21 * np.random.rand(n_points)
X[:, 2] = t * np.sin(t)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.scatter(X[:, 0], X[:, 2], c=t, cmap='viridis')
ax1.set_title('Original Swiss Roll')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis')
ax2.set_title('PCA')

ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t, cmap='viridis')
ax3.set_title('t-SNE')

plt.tight_layout()
plt.show()
```

Trang trình bày 3: Cấu hình toàn cầu và cục bộ

PCA tập trung vào công việc duy trì cấu trúc tổng thể của dữ liệu bằng cách tối đa hóa phương pháp sai theo chiều dọc theo từng thành phần chính. Cách tiếp cận này có kết quả hiệu quả để nắm bắt các xu hướng và tổng hợp mẫu trong dữ liệu. Mặt khác, t-SNE ưu tiên duy trì các mối quan hệ cục bộ, giữ các dữ liệu tương tự gần nhau trong không gian được giảm bớt. Trọng tâm cục bộ này cho phép t-SNE tiết lộ các cụm và hình cục bộ có thể bị ẩn trong phân tích toàn cầu.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate clustered data
X, y = make_blobs(n_samples=1000, n_features=50, centers=5, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
ax1.set_title('PCA (Global Structure)')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
ax2.set_title('t-SNE (Local Structure)')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Trang trình bày 4: Xác định và ngẫu nhiên

PCA là một thuật toán xác định chính xác, có nghĩa là nó luôn tạo ra cùng một kết quả cho một dữ liệu tối đa. Thuộc tính này làm cho kết quả PCA có thể lặp lại và chạy ít nhất qua nhiều lần. Ngược lại, t-SNE là một thuật toán ngẫu nhiên liên quan đến tính ngẫu nhiên trong quá trình tối ưu hóa của nó. Kết quả là, t-SNE có thể tạo ra các kết quả hơi nước khác nhau mỗi lần nó chạy trên cùng một dữ liệu, ngay lập tức cùng một hạt giống ngẫu nhiên.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate sample data
np.random.seed(42)
X = np.random.randn(500, 50)

# Apply PCA multiple times
pca_results = []
for _ in range(3):
    pca = PCA(n_components=2)
    pca_results.append(pca.fit_transform(X))

# Apply t-SNE multiple times
tsne_results = []
for _ in range(3):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results.append(tsne.fit_transform(X))

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, (pca_result, tsne_result) in enumerate(zip(pca_results, tsne_results)):
    axes[0, i].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    axes[0, i].set_title(f'PCA Run {i+1}')

    axes[1, i].scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
    axes[1, i].set_title(f't-SNE Run {i+1}')

plt.tight_layout()
plt.show()
```

Slide 5: Giải mã khả năng

PCA đưa ra những kết quả đơn giản và dễ hiểu. Mỗi thành phần chính là tính chất tuyến đường hợp lý của các đầu ban đặc biệt, cho phép chúng hiểu những đặc điểm nào đóng góp nhiều nhất vào các đặc biệt trong dữ liệu. Khả năng giải quyết này giúp PCA trở nên hữu ích trong việc lựa chọn tính năng và hiểu cơ sở cấu trúc của dữ liệu. t-SNE, mặc dù có khả năng hiển thị tuyệt vời nhưng lại khó diễn giải hơn. Các thành phần được không có mối quan hệ rõ ràng với các tính năng ban đầu, khiến t-SNE chủ yếu hữu ích cho công việc khám phá và trực quan hóa dữ liệu hơn là diễn giải tính năng.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
plt.title('PCA of Iris Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Plot feature contributions
plt.subplot(122)
components = pca.components_.T
plt.bar(feature_names, components[:, 0], alpha=0.5, label='PC1')
plt.bar(feature_names, components[:, 1], alpha=0.5, label='PC2')
plt.title('Feature Contributions to Principal Components')
plt.xlabel('Features')
plt.ylabel('Contribution')
plt.legend()
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 6: Chi phí tính toán

PCA có hiệu suất tính toán và khả năng mở rộng tốt trên các dữ liệu lớn. Độ phức tạp của thời gian là O(min(n^2d, nd^2)), trong đó n là mẫu số lượng và d là số lượng đặc biệt. Hiệu quả này làm cho PCA phù hợp với các nhiệm vụ giảm dữ liệu nhiều chiều. Ngược lại, t-SNE có giá thành cao hơn về mặt tính toán, đặc biệt đối với các bộ dữ liệu lớn hơn. Độ phức tạp của thời gian của nó là O(n^2), có thể trở nên hạn chế đối với các dữ liệu rất lớn. Do đó, t-SNE thường được áp dụng cho các dữ liệu nhỏ hơn hoặc được sử dụng như bước trực tiếp hóa cuối cùng sau khi giảm kích thước ban đầu bằng các phương pháp khác.

```python
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def compare_computation_time(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)

    # Measure PCA computation time
    start_time = time.time()
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    pca_time = time.time() - start_time

    # Measure t-SNE computation time
    start_time = time.time()
    tsne = TSNE(n_components=2)
    tsne.fit_transform(X)
    tsne_time = time.time() - start_time

    return pca_time, tsne_time

# Compare computation times for different dataset sizes
sizes = [100, 500, 1000, 2000]
pca_times = []
tsne_times = []

for size in sizes:
    pca_time, tsne_time = compare_computation_time(size, 50)
    pca_times.append(pca_time)
    tsne_times.append(tsne_time)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(sizes, pca_times, marker='o', label='PCA')
plt.plot(sizes, tsne_times, marker='o', label='t-SNE')
plt.xlabel('Number of Samples')
plt.ylabel('Computation Time (seconds)')
plt.title('PCA vs t-SNE Computation Time')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
```

Trang trình bày 7: Khi nào nên sử dụng PCA

PCA đặc biệt hữu ích khi bạn cần một phương pháp rút gọn đơn giản, dễ hiểu cho dữ liệu nhiều chiều. Đây là giải pháp lý tưởng để chuẩn bị dữ liệu cho các mô hình máy học yêu cầu cho phép biến đổi tuyến tính hoặc khi bạn muốn giảm nhiễu bằng cách loại bỏ ít tính năng hơn. PCA cũng có giá trị trong việc phân tích dòng truy cập dữ liệu, giúp xác định các biến quan trọng nhất trong dữ liệu của bạn.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('PCA of Digits Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Add some digit images to the plot
for i in range(10):
    idx = np.where(y == i)[0][0]
    plt.annotate(str(i), X_pca[idx], xytext=(5, 5), textcoords='offset points')
    plt.imshow(digits.images[idx], cmap='binary', extent=(X_pca[idx, 0]-10, X_pca[idx, 0]+10,
                                                          X_pca[idx, 1]-10, X_pca[idx, 1]+10))

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Trang trình bày 8: Khi nào nên sử dụng t-SNE

t-SNE có hiệu quả đặc biệt khi bạn muốn trực tiếp hóa dữ liệu chiều cao ở dạng 2D hoặc 3D trong khi vẫn duy trì bộ cụm dữ liệu cục bộ. Đây là giải pháp lý tưởng để khám phá các tập dữ liệu có quan hệ phức tạp, phi tuyến tính mà PCA có thể bỏ đi. t-SNE cũng hữu ích cho việc phân cụm và khám phá nội dung cấu trúc tại dữ liệu, đặc biệt khi các mối quan hệ cục bộ quan trọng hơn cấu trúc toàn cầu.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('t-SNE of Digits Dataset')
plt.xlabel('First t-SNE Component')
plt.ylabel('Second t-SNE Component')

# Add some digit images to the plot
for i in range(10):
    idx = np.where(y == i)[0][0]
    plt.annotate(str(i), X_tsne[idx], xytext=(5, 5), textcoords='offset points')
    plt.imshow(digits.images[idx], cmap='binary', extent=(X_tsne[idx, 0]-10, X_tsne[idx, 0]+10,
                                                          X_tsne[idx, 1]-10, X_tsne[idx, 1]+10))

plt.tight_layout()
plt.show()
```

Slide 9: Ví dụ thực tế: Xử lý hình ảnh

Trong quá trình xử lý ảnh, PCA có thể được sử dụng cho các tác vụ như nén ảnh và trích xuất đặc điểm. Ví dụ: chúng tôi có thể sử dụng PCA để giảm kích thước của hình ảnh dữ liệu trong khi vẫn giữ mức quan trọng nhất của hình ảnh thông tin. Kỹ thuật này đặc biệt hữu ích trong các hệ thống nhận dạng khuôn mặt, trong đó PCA thường được gọi là phương pháp "khuôn mặt riêng".

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people

# Load face dataset
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data
y = faces.target

# Apply PCA
n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X)

# Reconstruct faces using different numbers of components
n_reconstructions = [10, 50, 100, 150]
fig, axs = plt.subplots(5, len(n_reconstructions), figsize=(15, 12))

for i, n in enumerate(n_reconstructions):
    reconst_img = pca.inverse_transform(pca.transform(X[0].reshape(1, -1))[:, :n])
    axs[0, i].imshow(reconst_img.reshape(faces.images[0].shape), cmap='gray')
    axs[0, i].set_title(f'{n} components')
    axs[0, i].axis('off')

# Show original image
axs[0, 0].imshow(X[0].reshape(faces.images[0].shape), cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')

plt.tight_layout()
plt.show()

print("Explained variance ratio sum:", sum(pca.explained_variance_ratio_))
```

Trang trình bày 10: Ví dụ thực tế: Phân tích bộ gen dữ liệu

Trong bộ gen, t-SNE thường được sử dụng để trực tiếp hóa dữ liệu biểu hiện chiều cao của gen. Nó có thể tiết lộ các cụm gen có kiểu biểu hiện tương tự hoặc các mẫu nhóm có cấu hình truyền tải tương thích. Ứng dụng này rất quan trọng trong việc tìm hiểu các hệ thống sinh học phức tạp và xác định dấu ấn sinh học tiềm ẩn đối với bệnh tật.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

# Simulate gene expression data
n_samples = 1000
n_features = 100
n_clusters = 5

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of Simulated Gene Expression Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

Slide 11: Kết hợp PCA và t-SNE

Trong thực tế, việc kết hợp PCA và t-SNE thường có lợi, đặc biệt khi xử lý dữ liệu có nhiều chiều. PCA can be used as step tiền xử lý để giảm chiều của dữ liệu trước khi áp dụng t-SNE. Cách tiếp cận này có thể tăng tốc độ đáng kể quá trình tính toán t-SNE trong khi vẫn đảm bảo an toàn cho các cấu trúc quan trọng trong dữ liệu.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Apply PCA as a preprocessing step
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Apply t-SNE on PCA-reduced data
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype(int), cmap='tab10', alpha=0.5)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of MNIST (PCA preprocessed)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

Trang trình bày 12: Điều chỉnh siêu tham số trong t-SNE

Trong khi PCA có ít siêu tham số, t-SNE có một số siêu tham số có thể gây ảnh hưởng đáng kể đến đầu ra của nó. Điều quan trọng nhất là sự phức tạp và số lần lặp lại. Sự phức tạp cân bằng các cạnh cục bộ và toàn cầu của dữ liệu, trong khi số lần lặp lại ảnh hưởng đến mức độ ưu tiên của thuật toán khi nhúng. Điều quan trọng là phải thử nghiệm các tham số này để tìm ra hình ảnh trực quan tốt nhất cho dữ liệu của bạn.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Define different perplexity values
perplexities = [5, 30, 50, 100]

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()

for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)

    axs[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5)
    axs[i].set_title(f'Perplexity: {perplexity}')
    axs[i].set_xlabel('t-SNE 1')
    axs[i].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

Trang trình bày 13: Chế độ và cân bằng nhanh

Mặc dù PCA và t-SNE là những công cụ mạnh mẽ nhưng chúng cũng có những chế độ hạn chế. PCA giả định các tính chất tuyến tính quan hệ và có thể loại bỏ các cấu trúc quan trọng của tuyến tính. t-SNE double khi có thể tạo ra những hình ảnh trực quan gây hiểu lầm, đặc biệt khi rối loạn không thể điều chỉnh tốt. Cả hai phương pháp đều có thể gặp khó khăn với dữ liệu có chiều rất cao. Điều quan trọng là phải hiểu những giới hạn chế độ này và sử dụng các kỹ thuật này như một phần rộng hơn của phương pháp phân tích phương pháp, thay vì chỉ dựa vào chúng.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate S-curve dataset
X, color = make_s_curve(n_samples=1000, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.scatter(X[:, 0], X[:, 2], c=color, cmap='viridis')
ax1.set_title('Original S-curve')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis')
ax2.set_title('PCA')

ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap='viridis')
ax3.set_title('t-SNE')

plt.tight_layout()
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về PCA và t-SNE, đây là một số tài nguyên có giá trị:

1. "Giảm kích thước: Đánh giá so sánh" của L.J.P. van der Maaten, E.O. Postma và H.J. van den Herik (ArXiv:0904.3383)
2. "Trực quan hóa dữ liệu bằng t-SNE" của Laurens van der Maaten và Geoffrey Hinton (Tạp chí Nghiên cứu Máy học, 2008)
3. "Hướng dẫn phân tích thành phần chính" của Jonathon Shlens (ArXiv:1404.1100)
4. “Cách sử dụng hiệu quả t-SNE” của Martin Wattenberg, Fernanda Viégas và Ian Johnson (Distill, 2016)

Bài viết này cung cấp những giải pháp sâu sắc về thuật toán, nền tảng toán học của họ và các phương pháp hay nhất để ứng dụng chúng trong các lĩnh vực khác nhau.
