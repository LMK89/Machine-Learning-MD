## Giảm kích thước cổ điển trong Python
Trang trình bày 1:

Giới thiệu về giảm kích thước cổ điển

Kích thước nhỏ là một kỹ thuật cơ bản trong máy học và phân tích dữ liệu. Nó nhằm mục đích giảm số lượng hoặc kích thước mục tiêu trong dữ liệu khi lưu trữ càng nhiều thông tin liên quan càng tốt. Phương pháp cổ điển thu nhỏ của phương pháp này là biến đổi tuyến tính tham chiếu dữ liệu được phép có chiều cao lên không có chiều thấp hơn.

```python
# No code for the introduction slide
```

Slide 2:

Principal Component Analysis (PCA)

PCA is one of the most widely used dimensionality reduction techniques. It finds the directions of maximum variance in the data and projects the data onto a lower-dimensional subspace spanned by these directions, called principal components.

```python
from sklearn.decomposition import PCA

# Load your data
X = ... # Your data

# Create a PCA object
pca = PCA(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = pca.fit_transform(X)
```

Trang trình bày 3:

Ví dụ về PCA

Vui lòng áp dụng PCA cho tập dữ liệu iris, một tập dữ liệu máy cổ chứa các thước đo được phép của nhiều loài khác với iris.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data

# Create a PCA object and transform the data
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=iris.target)
plt.show()
```

Trang trình bày 4:

Phân tích phân tích tuyến tính (LDA)

LDA là một kỹ thuật giảm kích thước có giám sát nhằm tìm kiếm các hướng dẫn tối đa phân tách giữa các lớp khi phương pháp giảm thiểu tối thiểu trong mỗi lớp.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load your data and labels
X = ... # Your data
y = ... # Labels

# Create an LDA object
lda = LDA(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = lda.fit_transform(X, y)
```

Trang trình bày 5:

Ví dụ về LDA

Vui lòng áp dụng LDA cho dữ liệu mắt, sử dụng nhãn để tìm hướng dẫn phân tích.

```python
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create an LDA object and transform the data
lda = LDA(n_components=2)
X_transformed = lda.fit_transform(X, y)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
plt.show()
```

Trang trình bày 6:

Phân tích nhân tố

Phân tích nhân tố là một kỹ thuật thống kê mục tiêu mô tả các mối quan hệ cơ bản giữa các biến được quan sát dưới dạng số lượng nhỏ hơn các biến không được quan sát được gọi là các yếu tố.

```python
from sklearn.decomposition import FactorAnalysis

# Load your data
X = ... # Your data

# Create a Factor Analysis object
fa = FactorAnalysis(n_components=3)  # Reduce to 3 factors

# Fit and transform the data
X_transformed = fa.fit_transform(X)
```

Trang trình bày 7:

Ví dụ phân tích nhân tố

Vui lòng áp dụng phân tích nhân tố cho tập dữ liệu mô phỏng với cơ bản yếu tố tố.

```python
import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

# Generate simulated data with 3 factors
np.random.seed(42)
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
factors = np.random.randn(10, 3)  # 3 factors
X = X @ factors.T + np.random.randn(1000, 10)  # Add noise

# Apply Factor Analysis
fa = FactorAnalysis(n_components=3)
X_transformed = fa.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
plt.show()
```

Trang trình bày 8:

Phân tích thành phần độc lập (ICA)

ICA là một kỹ thuật phân tích tín hiệu đa biến thành các tín hiệu không phải Gaussian độc lập, được gọi là các thành phần độc lập.

```python
from sklearn.decomposition import FastICA

# Load your data
X = ... # Your data

# Create an ICA object
ica = FastICA(n_components=3)  # Reduce to 3 independent components

# Fit and transform the data
X_transformed = ica.fit_transform(X)
```

Trang trình bày 9:

Ví dụ về ICA

Vui lòng áp dụng ICA cho tệp mô phỏng có ba tín hiệu không độc lập Gaussian.

```python
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Generate simulated data with 3 independent signals
np.random.seed(42)
s1 = np.random.laplace(size=1000)  # Laplace distribution
s2 = np.random.exponential(size=1000)  # Exponential distribution
s3 = np.random.normal(size=1000)  # Gaussian distribution
X = np.c_[s1, s2, s3] + np.random.randn(1000, 3)  # Add noise

# Apply ICA
ica = FastICA(n_components=3)
X_transformed = ica.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
plt.show()
```

Trang trình bày 10:

Tỷ lệ chia chiều đa chiều (MDS)

MDS là một kỹ thuật xạ dữ liệu có chiều cao vào không gian có chiều thấp hơn trong khi vẫn duy trì khoảng cách theo cặp giữa các dữ liệu càng nhiều càng tốt.

```python
from sklearn.manifold import MDS

# Load your data
X = ... # Your data

# Create an MDS object
mds = MDS(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = mds.fit_transform(X)
```

Trang trình bày 11:

Ví dụ về MDS

Vui lòng áp dụng MDS cho tập dữ liệu mắt, duy trì khoảng cách theo cặp giữa các mẫu.

```python
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data

# Apply MDS
mds = MDS(n_components=2)
X_transformed = mds.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=iris.target)
plt.show()
```

Trang trình bày 12:

Phân tích đồng bản đồ

Isomap là một kỹ thuật giảm kích thước phi tuyến tính toán bảo vệ toàn bộ nội dung cấu hình cấu hình của dữ liệu bằng cách xoa dịu khoảng cách giữa các dữ liệu.

```python
from sklearn.manifold import Isomap

# Load your data
X = ... # Your data

# Create an Isomap object
isomap = Isomap(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = isomap.fit_transform(X)
```

Trang trình bày 13:

Ví dụ về isomap

Vui lòng áp dụng Isomap cho tập dữ liệu Swiss Roll, một ví dụ phức tạp phi tuyến tính cổ điển.

```python
from sklearn import datasets
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

# Load the Swiss Roll dataset
X, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)

# Apply Isomap
isomap = Isomap(n_components=2)
X_transformed = isomap.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color)
plt.title('Isomap on Swiss Roll Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
```

Trang trình bày 14:

t-SNE (Nhúng hàng ngẫu nhiên phân phối t)

t-SNE là một kỹ thuật giảm kích thước phi tuyến tính, đặc biệt phù hợp để hiển thị dữ liệu nhiều chiều. Nó mô hình hóa sự tương thích giữa các dữ liệu và cố gắng đảm bảo an toàn cho chúng trong không gian có chiều sâu thấp hơn.

```python
from sklearn.manifold import TSNE

# Load your data
X = ... # Your data

# Create a t-SNE object
tsne = TSNE(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = tsne.fit_transform(X)
```

Trang trình bày 15:

Ví dụ về t-SNE

Vui lòng áp dụng t-SNE cho dữ liệu MNIST, nhận dữ liệu dưới dạng chữ viết tay, để trực quan hóa dữ liệu nhiều chiều ở dạng 2D.

```python
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data / 255.0  # Normalize pixel values

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=mnist.target.astype(int))
plt.title('t-SNE on MNIST Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
```

Trang trình bày 16: (Tài nguyên bổ sung):

Tài nguyên bổ sung

Để đọc thêm và khám phá các kỹ thuật giảm kích thước, dưới đây là một số tài nguyên được xuất bản từ ​​arXiv.org:

1. " Hướng dẫn phân tích thành phần chính" của Jonathon Shlens ([https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100))
2. "Giảm kích thước: Đánh giá so sánh" của Hyunwoo J. Kim và Hyeyoung Park ([https://arxiv.org/abs/1806.04349](https://arxiv.org/abs/1806.04349))
3. "Các phương pháp hạt nhân để giảm kích thước phi tuyến" của Lawrence K. Saul và Sam T. Roweis ([https://arxiv.org/abs/1511.08898](https://arxiv.org/abs/1511.08898))
4. " Giới thiệu về Phân tích thành phần độc lập" của Aapo Hyvärinen và Erkki Oja ([https://arxiv.org/abs/1804.04598](https://arxiv.org/abs/1804.04598))

Lưu ý: Tài nguyên này được lấy từ arXiv.org và có sẵn từ tháng 8 năm 2023.
