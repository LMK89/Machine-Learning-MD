## Phép kiểm tra và loại đồng nhất được phép (UMAP)
##bị gián đoạn trình bày trang 15

Slide 1: Giới thiệu về UMAP

Phép đo và chà xát tối đa hệ thống nhất (UMAP) là một kỹ thuật giảm kích thước được sử dụng để hiển thị nhiều dữ liệu trong không gian có chiều thấp hơn. Nó đặc biệt hữu ích để khám phá và hiểu các bộ dữ liệu phức tạp trong máy học và dữ liệu phân tích.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random high-dimensional data
data = np.random.rand(1000, 50)

# Create UMAP object and fit the data
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(data)

# Plot the results
plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
plt.title("UMAP Projection of Random 50D Data")
plt.show()
```

Slide 2: Tổng quan về thuật toán UMAP

Hoạt động UMAP bằng cách xây dựng biểu tượng dữ liệu theo biểu tượng chiều cao của đồ thị và sau đó tìm cách tải xuống độ sâu chiều sâu để đảm bảo an toàn cấu trúc đồ thị. Nó cân bằng việc bảo tồn cấu trúc địa phương và toàn cầu, mang lại những hình ảnh trực quan có ý nghĩa.

```python
from sklearn.datasets import load_digits
from umap import UMAP

# Load the digits dataset
digits = load_digits()

# Create and fit UMAP model
umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = umap_model.fit_transform(digits.data)

print(f"Original shape: {digits.data.shape}")
print(f"Embedded shape: {embedding.shape}")
```

Slide 3: Thông số UMAP

Các tham số chính trong bao UMAP bao gồm n\_neighbors, min\_dist và n\_comComponents. Tham số này kiểm soát sự cân bằng giữa việc đảm bảo toàn bộ cấu trúc cục bộ và toàn cục, tính gọn nhẹ của việc nhúng và tính chiều của đầu ra.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.rand(1000, 20)

# Define different parameter sets
params = [
    {"n_neighbors": 5, "min_dist": 0.1},
    {"n_neighbors": 15, "min_dist": 0.5},
    {"n_neighbors": 50, "min_dist": 0.1}
]

# Plot UMAP embeddings with different parameters
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, param in enumerate(params):
    reducer = umap.UMAP(**param, n_components=2)
    embedding = reducer.fit_transform(data)
    axs[i].scatter(embedding[:, 0], embedding[:, 1], s=5)
    axs[i].set_title(f"n_neighbors={param['n_neighbors']}, min_dist={param['min_dist']}")

plt.tight_layout()
plt.show()
```

Slide 4: Chuẩn bị dữ liệu cho UMAP

Trước khi áp dụng UMAP, điều quan trọng phải xử lý trước dữ liệu. Điều này thường liên quan đến công việc mở rộng mô hình, xử lý việc thiếu giá trị và mã hóa các loại biến.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample data
data = pd.DataFrame({
    'numeric1': [1, 2, np.nan, 4],
    'numeric2': [5, 6, 7, 8],
    'categorical': ['A', 'B', 'A', 'C']
})

# Define preprocessing steps
numeric_features = ['numeric1', 'numeric2']
categorical_features = ['categorical']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', pd.get_dummies)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the data
processed_data = preprocessor.fit_transform(data)
print(processed_data)
```

Slide 5: UMAP để giảm kích thước

UMAP thường được sử dụng để giảm dữ liệu có chiều cao thành biểu tượng có chiều thấp hơn, thường là 2D hoặc 3D cho mục tiêu trực tiếp hóa.

```python
import umap
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()

# Create and fit UMAP model
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(digits.data)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title('UMAP projection of the Digits dataset')
plt.show()
```

Trang trình bày 6: UMAP so với t-SNE

UMAP thường cung cấp chất lượng hiển thị tương tự như t-SNE nhưng với thời gian tính toán nhanh hơn và bảo đảm cấu trúc tổng thể tốt hơn.

```python
from sklearn.datasets import load_digits
import umap
import time
import matplotlib.pyplot as plt

# Load data
digits = load_digits()

# UMAP
start_time = time.time()
umap_embedding = umap.UMAP().fit_transform(digits.data)
umap_time = time.time() - start_time

# t-SNE
start_time = time.time()
tsne_embedding = TSNE().fit_transform(digits.data)
tsne_time = time.time() - start_time

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
ax1.set_title(f'UMAP (Time: {umap_time:.2f}s)')

ax2.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
ax2.set_title(f't-SNE (Time: {tsne_time:.2f}s)')

plt.show()
```

Slide 7: UMAP để phân cụm

UMAP có thể được sử dụng như một bước xử lý tiền cho các cụm thuật toán, có khả năng cải thiện hiệu suất của chúng trên nhiều dữ liệu.

```python
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

# Load data
iris = load_iris()

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(iris.data)

# Perform clustering on the embedding
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embedding)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title('UMAP + K-means clustering of Iris dataset')
plt.show()
```

Trang trình bày 8: UMAP để phát hiện những điều không mong đợi

UMAP có thể giúp trực tiếp hóa các điểm bất thường trong dữ liệu có chiều cao bằng cách tham chiếu chúng vào không gian có chiều thấp hơn, nơi chúng có thể được xác định dễ dàng hơn.

```python
from sklearn.datasets import make_blobs
import umap
import matplotlib.pyplot as plt

# Generate normal data
X, _ = make_blobs(n_samples=1000, centers=3, n_features=10, random_state=42)

# Generate anomalies
anomalies = np.random.uniform(low=-10, high=10, size=(50, 10))

# Combine normal data and anomalies
X_with_anomalies = np.vstack([X, anomalies])

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(X_with_anomalies)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:-50, 0], embedding[:-50, 1], c='blue', s=5, label='Normal')
plt.scatter(embedding[-50:, 0], embedding[-50:, 1], c='red', s=20, label='Anomaly')
plt.legend()
plt.title('UMAP projection for Anomaly Detection')
plt.show()
```

Trang trình bày 9: UMAP được giám sát

UMAP có thể kết hợp thông tin nhãn để tạo ra nhiều thông tin nhúng hơn cho các nhiệm vụ học tập có giám sát.

```python
import umap
import matplotlib.pyplot as plt

# Load data
iris = load_iris()

# Apply supervised UMAP
supervised_reducer = umap.UMAP(n_components=2, random_state=42, target_metric='l2')
supervised_embedding = supervised_reducer.fit_transform(iris.data, y=iris.target)

# Apply unsupervised UMAP
unsupervised_reducer = umap.UMAP(n_components=2, random_state=42)
unsupervised_embedding = unsupervised_reducer.fit_transform(iris.data)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.scatter(supervised_embedding[:, 0], supervised_embedding[:, 1], c=iris.target, cmap='Spectral', s=5)
ax1.set_title('Supervised UMAP')

ax2.scatter(unsupervised_embedding[:, 0], unsupervised_embedding[:, 1], c=iris.target, cmap='Spectral', s=5)
ax2.set_title('Unsupervised UMAP')

plt.show()
```

Slide 10: UMAP để lựa chọn tính năng

UMAP có thể được sử dụng để xác định các tính năng quan trọng bằng cách kiểm tra lời khuyên đóng góp của từng tính năng vào công việc có chiều sâu được nhúng.

```python
import umap
import pandas as pd
import matplotlib.pyplot as plt

# Load data
cancer = load_breast_cancer()

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(cancer.data)

# Get feature importances
feature_importances = pd.Series(reducer.feature_importances_, index=cancer.feature_names)

# Plot top 10 important features
plt.figure(figsize=(12, 6))
feature_importances.nlargest(10).plot(kind='bar')
plt.title('Top 10 Important Features in UMAP Embedding')
plt.tight_layout()
plt.show()
```

Slide 11: UMAP cho dữ liệu văn bản

UMAP có thể được áp dụng cho dữ liệu văn bản sau khi chuyển đổi văn bản sang một số biểu tượng, nhưng có chế độ hạn chế như TF-IDF.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt

# Load data
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(newsgroups.data)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(tfidf_matrix)

# Plot results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=newsgroups.target, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title('UMAP projection of 20 Newsgroups dataset')
plt.show()
```

Trang hiển thị 12: UMAP cho dữ liệu hình ảnh

UMAP có thể được áp dụng cho dữ liệu hình ảnh để trực quan hóa sự tương đồng và khác biệt giữa các hình ảnh trong dữ liệu.

```python
import umap
import matplotlib.pyplot as plt

# Load data
digits = load_digits()

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(digits.data)

# Plot results
plt.figure(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.colorbar()
plt.title('UMAP projection of the Digits dataset')

# Plot some example digits
for i in range(10):
    plt.annotate(str(i), xy=(embedding[digits.target == i, 0].mean(),
                              embedding[digits.target == i, 1].mean()),
                 xytext=(0, 0), textcoords="offset points",
                 ha='center', va='center',
                 bbox=dict(boxstyle="round", fc="w"),
                 arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.show()
```

Trang trình tự 13: Ví dụ thực tế: Trình tự gen

UMAP có thể được sử dụng trong bộ gen để trực quan hóa và phân tích dữ liệu truyền tải nhiều chiều, giúp các nhà nghiên cứu xác định các mẫu kiểu và mối liên hệ giữa các truyền tải cấu hình khác nhau.

```python
import umap
import matplotlib.pyplot as plt

# Simulate genetic data (SNPs)
n_samples = 1000
n_snps = 10000
genetic_data = np.random.randint(0, 3, size=(n_samples, n_snps))

# Simulate population labels (e.g., different ethnic groups)
populations = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(genetic_data)

# Plot results
plt.figure(figsize=(12, 10))
for pop in np.unique(populations):
    mask = populations == pop
    plt.scatter(embedding[mask, 0], embedding[mask, 1], label=pop, s=5)

plt.legend()
plt.title('UMAP projection of simulated genetic data')
plt.show()
```

Trang trình bày 14: Ví dụ thực tế: Phân khúc khách hàng

UMAP có thể được sử dụng trong thị trường tiếp theo để phân khúc khách hàng dựa trên hành vi của họ, giúp doanh nghiệp điều chỉnh chiến lược của mình cho phù hợp với các nhóm khách hàng khác nhau.

```python
import pandas as pd
import umap
import matplotlib.pyplot as plt

# Simulate customer data
n_customers = 1000
customer_data = pd.DataFrame({
    'age': np.random.normal(40, 15, n_customers),
    'income': np.random.lognormal(10, 1, n_customers),
    'spending': np.random.lognormal(5, 1, n_customers),
    'frequency': np.random.poisson(10, n_customers),
    'loyalty_years': np.random.gamma(2, 2, n_customers)
})

# Normalize the data
normalized_data = (customer_data - customer_data.mean()) / customer_data.std()

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(normalized_data)

# Apply simple clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(embedding)

# Plot results
plt.figure(figsize=(12, 10))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', s=5)
plt.colorbar(scatter)
plt.title('UMAP projection of customer segments')
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến công việc tìm hiểu sâu hơn về UMAP, đây là một số tài nguyên có giá trị:

1. Bài báo gốc UMAP: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Phép xử lý và cho phép tối đa hóa dạng đồng nhất để giảm kích thước. ArXiv:1802.03426. URL: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
2. Tài liệu UMAP: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
3. So sánh các kỹ thuật giảm kích thước: Espadoto, M., Martins, R. M., Kerren, A., Hirata, N. S. T., & Telea, A. C. (2019). Hướng tới một cuộc khảo sát định lượng về các kỹ thuật giảm kích thước. Giao dịch của IEEE về Trực quan hóa và Đồ họa Máy tính. URL: [https://arxiv.org/abs/1904.08566](https://arxiv.org/abs/1904.08566)

Tài nguyên này cung cấp sự hiểu biết toàn diện về lý thuyết nền tảng, ứng dụng thực tế và so sánh UMAP với các kỹ thuật kích thước khác.
