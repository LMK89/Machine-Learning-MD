## Ưu điểm của Máy Vector Hỗ trợ trong Phân loại Mạnh mẽ
Trang trình bày 1: Phân loại ký quỹ tối đa

Máy vectơ hỗ trợ (SVM) thiết lập ranh giới quyết định tối ưu bằng cách tối đa hóa lề giữa các lớp, tạo ra một dấu phân cách mạnh mẽ giúp tăng cường khả năng khái quát hóa. Lề biểu thị khoảng cách giữa siêu phẳng và các điểm dữ liệu gần nhất từ ​​mỗi lớp, được gọi là vectơ hỗ trợ.

```python
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Create and train SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Plot decision boundary
w = clf.coef_[0]
b = clf.intercept_[0]
x_points = np.linspace(-3, 3)
y_points = -(w[0] * x_points + b) / w[1]

plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
plt.plot(x_points, y_points, 'k-')
plt.legend()
plt.show()
```

Slide 2: Cơ sở toán học của SVM

Bài toán tối ưu hóa SVM nhằm mục đích tìm ra siêu phẳng làm cực đại biên hình học đồng thời giảm thiểu các lỗi phân loại. Điều này liên quan đến việc giải một bài toán quy hoạch bậc hai với các ràng buộc tuyến tính.

```python
# Mathematical formulation in LaTeX notation:
"""
$$
\begin{aligned}
\text{minimize} \quad & \frac{1}{2}\|w\|^2 \\
\text{subject to} \quad & y_i(w^Tx_i + b) \geq 1, \quad i=1,\ldots,n
\end{aligned}
$$

For soft margin SVM:
$$
\begin{aligned}
\text{minimize} \quad & \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i \\
\text{subject to} \quad & y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad i=1,\ldots,n
\end{aligned}
$$
"""
```

Trang trình bày 3: Triển khai SVM từ đầu

Việc triển khai thể hiện các khái niệm cốt lõi của SVM bằng cách sử dụng tối ưu hóa giảm độ dốc để tìm các tham số siêu phẳng tối ưu w và b nhằm tối đa hóa lề giữa các lớp.

```python
class SimpleSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w -
                                       np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
```

Slide 4: Kernel Trick Implementation

The kernel trick allows SVM to handle non-linearly separable data by mapping features into a higher-dimensional space where linear separation becomes possible, without explicitly computing the transformation.

```python
def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2, axis=1)**2 / (2 * (sigma ** 2)))

class KernelSVM:
    def __init__(self, kernel=gaussian_kernel, C=1.0):
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Compute the kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            K[i,:] = self.kernel(X[i], X)

        # Solve the dual optimization problem
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        A = y.reshape(1, -1)
        b = np.zeros(1)

        from cvxopt import matrix, solvers
        solution = solvers.qp(matrix(P), matrix(q), matrix(-np.eye(n_samples)),
                            matrix(np.zeros(n_samples)), matrix(A), matrix(b))

        # Extract support vectors
        self.alpha = np.array(solution['x']).flatten()
        sv = self.alpha > 1e-5
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]
        self.alpha = self.alpha[sv]
```

Slide 5: Ứng dụng thực tế - Phân loại văn bản

SVM vượt trội trong các nhiệm vụ phân loại văn bản nhờ khả năng xử lý dữ liệu thưa thớt nhiều chiều một cách hiệu quả. Việc triển khai này thể hiện việc phân loại tài liệu bằng cách sử dụng các tính năng TF-IDF.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pandas as pd

# Sample text data
documents = [
    "machine learning algorithms optimize performance",
    "deep neural networks process complex patterns",
    "stock market analysis predicts trends",
    "financial forecasting uses historical data"
]
labels = [0, 0, 1, 1]  # 0: Tech, 1: Finance

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC())
])

# Train classifier
text_clf.fit(documents, labels)

# Predict new documents
new_docs = ["artificial intelligence improves automation",
            "market volatility affects investments"]
predictions = text_clf.predict(new_docs)
print(f"Predictions: {predictions}")  # Output: [0, 1]
```

Trang trình bày 6: Điều chỉnh siêu tham số SVM

Tối ưu hóa hiệu suất SVM yêu cầu điều chỉnh cẩn thận các siêu tham số như C (chính quy hóa) và tham số kernel. Việc triển khai này thể hiện sự tối ưu hóa siêu tham số có hệ thống bằng cách sử dụng tìm kiếm dạng lưới với xác thực chéo.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline with preprocessing and SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC())
])

# Define parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.1, 1],
    'svm__degree': [2, 3, 4]  # Only for poly kernel
}

# Perform grid search
grid_search = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit and get best parameters
grid_search.fit(X, y)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
```

Slide 7: Phân loại SVM nhiều lớp

SVM mở rộng cho các bài toán nhiều lớp bằng cách sử dụng chiến lược một đấu một hoặc một đấu với phần còn lại, cho phép phân loại theo nhiều danh mục trong khi vẫn duy trì các đặc tính lề tối đa của chúng.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

class MultiClassSVM:
    def __init__(self, kernel='rbf', C=1.0):
        self.encoder = LabelEncoder()
        self.classifier = OneVsRestClassifier(SVC(kernel=kernel, C=C))

    def fit(self, X, y):
        # Encode labels
        y_encoded = self.encoder.fit_transform(y)
        # Train classifier
        self.classifier.fit(X, y_encoded)

    def predict(self, X):
        # Predict and decode labels
        y_pred = self.classifier.predict(X)
        return self.encoder.inverse_transform(y_pred)

# Example usage
X = np.random.randn(300, 2)
y = np.array(['A', 'B', 'C'] * 100)

clf = MultiClassSVM(kernel='rbf', C=1.0)
clf.fit(X, y)
predictions = clf.predict(X[:5])
print(f"Sample predictions: {predictions}")
```

Trang trình bày 8: Triển khai hạt nhân tùy chỉnh

Các hàm nhân tùy chỉnh cho phép SVM nắm bắt các thước đo tương tự theo miền cụ thể giữa các điểm dữ liệu, nâng cao tính linh hoạt của chúng cho các ứng dụng chuyên biệt.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomKernelSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel_func, C=1.0):
        self.kernel_func = kernel_func
        self.C = C

    def spectrum_kernel(self, s1, s2, k=3):
        """Custom string kernel for sequence data"""
        def get_kmers(s):
            return set(s[i:i+k] for i in range(len(s)-k+1))

        s1_kmers = get_kmers(s1)
        s2_kmers = get_kmers(s2)
        return len(s1_kmers.intersection(s2_kmers))

    def matrix_kernel(self, X1, X2):
        """Compute kernel matrix"""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i,j] = self.kernel_func(X1[i], X2[j])
        return K

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.K = self.matrix_kernel(X, X)
        # Implement QP solver here for weight calculation
        return self

    def predict(self, X):
        K_pred = self.matrix_kernel(X, self.X_train)
        # Implement prediction using kernel matrix
        return np.sign(K_pred.dot(self.alpha * self.y_train) + self.b)

# Example usage with custom string kernel
def string_kernel(s1, s2, k=3):
    return CustomKernelSVM.spectrum_kernel(None, s1, s2, k)

svm = CustomKernelSVM(kernel_func=string_kernel)
```

Trang trình bày 9: SVM để phát hiện bất thường

SVM có thể được điều chỉnh để phát hiện sự bất thường bằng cách tìm hiểu ranh giới bao quanh các điểm dữ liệu bình thường, giúp chúng có hiệu quả trong việc xác định các điểm bất thường và các mẫu bất thường.

```python
from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt

class AnomalyDetectorSVM:
    def __init__(self, nu=0.1, kernel='rbf'):
        self.detector = OneClassSVM(nu=nu, kernel=kernel)

    def fit_detect(self, X, plot=True):
        # Fit the model
        self.detector.fit(X)

        # Get predictions
        y_pred = self.detector.predict(X)

        if plot:
            # Create mesh grid
            xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5,
                                           X[:, 0].max()+0.5, 100),
                                np.linspace(X[:, 1].min()-0.5,
                                           X[:, 1].max()+0.5, 100))

            # Get predictions on mesh grid
            Z = self.detector.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot results
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
            plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Paired)
            plt.title('SVM Anomaly Detection')
            plt.show()

        return y_pred

# Generate sample data with anomalies
X_normal = np.random.randn(100, 2)
X_anomalies = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X_normal, X_anomalies])

# Detect anomalies
detector = AnomalyDetectorSVM(nu=0.1)
predictions = detector.fit_detect(X)
print(f"Number of anomalies detected: {sum(predictions == -1)}")
```

Slide 10: Học trực tuyến với SVM

Học trực tuyến cho phép SVM thích ứng với việc truyền dữ liệu bằng cách cập nhật mô hình dần dần. Việc triển khai này trình bày cách xử lý các tập dữ liệu quy mô lớn không vừa với bộ nhớ.

```python
class OnlineSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.0001):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.w = None
        self.b = 0

    def partial_fit(self, x, y):
        if self.w is None:
            self.w = np.zeros(x.shape[0])

        # Compute prediction
        prediction = np.dot(self.w, x) + self.b

        # Update if prediction is wrong
        if y * prediction < 1:
            self.w = (1 - self.lr * self.lambda_param) * self.w + \
                    self.lr * y * x
            self.b += self.lr * y
        else:
            self.w = (1 - self.lr * self.lambda_param) * self.w

    def predict(self, x):
        return np.sign(np.dot(self.w, x) + self.b)

# Example usage with streaming data
online_svm = OnlineSVM()
for _ in range(1000):
    # Simulate streaming data
    x = np.random.randn(10)
    y = np.sign(x[0] + x[1])

    # Update model
    online_svm.partial_fit(x, y)

    # Optional: evaluate performance periodically
    if _ % 100 == 0:
        correct = 0
        total = 100
        for i in range(total):
            x_test = np.random.randn(10)
            y_test = np.sign(x_test[0] + x_test[1])
            correct += (online_svm.predict(x_test) == y_test)
        print(f"Accuracy at iteration {_}: {correct/total:.2f}")
```

Trang trình bày 11: SVM cho hồi quy (SVR)

Hỗ trợ hồi quy vectơ mở rộng các nguyên tắc SVM cho các biến đầu ra liên tục bằng cách giới thiệu hàm mất mát không nhạy cảm ε tạo ra một ống bao quanh đường hồi quy.

```python
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

class SVRegressor:
    def __init__(self, kernel='rbf', epsilon=0.1, C=1.0):
        self.model = SVR(kernel=kernel, epsilon=epsilon, C=C)

    def fit_and_visualize(self, X, y):
        # Fit the model
        self.model.fit(X.reshape(-1, 1), y)

        # Create prediction line
        X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = self.model.predict(X_test)

        # Plot results
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(X_test, y_pred, color='red', label='SVR prediction')
        plt.plot(X_test, y_pred + self.model.epsilon, 'k--',
                label='ε-tube boundary')
        plt.plot(X_test, y_pred - self.model.epsilon, 'k--')
        plt.legend()
        plt.show()

        # Return support vectors
        return self.model.support_vectors_

# Generate sample regression data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100))
y = np.sin(X) + np.random.normal(0, 0.1, 100)

# Create and train SVR model
svr = SVRegressor(epsilon=0.1, C=1.0)
support_vectors = svr.fit_and_visualize(X, y)
print(f"Number of support vectors: {len(support_vectors)}")
```

Slide 12: Lựa chọn tính năng với SVM

SVM có thể được sử dụng để lựa chọn tính năng bằng cách phân tích trọng số được gán cho các tính năng khác nhau, giúp xác định các biến phù hợp nhất để phân loại.

```python
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

class SVMFeatureSelector:
    def __init__(self, C=1.0, threshold='mean'):
        self.svm = LinearSVC(C=C, penalty='l1', dual=False)
        self.selector = SelectFromModel(self.svm, prefit=False,
                                      threshold=threshold)
        self.scaler = StandardScaler()

    def fit_transform(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit selector
        self.selector.fit(X_scaled, y)

        # Get selected features
        selected_features = self.selector.get_support()
        feature_importance = np.abs(self.selector.estimator_.coef_).reshape(-1)

        # Sort features by importance
        feature_ranks = np.argsort(feature_importance)[::-1]

        # Transform data
        X_selected = self.selector.transform(X_scaled)

        return X_selected, selected_features, feature_ranks

# Example usage
X = np.random.randn(200, 20)  # 20 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Only first 2 features are relevant

selector = SVMFeatureSelector(C=0.1)
X_selected, selected_features, feature_ranks = selector.fit_transform(X, y)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
print(f"Top 5 feature indices: {feature_ranks[:5]}")
```

Trang trình bày 13: Tài nguyên bổ sung

* Bài viết ArXiv: "Đào tạo SVM quy mô lớn với độ dốc giảm dần ngẫu nhiên" - [https://arxiv.org/abs/1202.6547](https://arxiv.org/abs/1202.6547)
* Bài viết ArXiv: "Học nhiều hạt nhân để phân loại hình ảnh dựa trên SVM" - [https://arxiv.org/abs/1902.00415](https://arxiv.org/abs/1902.00415)
* Bài viết ArXiv: "Máy vectơ hỗ trợ trực tuyến cho dữ liệu quy mô lớn" - [https://arxiv.org/abs/1803.02346](https://arxiv.org/abs/1803.02346)
* Cụm từ tìm kiếm gợi ý cho Google Scholar:
    * "Hỗ trợ kỹ thuật tối ưu hóa Máy Vector"
    * "Phương pháp lựa chọn hạt nhân SVM"
    * "Triển khai SVM trực tuyến"
    * "Lựa chọn tính năng với SVM"
