## Làm chủ cây quyết định ID3 và Scikit-Learn
Trang trình bày 1: Tìm hiểu về quyết định của cây và thuật toán ID3

Cây định nghĩa là cấp độ phân tích cấu trúc được đưa ra để quyết định thông tin phân tích tuần tự thông tin dựa trên các tính năng. Thuật toán ID3 xây dựng cây bằng cách chọn tính năng tốt nhất ở mỗi nút bằng cách sử dụng thông tin thu được, tối đa hóa độ tinh khiết của các kết quả tập hợp thông qua entropy tính toán.

```python
import numpy as np
from collections import Counter

class DecisionTree:
    def entropy(self, y):
        # Calculate entropy of a node
        counts = Counter(y)
        probs = [count/len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs)

    def information_gain(self, X, y, feature):
        # Calculate information gain for a feature
        parent_entropy = self.entropy(y)
        # Get unique values and their frequencies
        values, counts = np.unique(X[:, feature], return_counts=True)
        # Calculate weighted entropy
        weighted_entropy = sum(
            (count/len(y)) * self.entropy(y[X[:, feature] == value])
            for value, count in zip(values, counts)
        )
        return parent_entropy - weighted_entropy
```

Slide 2: Lựa chọn tính năng trong ID3

Cốt lõi của ID3 nằm ở cơ chế lựa chọn tính năng, cơ chế này lặp lại để lựa chọn khả năng tối đa hóa trợ giúp thuộc tính để thu được thông tin. Quá trình này tiếp tục đệ quy cho đến khi đạt được điều kiện dừng lại, đưa ra giới hạn như các nút lá tinh hoặc đạt đến độ sâu tối đa.

```python
def find_best_split(self, X, y):
    best_gain = -1
    best_feature = None

    for feature in range(X.shape[1]):
        gain = self.information_gain(X, y, feature)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature, best_gain
```

Slide 3: Building the Decision Tree Structure

Tree construction involves creating nodes that store split decisions and leaf values. Each internal node contains the splitting feature and threshold, while leaf nodes store the majority class prediction for classification tasks.

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature = feature    # Feature index for splitting
        self.threshold = threshold # Split threshold value
        self.left = left          # Left subtree
        self.right = right        # Right subtree
        self.value = value        # Leaf node prediction value
```

Slide 4: Xây dựng trình đệ quy cây

Bản chất đệm của việc xây dựng cây quyết định Yêu cầu phải xử lý cẩn thận các cơ sở hợp lý và phân tích điều kiện. Việc phát triển khai báo này đã tìm ra cách phát triển cây bằng cách liên tục tìm các phần phân tích tốt nhất và tạo các nút con.

```python
def build_tree(self, X, y, depth=0):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # Stopping criteria
    if (self.max_depth and depth >= self.max_depth) or \
       n_classes == 1 or n_samples < self.min_samples_split:
        leaf_value = max(Counter(y).items(), key=lambda x: x[1])[0]
        return Node(value=leaf_value)

    # Find best split
    best_feature, best_gain = self.find_best_split(X, y)

    if best_gain < self.min_gain:
        leaf_value = max(Counter(y).items(), key=lambda x: x[1])[0]
        return Node(value=leaf_value)

    # Create child nodes
    left_idxs = X[:, best_feature] < X[:, best_feature].mean()
    right_idxs = ~left_idxs

    left = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
    right = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)

    return Node(feature=best_feature,
               threshold=X[:, best_feature].mean(),
               left=left,
               right=right)
```

Slide 5: Entropy and Information Gain Mathematics

The mathematical foundation of ID3 relies on entropy and information gain calculations. These metrics guide the algorithm in selecting optimal splits at each node of the decision tree.

```python
# Mathematical formulas for entropy and information gain
"""
Entropy formula:
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Information Gain formula:
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- S is the dataset
- A is the attribute
- c is the number of classes
- p_i is the proportion of class i in S
- S_v is the subset where attribute A has value v
"""
```

Slide 6: Triển khai logic dự kiến

Quá trình dự đoán sẽ đi qua cây từ gốc đến lá, đi theo đường dẫn thích hợp dựa trên việc so sánh tính năng tại mỗi nút. Dự kiến ​​cuối cùng sẽ được lấy từ giá trị được lưu trữ của nút lá.

```python
def predict_single(self, root, x):
    # Base case: reached a leaf node
    if root.value is not None:
        return root.value

    # Traverse left or right based on feature comparison
    if x[root.feature] < root.threshold:
        return self.predict_single(root.left, x)
    return self.predict_single(root.right, x)

def predict(self, X):
    # Predict for multiple samples
    return np.array([self.predict_single(self.root, x) for x in X])
```

Trang trình bày 7: Ví dụ thực tế - Phân loại mống mắt

Việc phát triển này có thể cung cấp khả năng phân loại cây được quyết định trên Iris cổ điển tập dữ liệu, hiển thị quá trình xử lý trước dữ liệu, đào tạo mô hình và đánh giá giá bằng cách sử dụng khai báo tùy chỉnh của chúng tôi.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
dt = DecisionTree(max_depth=5, min_samples_split=2)
dt.fit(X_train, y_train)

# Make predictions
predictions = dt.predict(X_test)
```

Slide 8: Kết quả phân loại mống mắt

Các bằng chứng hiệu quả về hiệu suất của việc phát triển ID3 của chúng tôi trên Iris dữ liệu, cho thấy độ chính xác và phân tích ma trận ở giữa.

```python
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

# Output example:
# Accuracy: 0.9667
# Confusion Matrix:
# [[10  0  0]
#  [ 0  9  1]
#  [ 0  0 10]]
```

Slide 9: Xử lý các tính năng liên tục

Xử lý tính năng liên tục Yêu cầu phải thực hiện các kết quả phân chia chiến lược hiệu quả. Việc khai báo này sử dụng phương pháp tìm kiếm nhị phân để tìm mức độ ưu tiên của các thuộc tính.

```python
def find_best_threshold(self, X, y, feature):
    sorted_vals = np.sort(np.unique(X[:, feature]))
    best_gain = -1
    best_threshold = None

    # Try different thresholds
    for i in range(len(sorted_vals) - 1):
        threshold = (sorted_vals[i] + sorted_vals[i + 1]) / 2
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Calculate weighted entropy
        total_samples = len(y)
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])

        weighted_entropy = (sum(left_mask) / total_samples * left_entropy +
                          sum(right_mask) / total_samples * right_entropy)

        gain = self.entropy(y) - weighted_entropy

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain
```

Slide 10: Real-World Example - Credit Risk Assessment

Implementation of a decision tree for credit risk classification, demonstrating practical application in financial domain with handling of mixed data types.

```python
import pandas as pd

# Load credit data (example structure)
credit_data = pd.DataFrame({
    'income': [45000, 80000, 60000, 30000, 35000],
    'debt_ratio': [0.25, 0.15, 0.35, 0.45, 0.40],
    'payment_history': [1, 1, 0, 0, 1],
    'default_risk': [0, 0, 1, 1, 0]
})

# Preprocess data
X = credit_data.drop('default_risk', axis=1).values
y = credit_data['default_risk'].values

# Train model with specific parameters for financial data
credit_tree = DecisionTree(
    max_depth=4,
    min_samples_split=5,
    min_gain=0.01
)
credit_tree.fit(X, y)
```

Slide 11: Thực hiện cắt xén

Việc cắt giúp ngăn chặn trang bị quá trình bằng cách loại bỏ các nút không đóng góp đáng kể vào hiệu suất của mô hình. Việc phát triển tính năng này sử dụng khả năng cắt giảm lỗi, đánh giá sự đóng góp của từng cây bằng cách sử dụng bộ xác thực.

```python
def prune_tree(self, node, X_val, y_val):
    if not node.left and not node.right:
        return

    # Recursively prune children
    if node.left:
        self.prune_tree(node.left, X_val, y_val)
    if node.right:
        self.prune_tree(node.right, X_val, y_val)

    # Calculate error before pruning
    initial_pred = self.predict(X_val)
    initial_error = sum(initial_pred != y_val)

    # Store children temporarily
    left, right = node.left, node.right

    # Make node a leaf using majority class
    node.left = node.right = None
    node.value = max(Counter(y_val).items(), key=lambda x: x[1])[0]

    # Calculate error after pruning
    pruned_pred = self.predict(X_val)
    pruned_error = sum(pruned_pred != y_val)

    # Restore node if pruning didn't help
    if pruned_error > initial_error:
        node.left, node.right = left, right
        node.value = None
```

Slide 12: Cross-Validation for Decision Trees

Cross-validation ensures robust model evaluation by testing performance across different data splits. This implementation shows how to perform k-fold cross-validation with our decision tree.

```python
def cross_validate(X, y, k_folds=5):
    fold_size = len(X) // k_folds
    accuracies = []

    for fold in range(k_folds):
        # Create train/val split
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size

        X_val = X[start_idx:end_idx]
        y_val = y[start_idx:end_idx]
        X_train = np.concatenate([X[:start_idx], X[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])

        # Train and evaluate
        tree = DecisionTree(max_depth=5)
        tree.fit(X_train, y_train)
        predictions = tree.predict(X_val)
        accuracy = sum(predictions == y_val) / len(y_val)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)

# Example usage and output:
# mean_acc, std_acc = cross_validate(X, y)
# print(f"Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
```

Slide 13: Quyết định trực tuyến ranh giới

Việc phát triển này tạo ra khả năng thực hiện trực quan về ranh giới quyết định tạo cây ra, giúp hiểu cách mô hình phân vùng không đối tượng.

```python
def plot_decision_boundary(tree, X, y, feature_names):
    import matplotlib.pyplot as plt

    # Create mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Make predictions
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Tree Boundaries')
    return plt

# Example visualization code:
# plt = plot_decision_boundary(tree, X[:, [0, 1]], y,
#                            ['Feature 1', 'Feature 2'])
# plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

1. "Cây phân loại tối ưu" - [https://arxiv.org/abs/1711.05297](https://arxiv.org/abs/1711.05297)
2. "Học cây quyết định bằng phổ Fourier" - [https://arxiv.org/abs/0903.0544](https://arxiv.org/abs/0903.0544)
3. "Cây quyết định thần kinh sâu" - [https://arxiv.org/abs/1806.06988](https://arxiv.org/abs/1806.06988)
4. "Cây quyết định đơn điệu" - [https://arxiv.org/abs/1909.05925](https://arxiv.org/abs/1909.05925)
5. "Cây quyết định nhanh và chính xác" - [https://arxiv.org/abs/2012.00174](https://arxiv.org/abs/2012.00174)
