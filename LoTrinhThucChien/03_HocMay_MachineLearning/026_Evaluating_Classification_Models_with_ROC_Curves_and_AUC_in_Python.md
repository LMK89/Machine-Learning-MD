##Đánh giá các loại phân loại mô hình bằng đường cong ROC và AUC trong Python
Slide 1: Giới thiệu về ROC Curves và AUC

Đường công ROC (Đặc tính hoạt động của máy thu) và AUC (Khu vực dưới đường cong) là những công cụ mạnh mẽ để đánh giá và so sánh các loại phân loại mô hình. Chúng tôi cung cấp trình bày trực quan về hiệu suất của các mô hình qua các loại phân loại ngưỡng khác nhau và đưa ra một số liệu duy nhất để tắt hiệu suất đó.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Example data
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_scores = np.array([0.1, 0.7, 0.8, 0.3, 0.9, 0.6, 0.2, 0.4, 0.7, 0.5])

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

Trang trình bày 2: Tìm hiểu tỷ lệ dương tính thật và tỷ lệ dương tính giả

Tỷ lệ dương tính thực tế (TPR) và Tỷ lệ dương tính giả (FPR) là các thành phần chính của đường cong ROC. TPR, còn được gọi là độ nhạy hoặc cường độ thu hồi, đo tỷ lệ các trường hợp lý tính thực tế được xác định chính xác. FPR đại diện cho tỷ lệ các trường hợp âm tính thực tế được phân loại không chính xác là dương tính.

```python
def calculate_tpr_fpr(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))

    tpr = true_positives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)

    return tpr, fpr

# Example usage
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 1])

tpr, fpr = calculate_tpr_fpr(y_true, y_pred)
print(f"True Positive Rate: {tpr:.2f}")
print(f"False Positive Rate: {fpr:.2f}")
```

Slide 3: Tạo đường cong ROC

Để tạo đường cong ROC, chúng tôi cần tính TPR và FPR cho các loại phân loại ngưỡng khác nhau. Chúng ta sẽ sử dụng hàm roc\_curve của scikit-learn để tạo các dữ liệu cần thiết.

```python
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

Slide 4: Tính diện dưới đường cong (AUC)

Vùng bên dưới đường cong ROC (AUC) cung cấp một giá trị vô hướng duy nhất để đo lường hiệu suất tổng thể của các loại phân loại. AUC dao động từ 0 đến 1, với 0,5 đại diện cho bộ phân loại ngẫu nhiên và 1 đại diện cho bộ phân loại hoàn hảo.

```python
from sklearn.metrics import roc_auc_score
import numpy as np

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate AUC
auc = roc_auc_score(y_true, y_scores)

print(f"Area Under the Curve (AUC): {auc:.3f}")

# Interpret AUC
if auc < 0.5:
    print("Poor performance (worse than random)")
elif auc < 0.7:
    print("Fair performance")
elif auc < 0.8:
    print("Good performance")
elif auc < 0.9:
    print("Very good performance")
else:
    print("Excellent performance")
```

Trang trình bày 5: So sánh nhiều loại phân loại

Đường công ROC và AUC đặc biệt hữu ích để so sánh hiệu suất của nhiều loại bộ trên cùng một dữ liệu. Điều này cho phép chúng tôi đánh giá một cách trực quan và định lượng bất kỳ hoạt động nào tốt hơn so với các loại phân loại khác nhau.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifiers
lr = LogisticRegression()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict probabilities
lr_probs = lr.predict_proba(X_test)[:, 1]
rf_probs = rf.predict_proba(X_test)[:, 1]

# Calculate ROC curves and AUC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

lr_auc = auc(lr_fpr, lr_tpr)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiple Classifiers')
plt.legend()
plt.show()
```

Slide 6: Xử lý bộ dữ liệu không cân bằng

Khi làm việc với các dữ liệu không cân bằng, đường cong ROC không thể cung cấp bức tranh hoàn chỉnh về hiệu suất của mô hình. Trong những trường hợp như vậy, sẽ rất hữu ích khi xem xét các đường cong Precision-Recall cùng với các đường cong ROC.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_probs = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve and Average Precision
precision, recall, _ = precision_recall_curve(y_test, y_probs)
avg_precision = average_precision_score(y_test, y_probs)

# Plot ROC curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Plot Precision-Recall curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()
```

Trang trình bày 7: Xác thực chéo để ước tính AUC mạnh mẽ

Để có được tính chất đáng tin cậy hơn về hiệu suất của mô hình, chúng tôi có thể sử dụng xác thực chéo để tính điểm AUC trên nhiều phần dữ liệu.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Initialize classifier and cross-validation
clf = LogisticRegression()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
auc_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf.fit(X_train, y_train)
    y_probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_probs)
    auc_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.3f}")

# Calculate mean and standard deviation of AUC scores
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
print(f"\nMean AUC: {mean_auc:.3f} (+/- {std_auc:.3f})")
```

Trang trình bày 8: Loại phân loại ngưỡng tối ưu

Ngưỡng phân loại mặc định thường là 0,5, nhưng chúng tôi có thể tối ưu hóa ngưỡng này dựa trên đường cong ROC để tìm ra sự cân bằng tốt nhất giữa tỷ lệ dương tính thật và tỷ lệ dương tính giả.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_probs = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")

# Apply optimal threshold
y_pred_optimal = (y_probs >= optimal_threshold).astype(int)

# Calculate accuracy with optimal threshold
accuracy_optimal = np.mean(y_pred_optimal == y_test)
print(f"Accuracy with optimal threshold: {accuracy_optimal:.3f}")

# Compare with default threshold
y_pred_default = (y_probs >= 0.5).astype(int)
accuracy_default = np.mean(y_pred_default == y_test)
print(f"Accuracy with default threshold: {accuracy_default:.3f}")
```

Slide 9: Quyết định ranh giới trực tuyến

Để hiểu rõ hơn mối liên hệ giữa đường cong ROC với ranh giới được xác định của mô hình, chúng tôi có thể hình dung ranh giới quyết định theo chiều dọc theo đường cong ROC cho một tệp 2D đơn giản.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Generate 2D dataset
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_redundant=0, n_informative=2, random_state=42)

# Train logistic regression
clf = LogisticRegression()
clf.fit(X, y)

# Create a mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict probabilities for the mesh grid
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Calculate ROC curve and AUC
y_pred_proba = clf.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot decision boundary and data points
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')

# Plot ROC curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
```

Trang trình bày 10: Đường công ROC để phân loại nhiều lớp

Mặc dù ROC đường cong thường được sử dụng để phân loại nhị phân, nhưng chúng có thể được mở rộng cho nhiều lớp toán toán bằng cách sử dụng phương pháp một đối số.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a multi-class classifier
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)

# Compute ROC curve and ROC area for each class
y_score = clf.predict_proba(X_test)
n_classes = len(np.unique(y))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc="lower right")
plt.show()
```

Trang trình bày 11: Độ tin cậy cho AUC

Để đánh giá độ tin cậy của AUC, chúng tôi có thể tính toán độ tin cậy bằng cách sử dụng bootstrapping.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, ci=0.95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        # Bootstrap by sampling with replacement
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Compute confidence interval
    confidence_lower = sorted_scores[int((1.0-ci)/2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1.0+ci)/2 * len(sorted_scores))]
    return np.mean(bootstrapped_scores), confidence_lower, confidence_upper

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_pred = clf.predict_proba(X_test)[:, 1]

# Calculate AUC and confidence interval
auc, ci_lower, ci_upper = bootstrap_auc(y_test, y_pred)

print(f"AUC: {auc:.3f}")
print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

Trang trình bày 12: AUC one part

Trong một số ứng dụng, chúng tôi có thể chỉ quan tâm đến một vùng cụ thể của đường cong ROC. Một phần AUC cho phép chúng tôi tập trung vào một công cụ giả lập tỷ lệ vi phạm.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

def partial_auc(fpr, tpr, max_fpr):
    # Find the index of the first FPR value greater than max_fpr
    cut_point = next(i for i, x in enumerate(fpr) if x > max_fpr)

    # Linearly interpolate the TPR at max_fpr
    slope = (tpr[cut_point] - tpr[cut_point-1]) / (fpr[cut_point] - fpr[cut_point-1])
    tpr_interp = tpr[cut_point-1] + slope * (max_fpr - fpr[cut_point-1])

    # Compute partial AUC
    partial_auc = np.trapz([tpr[0]] + list(tpr[:cut_point]) + [tpr_interp], [0] + list(fpr[:cut_point]) + [max_fpr])
    return partial_auc / max_fpr

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_pred = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Calculate partial AUC
max_fpr = 0.2
pauc = partial_auc(fpr, tpr, max_fpr)

# Plot ROC curve and partial AUC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, max_fpr], [0, tpr[np.argmax(fpr > max_fpr)]], color='red', lw=2, linestyle='--', label=f'Partial AUC (FPR <= {max_fpr})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve and Partial AUC (pAUC = {pauc:.3f})')
plt.legend(loc="lower right")
plt.show()
```

Trang trình bày 13: Đường công ROC cho bộ dữ liệu không cân bằng

Khi xử lý các bộ dữ liệu không cân bằng, điều quan trọng là phải xem xét các lựa chọn thay thế cho đường cong ROC, hạn chế như đường Precision-Recall, có thể cung cấp cái nhìn nhiều thông tin hơn về hiệu suất của mô hình.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve and Average Precision
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Plot ROC curve and Precision-Recall curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random classifier')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
ax1.legend(loc="lower right")

ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc="lower left")

plt.tight_layout()
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về các đường cong ROC, AUC và các chủ đề liên quan, đây là một số tài nguyên có giá trị:

1. Fawcett, T. (2006). Giới thiệu về ROC phân tích. Mẫu nhận dạng chữ, 27(8), 861-874. Liên kết ArXiv: [https://arxiv.org/abs/cs/0303029](https://arxiv.org/abs/cs/0303029)
2. Bradley, A. P. (1997). Việc sử dụng mô hình dưới đường cong ROC trong việc đánh giá các máy học thuật toán. Đã nhận mẫu dạng, 30(7), 1145-1159.
3. Davis, J., & Goadrich, M. (2006). Mối quan hệ giữa các đường cong Precision-Recall và ROC. Kỷ yếu của Hội nghị Quốc tế lần thứ 23 về học máy. Liên kết ArXiv: [https://arxiv.org/abs/cs/0606118](https://arxiv.org/abs/cs/0606118)
4. Hanley, J. A., & McNeil, B. J. (1982). Ý nghĩa và cách sử dụng vùng bên dưới đường cong đặc tính vận hành máy thu (ROC). X quang, 143(1), 29-36.

Tài nguyên này cung cấp các giải pháp thích hợp và phân tích chuyên sâu về đường cong ROC, AUC cũng như các ứng dụng của chúng trong máy học và thống kê.
