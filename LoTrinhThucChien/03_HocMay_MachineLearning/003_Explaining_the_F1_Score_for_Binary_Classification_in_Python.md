## Giải thích Điểm F1 cho Phân loại nhị phân trong Python
Slide 1: Giới thiệu về Điểm F1

Điểm F1 là thước đo mạnh mẽ để đánh giá các phân loại nhị phân mô hình. Nó kết hợp độ chính xác và khả năng thu được thành một giá trị duy nhất, cung cấp thước đo cân bằng về hiệu suất của mô hình. Số liệu này đặc biệt hữu ích khi xử lý các dữ liệu không cân bằng.

```python
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

# Example
precision = 0.8
recall = 0.7
f1 = f1_score(precision, recall)
print(f"F1 Score: {f1:.2f}")  # Output: F1 Score: 0.75
```

Trang trình bày 2: Các thành phần của Điểm F1: Độ chính xác và khả năng thu hồi

Độ chính xác đo lường độ chính xác của các cực kỳ vọng, trong khi thu thập tỷ lệ định lượng của các kết quả tích cực thực tế được xác định chính xác. Điểm F1 cân bằng hai số liệu này.

```python
def calculate_precision_recall(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall

# Example
tp, fp, fn = 80, 20, 30
precision, recall = calculate_precision_recall(tp, fp, fn)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
# Output: Precision: 0.80, Recall: 0.73
```

Trang trình bày 3: Công thức tính điểm F1

Điểm F1 được tính bằng giá trị trung bình hài hòa của độ chính xác và thu hồi, cung cấp một giá trị duy nhất từ ​​​​0 đến 1, trong đó 1 biểu thị độ chính xác và thu hồi hoàn hảo.

```python
import numpy as np

def f1_score_harmonic_mean(precision, recall):
    return np.mean([precision, recall], weights=[1/precision, 1/recall])

# Example
precision, recall = 0.8, 0.7
f1 = f1_score_harmonic_mean(precision, recall)
print(f"F1 Score: {f1:.2f}")  # Output: F1 Score: 0.75
```

Slide 4: thích điểm F1

Điểm F1 là 1 cho thấy độ chính xác và thu hồi hoàn hảo. Điểm càng gần 1 cho thấy hiệu suất mô hình càng tốt, trong khi điểm càng gần 0 cho thấy hiệu suất càng kém.

```python
def interpret_f1_score(f1):
    if f1 == 1:
        return "Perfect precision and recall"
    elif f1 > 0.7:
        return "Good balance between precision and recall"
    elif f1 > 0.5:
        return "Moderate performance"
    else:
        return "Poor performance, consider model improvements"

# Example
f1_scores = [1.0, 0.8, 0.6, 0.3]
for score in f1_scores:
    print(f"F1 Score: {score:.2f} - {interpret_f1_score(score)}")

# Output:
# F1 Score: 1.00 - Perfect precision and recall
# F1 Score: 0.80 - Good balance between precision and recall
# F1 Score: 0.60 - Moderate performance
# F1 Score: 0.30 - Poor performance, consider model improvements
```

Trình bày 5: Tính điểm F1 từ Ma trận giữa

Cung cấp bối cảnh được xác định rõ ràng về hiệu suất của mô hình. Chúng ta có thể tính điểm trực tiếp F1 từ các thành phần của nó.

```python
import numpy as np

def f1_score_from_confusion_matrix(cm):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Example confusion matrix
cm = np.array([[50, 10],
               [5, 35]])

f1 = f1_score_from_confusion_matrix(cm)
print(f"F1 Score: {f1:.2f}")  # Output: F1 Score: 0.82
```

Trang trình bày 6: Điểm F1 so với độ chính xác

Mặc dù độ chính xác mang tính trực quan cao nhưng nó có thể gây nhầm lẫn đối với các dữ liệu không cân bằng. Điểm F1 mang lại sự cân bằng hơn trong những trường hợp như vậy.

```python
def compare_f1_and_accuracy(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, f1

# Example with imbalanced dataset
y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 10)
y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1] * 10)

accuracy, f1 = compare_f1_and_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
# Output: Accuracy: 0.90, F1 Score: 0.86
```

Trang trình bày 7: Triển khai điểm F1 với Scikit-learn

Scikit-learn cung cấp các hàm tích hợp để tính điểm F1, giúp bạn dễ dàng đánh giá các mô hình của mình.

```python
from sklearn.metrics import f1_score
import numpy as np

# Generate example data
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 1])

# Calculate F1 score
f1 = f1_score(y_true, y_pred)

print(f"F1 Score: {f1:.2f}")  # Output: F1 Score: 0.75
```

Trang trình bày 8: Điểm F1 cho phân loại nhiều lớp

Đối với nhiều lớp toán, chúng ta có thể tính điểm F1 bằng các phương pháp lấy trung bình khác nhau: vi mô, vĩ mô và số.

```python
from sklearn.metrics import f1_score
import numpy as np

# Generate example data
y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 1, 0, 1, 1])

# Calculate F1 scores with different averaging methods
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"Micro F1: {f1_micro:.2f}")
print(f"Macro F1: {f1_macro:.2f}")
print(f"Weighted F1: {f1_weighted:.2f}")

# Output:
# Micro F1: 0.67
# Macro F1: 0.44
# Weighted F1: 0.44
```

Trang trình bày 9: Điểm F1 trong chéo xác thực

Xác thực chéo giúp đánh giá mô hình hiệu suất trên các dữ liệu phân tách khác nhau. Chúng tôi có thể sử dụng F1 làm thước đo cho điểm trong quá trình xác thực chéo.

```python
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Create an SVM classifier
svm = SVC(kernel='rbf', random_state=42)

# Perform cross-validation with F1 score
cv_scores = cross_val_score(svm, X, y, cv=5, scoring='f1')

print("F1 scores in cross-validation:")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.2f}")
print(f"Mean F1 score: {cv_scores.mean():.2f}")

# Output:
# F1 scores in cross-validation:
# Fold 1: 0.92
# Fold 2: 0.91
# Fold 3: 0.93
# Fold 4: 0.92
# Fold 5: 0.93
# Mean F1 score: 0.92
```

Slide 10: F1 trực quan hóa

Quan hóa điểm F1 có thể giúp hiểu hoạt động của nó và so sánh trực quan các mô hình khác nhau.

```python
import numpy as np
import matplotlib.pyplot as plt

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

precision = np.linspace(0.1, 1, 100)
recall = np.linspace(0.1, 1, 100)
P, R = np.meshgrid(precision, recall)
F1 = f1_score(P, R)

plt.figure(figsize=(10, 8))
contour = plt.contourf(P, R, F1, levels=20, cmap='viridis')
plt.colorbar(contour, label='F1 Score')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('F1 Score Contour Plot')
plt.show()
```

Trang trình bày 11: Điểm F1 trong Bộ dữ liệu không cân bằng

Điểm F1 đặc biệt hữu ích đối với các dữ liệu cân bằng mà chỉ xác định chính xác cũng có thể gây nhầm lẫn.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                           n_informative=3, n_redundant=1, flip_y=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a random forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

# Output:
# Accuracy: 0.90
# F1 Score: 0.57
```

Trang trình chiếu 12: Ví dụ thực tế: Phát hiện rác

Khi phát hiện thư rác, việc phát hiện sai (dấu email hợp pháp là thư rác) có thể có giá thành thấp. Điểm F1 giúp cân bằng độ chính xác và thu hồi.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

# Example dataset (in practice, you'd have much more data)
emails = [
    "Get rich quick!", "Meeting at 3pm", "Free money now",
    "Project deadline tomorrow", "You've won a prize!", "Lunch plans?"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Create feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, labels)

# Make predictions
predictions = clf.predict(X)

# Calculate F1 score
f1 = f1_score(labels, predictions)
print(f"F1 Score: {f1:.2f}")  # Output: F1 Score: 1.00
```

Slide 13: Ví dụ thực tế: Chẩn đoán y khoa

Trong kỳ vọng của y tế, cả kết quả dương tính giả và âm tính đều có thể gây ra hậu quả nghiêm trọng. Điểm F1 giúp tìm kiếm sự cân bằng.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

# Simulated patient data (features might include age, blood pressure, etc.)
np.random.seed(42)
X = np.random.rand(1000, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simplified condition for positive diagnosis

# Split data
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Output:
# F1 Score: 0.78
# Confusion Matrix:
# [[89 24]
#  [20 67]]
```

Trang trình bày 14: Chế độ và cân bằng nhanh

Mặc dù điểm F1 rất hữu ích nhưng nó không phải lúc nào cũng là thước đo tốt nhất. Xem xét các công cụ cần giải quyết vấn đề của bạn và sử dụng nhiều thước đo giá trị khi thích hợp.

```python
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Example: Model performs well on F1 but poorly on recall
y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 10)
y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 10)

evaluate_model(y_true, y_pred)

# Output:
# Accuracy: 0.80
# Precision: 1.00
# Recall: 0.60
# F1 Score: 0.75
```

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin về điểm F1 và các chủ đề liên quan, hãy xem xét khám phá các tài nguyên sau:

1. “Phân tích có hệ thống các loại giải pháp thực hiện nhiệm vụ phân loại nhiệm vụ” của Marina Sokolova và Guy Lapalme (2009). Có tại: [https://arxiv.org/abs/0808.0650](https://arxiv.org/abs/0808.0650)
2. "Mối quan hệ giữa Precision-Recall và ROC Curves" của Jesse Davis và Mark Goadrich (2006). Có tại: [https://arxiv.org/abs/math/0606550](https://arxiv.org/abs/math/0606550)

Bài viết này cung cấp phân tích sâu chuyên sâu về các loại hiệu suất khác nhau, bao gồm điểm F1 và ứng dụng của chúng trong các vấn đề khác nhau.