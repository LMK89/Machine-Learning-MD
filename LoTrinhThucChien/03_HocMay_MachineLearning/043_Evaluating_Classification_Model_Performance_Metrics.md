##Đánh giá hiệu suất dữ liệu của loại phân loại mô hình
Trang trình bày 1: Tìm hiểu các phần hỗn hợp của các thành phần

Ma trận hỗn hợp cung cấp thông tin cần mô tả về loại hiệu suất của hình học bằng cách sắp xếp các dự kiến ​​thành loại cơ sở: Kết quả dương tính thực (TP), Âm tính thực (TN), Kết quả dương tính giả (FP) và Âm tính giả (FN). Tổ chức này cho phép phân tích biểu hiện toàn diện.

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create heatmap visualization
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Example usage
    y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0]
    plot_confusion_matrix(y_true, y_pred)
```

Slide 2: Tính toán chính xác

Độ chính xác có thể hiển thị tỷ lệ dự kiến ​​​​trong số tất cả các dự đoán đúng được thực hiện. Mặc dù được sử dụng phổ biến nhưng nó có thể không phù hợp với các loại dữ liệu bị mất cân bằng trong đó một lớp vượt trội hơn đáng kể so với các lớp khác.

```python
def calculate_accuracy(y_true, y_pred):
    # Convert inputs to numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate accuracy using mathematical formula
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    # Alternative using confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy_cm = (tp + tn) / (tp + tn + fp + fn)

    return accuracy, accuracy_cm
```

Slide 3: Implementing Precision Metric

Precision measures the accuracy of positive predictions by calculating the ratio of true positives to all positive predictions. This metric is crucial in applications where false positives are particularly costly or undesirable.

```python
def calculate_precision(y_true, y_pred):
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate precision with error handling for division by zero
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision
```

Slide 4: Nhớ lại máy tính (Độ nhạy)

Việc thu thập khả năng định lượng của mô hình trong việc xác định chính xác tất cả các trường hợp tích tích. Số liệu này đặc biệt quan trọng trong dự đoán y tế và phát hiện khoa học viễn tưởng khi thiếu các trường hợp dương tính có thể gây ra hậu quả nghiêm trọng.

```python
def calculate_recall(y_true, y_pred):
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate recall with error handling
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall
```

Slide 5: Implementing Specificity Metric

Specificity measures the model's ability to correctly identify negative instances, complementing sensitivity in providing a complete picture of model performance, especially important in medical screening and security applications.

```python
def calculate_specificity(y_true, y_pred):
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate specificity with error handling
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity
```

Trang trình bày 6: Thực hiện điểm F1

Điểm F1 cung cấp thước đo cân bằng hiệu suất của mô hình bằng cách tính giá trị trung bình hài hòa của độ chính xác và khả năng thu hồi. Số liệu này đặc biệt hữu ích khi xử lý các dữ liệu cân bằng mà chỉ xác định chính xác cũng có thể gây nhầm lẫn.

```python
def calculate_f1_score(y_true, y_pred):
    # Calculate precision and recall
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)

    # Calculate F1-score with error handling
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0.0

    return f1_score
```

Slide 7: Mathematical Foundations

The fundamental mathematical relationships between confusion matrix components and evaluation metrics form the basis for understanding model performance assessment in classification tasks.

```python
"""
Accuracy Formula:
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

Precision Formula:
$$Precision = \frac{TP}{TP + FP}$$

Recall Formula:
$$Recall = \frac{TP}{TP + FN}$$

Specificity Formula:
$$Specificity = \frac{TN}{TN + FP}$$

F1-Score Formula:
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
"""
```

Trang trình bày 8: Ví dụ thực tế: Phát hiện mãng thẻ tín dụng

Việc phát triển các loại vật liệu hỗn hợp trong việc phát hiện thẻ tín dụng nano để tìm ra tầm quan trọng của số liệu đánh giá cân bằng cách xử lý các dữ liệu cân bằng bằng cao điển hình trong các vấn đề phát hiện gian nan.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def fraud_detection_example():
    # Generate synthetic fraud data
    np.random.seed(42)
    n_samples = 10000

    # Create imbalanced dataset (0.1% fraud)
    X = np.random.randn(n_samples, 5)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, size=int(n_samples*0.001))
    y[fraud_indices] = 1

    # Split and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred
```

Trang trình bày 9: Kết quả phát hiện thẻ tín dụng nano

```python
y_test, y_pred = fraud_detection_example()

metrics = {
    'Accuracy': calculate_accuracy(y_test, y_pred)[0],
    'Precision': calculate_precision(y_test, y_pred),
    'Recall': calculate_recall(y_test, y_pred),
    'Specificity': calculate_specificity(y_test, y_pred),
    'F1-Score': calculate_f1_score(y_test, y_pred)
}

print("Fraud Detection Results:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Trang trình chiếu 10: Ví dụ thực tế: Chẩn đoán y khoa

Chẩn đoán y tế đòi hỏi phải xem xét cẩn thận các kết quả âm tính giả và dương tính giả, làm cho nó trở thành một nghiên cứu điển hình tuyệt vời để hiểu được tầm quan trọng thực tế của các số liệu đánh giá khác nhau trong các loại phân tích nhiệm vụ.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def medical_diagnosis_example():
    # Generate synthetic medical data
    np.random.seed(42)
    n_patients = 1000

    # Create features (symptoms and test results)
    X = np.random.randn(n_patients, 10)  # 10 medical indicators

    # Generate diagnoses (1: disease present, 0: healthy)
    # Assuming 15% disease prevalence
    y = np.zeros(n_patients)
    disease_indices = np.random.choice(n_patients, size=int(n_patients*0.15))
    y[disease_indices] = 1

    # Preprocess and split data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

    # Train model
    model = SVC(kernel='rbf', class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred
```

Slide 11: Kết quả mô hình dự đoán y khoa

```python
# Execute medical diagnosis example
y_test, y_pred = medical_diagnosis_example()

# Calculate comprehensive metrics
def print_medical_metrics(y_test, y_pred):
    metrics = {
        'Accuracy': calculate_accuracy(y_test, y_pred)[0],
        'Precision': calculate_precision(y_test, y_pred),
        'Recall (Sensitivity)': calculate_recall(y_test, y_pred),
        'Specificity': calculate_specificity(y_test, y_pred),
        'F1-Score': calculate_f1_score(y_test, y_pred)
    }

    print("\nMedical Diagnosis Model Results:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")

print_medical_metrics(y_test, y_pred)
```

Trang trình bày 12: Triển khai xác thực chéo bằng dữ liệu

Xác thực chéo cung cấp máy mạnh hơn về hiệu suất mô hình bằng cách đánh giá số liệu trên nhiều dữ liệu phân tích, điều cần thiết để đánh giá mô tả đáng tin cậy trong môi trường sản xuất.

```python
from sklearn.model_selection import KFold
import numpy as np

def cross_validate_metrics(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_dict = {
        'accuracy': [], 'precision': [],
        'recall': [], 'specificity': [],
        'f1': []
    }

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics_dict['accuracy'].append(calculate_accuracy(y_test, y_pred)[0])
        metrics_dict['precision'].append(calculate_precision(y_test, y_pred))
        metrics_dict['recall'].append(calculate_recall(y_test, y_pred))
        metrics_dict['specificity'].append(calculate_specificity(y_test, y_pred))
        metrics_dict['f1'].append(calculate_f1_score(y_test, y_pred))

    return {k: np.mean(v) for k, v in metrics_dict.items()}
```

Trang trình bày 13: Số liệu đánh giá trực quan

Tìm hiểu mối quan hệ giữa các vật liệu khác nhau giúp lựa chọn ngưỡng thích hợp và đưa ra quyết định sáng suốt về việc phát triển mô hình trong các ứng dụng trong thế giới thực.

```python
def plot_metric_tradeoffs(y_true, y_pred_proba):
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, specificities = [], [], []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precisions.append(calculate_precision(y_true, y_pred))
        recalls.append(calculate_recall(y_true, y_pred))
        specificities.append(calculate_specificity(y_true, y_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, specificities, label='Specificity')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metric Trade-offs vs Classification Threshold')
    plt.legend()
    plt.grid(True)
    return plt
```

Slide 14: Additional Resources

*   "A systematic analysis of performance measures for classification tasks" - [https://arxiv.org/abs/1909.03622](https://arxiv.org/abs/1909.03622)
*   "Beyond Accuracy: Precision and Recall" - [https://arxiv.org/abs/1502.05893](https://arxiv.org/abs/1502.05893)
*   "The Relationship Between Precision-Recall and ROC Curves" - [https://arxiv.org/abs/1608.04802](https://arxiv.org/abs/1608.04802)
*   "On the Modern Theory of Classification" - [https://arxiv.org/abs/1804.09281](https://arxiv.org/abs/1804.09281)
*   "A Survey of Evaluation Metrics for Classification Models" - [https://arxiv.org/abs/2002.05274](https://arxiv.org/abs/2002.05274)
