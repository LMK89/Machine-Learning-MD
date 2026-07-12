## Các thành phần của Ma trận hỗn loạn trong phân loại nhị phân
Trang trình bày 1: Tìm hiểu các phần hỗn hợp của các thành phần

Ma trận nhầm lẫn đóng vai trò là thước đo đánh giá cơ bản trong phân loại nhị phân, bao gồm bốn thành phần thiết yếu để đo lường sự liên kết giữa giá trị dự đoán và giá trị thực tế. Các thành phần này tạo thành cơ sở để tính toán các số liệu hiệu suất quan trọng trong các mô hình học máy.

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_confusion_matrix(y_true, y_pred):
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    return cm

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

cm = create_confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
```

Trang trình bày 2: Triển khai các thước đo hiệu suất

Số liệu hiệu suất bắt nguồn từ các thành phần ma trận nhầm lẫn cung cấp thông tin chi tiết toàn diện về hành vi của mô hình. Những tính toán này giúp đánh giá hiệu quả của mô hình trên các khía cạnh khác nhau của hiệu suất phân loại.

```python
def calculate_metrics(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Advanced metrics
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)  # Negative Predictive Value

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Specificity': specificity,
        'NPV': npv
    }
    return metrics
```

Slide 3: Visualizing Confusion Matrix

Creating effective visualizations of confusion matrices enhances interpretation and communication of model performance. This implementation uses seaborn to generate an informative heatmap with annotated values and percentage calculations.

```python
def plot_confusion_matrix(cm, labels=['Negative', 'Positive']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Calculate percentages
    total = np.sum(cm)
    percentages = cm / total * 100

    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'({percentages[i,j]:.1f}%)',
                    ha='center', va='center')
    return plt
```

Trang trình bày 4: Ví dụ thực tế - Phát hiện gian lận thẻ tín dụng

Việc triển khai này thể hiện sự phân tích ma trận nhầm lẫn trong việc phát hiện gian lận thẻ tín dụng, trong đó sự mất cân bằng giữa các lớp là một thách thức chung. Ví dụ này bao gồm tiền xử lý dữ liệu và xử lý các lớp không cân bằng.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def prepare_fraud_detection_data(X, y):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Apply SMOTE for balance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, X_test, y_train_balanced, y_test

# Example usage with dummy data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.random.binomial(1, 0.1, 1000)
```

Trang trình bày 5: Cơ sở toán học của các số liệu ma trận nhầm lẫn

Mối quan hệ toán học giữa các thành phần ma trận nhầm lẫn tạo thành cơ sở cho các số liệu hiệu suất khác nhau. Những công thức này cung cấp nền tảng lý thuyết để hiểu được việc đánh giá mô hình.

```python
# Mathematical formulas in LaTeX notation
formulas = """
Accuracy: $$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$

Precision: $$\\text{Precision} = \\frac{TP}{TP + FP}$$

Recall: $$\\text{Recall} = \\frac{TP}{TP + FN}$$

F1-Score: $$\\text{F1} = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$

Specificity: $$\\text{Specificity} = \\frac{TN}{TN + FP}$$
"""
```

Trang trình bày 6: Triển khai các số liệu nâng cao và đường cong ROC

Đường cong Đặc tính hoạt động của máy thu (ROC) cung cấp thông tin chi tiết về hiệu suất của mô hình qua các ngưỡng phân loại khác nhau. Việc triển khai này tính toán TPR và FPR cho các giá trị ngưỡng khác nhau.

```python
def calculate_roc_curve(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)
    tpr_list, fpr_list = [], []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        cm = create_confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds
```

Slide 7: Precision-Recall Curve Implementation

The Precision-Recall curve is particularly useful for imbalanced datasets, providing a more informative view of model performance than accuracy alone.

```python
def calculate_pr_curve(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)
    precision_list, recall_list = [], []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        cm = create_confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    return np.array(precision_list), np.array(recall_list), thresholds
```

Trang trình bày 8: Xác thực chéo bằng Ma trận nhầm lẫn

Việc triển khai xác thực chéo với các số liệu ma trận nhầm lẫn đảm bảo đánh giá mô hình mạnh mẽ trên các phần tách dữ liệu khác nhau, cung cấp các ước tính hiệu suất đáng tin cậy hơn.

```python
from sklearn.model_selection import KFold
import numpy as np

def cross_validate_confusion_matrix(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Calculate confusion matrix
        cm = create_confusion_matrix(y_val, y_pred)
        metrics = calculate_metrics(cm)
        metrics_per_fold.append(metrics)

    # Calculate average metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in metrics_per_fold])
                  for metric in metrics_per_fold[0].keys()}

    return avg_metrics, metrics_per_fold
```

Trang trình bày 9: Phân tích ma trận nhầm lẫn chuỗi thời gian

Phân tích ma trận nhầm lẫn trong bối cảnh chuỗi thời gian đòi hỏi phải xem xét đặc biệt về sự phụ thuộc thời gian và đánh giá cửa sổ trượt.

```python
def time_series_confusion_matrix(y_true, y_pred, window_size=30):
    total_length = len(y_true)
    window_metrics = []

    for start_idx in range(0, total_length - window_size + 1):
        end_idx = start_idx + window_size

        # Calculate confusion matrix for current window
        window_cm = create_confusion_matrix(
            y_true[start_idx:end_idx],
            y_pred[start_idx:end_idx]
        )

        # Calculate metrics for window
        window_metrics.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'metrics': calculate_metrics(window_cm)
        })

    return window_metrics
```

Slide 10: Implementing Cost-Sensitive Confusion Matrix

Cost-sensitive analysis assigns different weights to classification errors, crucial for scenarios where certain types of mistakes are more costly than others.

```python
def cost_sensitive_evaluation(confusion_matrix, cost_matrix):
    """
    Cost matrix format:
    [[TN_cost, FP_cost],
     [FN_cost, TP_cost]]
    """
    tn, fp, fn, tp = confusion_matrix.ravel()
    cost_tn, cost_fp, cost_fn, cost_tp = cost_matrix.ravel()

    # Calculate total cost
    total_cost = (tn * cost_tn + fp * cost_fp +
                 fn * cost_fn + tp * cost_tp)

    # Calculate cost-adjusted metrics
    cost_precision = (tp * cost_tp) / (tp * cost_tp + fp * cost_fp)
    cost_recall = (tp * cost_tp) / (tp * cost_tp + fn * cost_fn)

    return {
        'total_cost': total_cost,
        'cost_precision': cost_precision,
        'cost_recall': cost_recall
    }
```

Slide 11: Ví dụ thực tế - Hệ thống chẩn đoán y tế

Việc triển khai này thể hiện việc phân tích ma trận nhầm lẫn trong bối cảnh chẩn đoán y tế, trong đó kết quả âm tính giả có thể gây ra hậu quả nghiêm trọng và cần phải xử lý đặc biệt.

```python
def medical_diagnosis_evaluation(y_true, y_pred, disease_prevalence=0.1):
    cm = create_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate clinical metrics
    sensitivity = tp / (tp + fn)  # Same as recall
    specificity = tn / (tn + fp)

    # Positive and Negative Predictive Values adjusted for prevalence
    ppv = (sensitivity * disease_prevalence) / \
          (sensitivity * disease_prevalence + (1 - specificity) * (1 - disease_prevalence))
    npv = (specificity * (1 - disease_prevalence)) / \
          ((1 - sensitivity) * disease_prevalence + specificity * (1 - disease_prevalence))

    # Calculate likelihood ratios
    positive_lr = sensitivity / (1 - specificity)
    negative_lr = (1 - sensitivity) / specificity

    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'Positive_LR': positive_lr,
        'Negative_LR': negative_lr
    }
```

Slide 12: Bootstrapped Confidence Intervals for Confusion Matrix Metrics

Computing confidence intervals through bootstrapping provides statistical reliability measures for confusion matrix metrics.

```python
def bootstrap_confusion_matrix_metrics(y_true, y_pred, n_iterations=1000, confidence=0.95):
    bootstrap_metrics = []
    n_samples = len(y_true)

    for _ in range(n_iterations):
        # Random sampling with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]

        # Calculate confusion matrix and metrics
        cm = create_confusion_matrix(boot_y_true, boot_y_pred)
        metrics = calculate_metrics(cm)
        bootstrap_metrics.append(metrics)

    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    metric_cis = {}

    for metric in bootstrap_metrics[0].keys():
        values = [m[metric] for m in bootstrap_metrics]
        lower = np.percentile(values, alpha * 100)
        upper = np.percentile(values, (1 - alpha) * 100)
        metric_cis[metric] = (lower, upper)

    return metric_cis
```

Trang trình bày 13: Triển khai ma trận nhầm lẫn nhiều lớp

Việc mở rộng các khái niệm ma trận nhầm lẫn nhị phân sang các kịch bản nhiều lớp đòi hỏi phải cân nhắc bổ sung và tính toán số liệu.

```python
def multiclass_confusion_matrix(y_true, y_pred, classes):
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Build confusion matrix
    for i in range(len(y_true)):
        true_idx = np.where(classes == y_true[i])[0][0]
        pred_idx = np.where(classes == y_pred[i])[0][0]
        cm[true_idx, pred_idx] += 1

    # Calculate per-class metrics
    per_class_metrics = {}
    for i, class_label in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn

        metrics = {
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'F1-score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        }
        per_class_metrics[class_label] = metrics

    return cm, per_class_metrics
```

Slide 14: Additional Resources

*   "Deep Learning with Confusion Matrices: Visualization and Performance Enhancement" [https://arxiv.org/abs/2107.02192](https://arxiv.org/abs/2107.02192)
*   "Statistical Analysis of Confusion Matrix Metrics for Imbalanced Data" [https://arxiv.org/abs/2106.09645](https://arxiv.org/abs/2106.09645)
*   "Confidence Intervals for Performance Metrics in Binary Classification" [https://arxiv.org/abs/2003.01200](https://arxiv.org/abs/2003.01200)
*   "Multi-class Confusion Matrix Analysis: A Comprehensive Review" [https://arxiv.org/abs/2108.05288](https://arxiv.org/abs/2108.05288)
*   "Cost-Sensitive Learning with Confusion Matrix Optimization" [https://arxiv.org/abs/2105.09541](https://arxiv.org/abs/2105.09541)
