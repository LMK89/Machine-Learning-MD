## Xử lý bộ dữ liệu không cân bằng trong bảng dạng phân loại

Slide 1: Bộ dữ liệu cân bằng trong phân loại

Bộ mất cân bằng dữ liệu là một biến số phổ biến trong các loại bảng nhiệm vụ. Chúng xảy ra khi một lớp đông hơn đáng kể so với các lớp khác, dẫn đến các mô hình sai lệch hoạt động kém đối với các lớp tối thiểu. Sự mất cân bằng này thường cố hữu trong dữ liệu trong thế giới thực, được cho là có giới hạn như phát hiện đột phá hoặc dự đoán bệnh độc gặp. Hiểu biết và giải quyết vấn đề này là rất quan trọng để phát triển các mô hình phân loại hiệu quả.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate an imbalanced dataset
np.random.seed(42)
majority_class = np.random.normal(0, 1, (1000, 2))
minority_class = np.random.normal(3, 1, (100, 2))

# Visualize the imbalanced dataset
plt.figure(figsize=(10, 6))
plt.scatter(majority_class[:, 0], majority_class[:, 1], label='Majority Class', alpha=0.5)
plt.scatter(minority_class[:, 0], minority_class[:, 1], label='Minority Class', alpha=0.5)
plt.legend()
plt.title('Imbalanced Dataset Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

Trang trình bày 2: Kỹ thuật thu thập mẫu quá trình

Lấy mẫu quá mức là một cách tiếp cận phổ biến để giải quyết sự mất cân bằng giữa các lớp. Nó liên quan đến việc tăng số lượng phiên bản ở mức tối thiểu để cân bằng dữ liệu. Có nhiều kỹ thuật lấy mẫu quá mức khác nhau, bao gồm lấy mẫu quá ngẫu nhiên và các phương pháp phức tạp hơn như SMOTE (Kỹ thuật lấy mẫu quá mức tổng hợp tối thiểu). Những kỹ thuật này nhằm mục đích cải thiện hiệu suất hiển thị ở mức tối thiểu mà không làm mất thông tin ở lớp đa số.

```python
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Apply random oversampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print(f"Original dataset shape: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Random oversampled shape: {dict(zip(*np.unique(y_ros, return_counts=True)))}")
print(f"SMOTE oversampled shape: {dict(zip(*np.unique(y_smote, return_counts=True)))}")
```

Slide 3: SMOTE (Kỹ thuật lấy số liệu tổng hợp tối thiểu)

SMOTE là một phương pháp nâng cao mẫu nhằm tạo ra các ví dụ tổng hợp trong không gian cụ thể. Nó hoạt động bằng cách chọn các loại có thể hiện ở mức tối thiểu và nội dung có thể mới giữa chúng và các loại lân cận gần nhất. Cách tiếp cận này nhắm đến mục tiêu tạo ra các dạng đa dạng và đại diện hơn cho các tầng lớp tối thiểu, có khả năng cải thiện khả năng hóa học của mô hình.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote(X, y, k=5, n_samples=100):
    minority_class = X[y == 1]
    nn = NearestNeighbors(n_neighbors=k+1).fit(minority_class)

    synthetic_samples = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(minority_class))
        sample = minority_class[idx]
        neighbors = nn.kneighbors([sample], return_distance=False)[0][1:]
        nn_idx = np.random.choice(neighbors)
        nn_sample = minority_class[nn_idx]

        alpha = np.random.random()
        new_sample = sample + alpha * (nn_sample - sample)
        synthetic_samples.append(new_sample)

    return np.vstack([X, synthetic_samples]), np.hstack([y, np.ones(n_samples)])

# Example usage
X = np.random.randn(100, 2)
y = np.hstack([np.zeros(90), np.ones(10)])
X_resampled, y_resampled = smote(X, y, n_samples=90)
print(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
```

Slide 4: Lợi ích của SMOTE

SMOTE cung cấp một số lợi ích trong việc xử lý dữ liệu không cân bằng. Bằng cách tạo ra các ví dụ tổng hợp, nó làm tăng tính đa dạng của số lớp tối thiểu, điều này có thể dẫn đến ranh giới quyết định tốt hơn và cải thiện khả năng hóa học độc đáo. SMOTE có thể giúp ngăn chặn hoạt động của trang ở mức tối đa và nâng cao khả năng của mô hình trong việc nhận các mẫu ở mức tối thiểu. Kỹ thuật này đặc biệt hữu ích khi tầng lớp tối thiểu có số lượng ít đại diện và việc thu thập dữ liệu thực tế bổ sung khó khăn hoặc rẻ tiền.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train without SMOTE
clf_no_smote = RandomForestClassifier(random_state=42)
clf_no_smote.fit(X_train, y_train)

# Apply SMOTE and train
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_train_smote, y_train_smote)

# Compare results
print("Without SMOTE:")
print(classification_report(y_test, clf_no_smote.predict(X_test)))
print("\nWith SMOTE:")
print(classification_report(y_test, clf_smote.predict(X_test)))
```

Slide 5: Những hạn chế bảo tàng của SMOTE

Mặc dù SMOTE có thể có lợi nhưng không phải lúc nào nó cũng là giải pháp tối ưu. SMOTE có thể tạo ra nhiễu hoặc tạo ra các ví dụ tổng hợp không thực tế, đặc biệt là trong không gian nhiều chiều hoặc với các thùng dữ liệu phân tích. Điều này có thể dẫn đến việc trang bị quá trình độ hoặc tạo ra các mẫu nhân vật không tồn tại trong quá trình thực thi dữ liệu. Ngoài ra, SMOTE giả định rằng không có đối tượng nào liên tục và được phép nội dung giữa các mẫu có ý nghĩa, điều này có thể không đúng với tất cả các loại dữ liệu.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from imblearn.over_sampling import SMOTE

# Generate imbalanced, non-linear dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_minority = X[y == 1]
X_majority = X[y == 0][:100]
X_imbalanced = np.vstack([X_majority, X_minority])
y_imbalanced = np.hstack([np.zeros(100), np.ones(len(X_minority))])

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imbalanced, y_imbalanced)

# Visualize original and SMOTE-resampled data
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X_imbalanced[y_imbalanced == 0][:, 0], X_imbalanced[y_imbalanced == 0][:, 1], label='Majority')
plt.scatter(X_imbalanced[y_imbalanced == 1][:, 0], X_imbalanced[y_imbalanced == 1][:, 1], label='Minority')
plt.title('Original Imbalanced Dataset')
plt.legend()

plt.subplot(122)
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label='Majority')
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], label='Minority (SMOTE)')
plt.title('SMOTE-Resampled Dataset')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 6: Giới thiệu về SMOTE và Noise

SMOTE có thể vô hiệu hóa nhiễu đầu vào dữ liệu. Điều này xảy ra khi các tổng mẫu được tạo ở những vùng không thể xác định chính xác thực tế phân tích của số tầng tối thiểu. Ví dụ: trong các tập dữ liệu có các lớp chéo hoặc ranh giới được xác định tạp chất phức tạp, SMOTE có thể tạo các tổng hợp mẫu rơi vào khu vực lớp đa số, dẫn đến tăng cường sự hỗn hợp cho loại phân vùng.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# Generate an imbalanced dataset with overlapping classes
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                           n_clusters_per_class=1, class_sep=0.5, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Visualize original and SMOTE-resampled data
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Majority', alpha=0.5)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Minority', alpha=0.5)
plt.title('Original Imbalanced Dataset')
plt.legend()

plt.subplot(122)
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label='Majority', alpha=0.5)
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], label='Minority (SMOTE)', alpha=0.5)
plt.title('SMOTE-Resampled Dataset')
plt.legend()

plt.tight_layout()
plt.show()
```

Trang trình chiếu 7: Ví dụ thực tế: Phát hiện bệnh độc gặp

Hãy xem xét một vấn đề phát hiện bệnh hiếm gặp khi chỉ có 1% bệnh nhân bệnh. SMOTE có thể được áp dụng để cân bằng dữ liệu, nhưng nó có thể gây nhiễu bằng cách tạo ra các bệnh nhân tổng hợp với các triệu chứng không thực tế. Điều này có thể dẫn đến kết quả dương tính giả trong dự đoán của mô hình, có khả năng gây căng thẳng không cần thiết và phải thử nghiệm bổ sung đối với những người khỏe mạnh.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Generate synthetic patient data
np.random.seed(42)
n_samples = 10000
n_features = 10

X = np.random.randn(n_samples, n_features)
y = np.zeros(n_samples)
y[:100] = 1  # 1% of patients have the rare disease

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train without SMOTE
clf_no_smote = RandomForestClassifier(random_state=42)
clf_no_smote.fit(X_train, y_train)

# Apply SMOTE and train
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_train_smote, y_train_smote)

# Compare results
print("Without SMOTE:")
print(classification_report(y_test, clf_no_smote.predict(X_test)))
print("\nWith SMOTE:")
print(classification_report(y_test, clf_smote.predict(X_test)))
```

Trang trình bày 8: Ví dụ thực tế: Phân loại hình ảnh

Trong các nhiệm vụ phân loại hình ảnh, được xác định như xác định các vật thể lạ trong hình ảnh bảo vệ, SMOTE có thể gặp vấn đề. Việc tạo hình ảnh tổng hợp bằng cách nội suy giữa các hình ảnh hiện có có thể tạo ra hình ảnh phi thực tế hoặc vô nghĩa. Điều này có thể dẫn đến khả năng hóa học gần hơn và giảm hiệu suất khi áp dụng dữ liệu trong thế giới thực.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from imblearn.over_sampling import SMOTE

# Load digit dataset and select two classes
digits = load_digits()
X = digits.data[(digits.target == 0) | (digits.target == 1)]
y = digits.target[(digits.target == 0) | (digits.target == 1)]

# Make it imbalanced by reducing class 1
X_imbalanced = np.vstack([X[y == 0], X[y == 1][:10]])
y_imbalanced = np.hstack([y[y == 0], y[y == 1][:10]])

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imbalanced, y_imbalanced)

# Visualize original and synthetic images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes[0]):
    ax.imshow(X_imbalanced[y_imbalanced == 1][i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Original {i+1}")
    ax.axis('off')

for i, ax in enumerate(axes[1]):
    synthetic_idx = np.where((y_resampled == 1) & (y_imbalanced != 1))[0][i]
    ax.imshow(X_resampled[synthetic_idx].reshape(8, 8), cmap='gray')
    ax.set_title(f"Synthetic {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 9: Lựa chọn thay thế cho SMOTE

Mặc dù SMOTE có thể mang lại hiệu quả nhưng các kỹ thuật khác có thể phù hợp hơn tùy thuộc vào dữ liệu và vấn đề cụ thể. Các phương pháp lấy mẫu dưới đây, có giới hạn như Lấy mẫu ngẫu nhiên hoặc Tomek liên kết, giảm lớp đa số thay vì tăng số lượng tối thiểu. Các phương thức tập hợp như BalancedRandomForestClassifier kết hợp nhiều mô hình để xử lý tình trạng mất cân bằng. Ngoài ra, việc điều chỉnh số lớp hoặc sử dụng các chức năng mất mát chuyên dụng có thể giải quyết tình trạng mất cân bằng mà không cần chỉnh sửa dữ liệu.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Generate imbalanced dataset
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.99, 0.01], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with class weights
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_train, y_train)

# Evaluate
y_pred = rf_weighted.predict(X_test)
print("Random Forest with class weights:")
print(classification_report(y_test, y_pred))
```

Trang trình chiếu 10: Đánh giá nhu cầu về SMOTE

Trước khi áp dụng SMOTE, điều quan trọng là phải đánh giá xem nó có cần thiết và mang lại lợi ích cho vấn đề cụ thể của bạn hay không. Đánh giá các đặc điểm của dữ liệu, phân loại như phân phối lớp và các mối quan hệ hệ thống. Xem xét phạm vi vấn đề và kết quả cuối cùng của kết quả tính toán dương tính và âm tính giả. Đôi khi, sự mất cân bằng tự nhiên trong dữ liệu phản ánh sự phân tích trong thế giới thực và không nên thay đổi.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def evaluate_smote_necessity(X, y, cv=5):
    clf = RandomForestClassifier(random_state=42)

    # Evaluate without SMOTE
    scores_no_smote = cross_val_score(clf, X, y, cv=cv, scoring='f1')

    # Evaluate with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    scores_smote = cross_val_score(clf, X_resampled, y_resampled, cv=cv, scoring='f1')

    print(f"Mean F1-score without SMOTE: {np.mean(scores_no_smote):.3f}")
    print(f"Mean F1-score with SMOTE: {np.mean(scores_smote):.3f}")

    if np.mean(scores_smote) > np.mean(scores_no_smote):
        print("SMOTE appears to be beneficial for this dataset.")
    else:
        print("SMOTE does not seem to improve performance significantly.")

# Example usage
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)
evaluate_smote_necessity(X, y)
```

Slide 11: Điều chỉnh siêu thông số SMOTE

Khi sử dụng SMOTE, việc điều chỉnh cẩn thận các siêu tham số của nó là điều cần thiết để tối đa hóa hiệu quả hóa của nó đồng thời giảm thiểu những nhược điểm tiềm ẩn. Các tham số chính bao gồm chiến lược lấy mẫu (xác định tỷ lệ mong muốn giữa số mẫu nhỏ và số mẫu lớn) và số lượng lân cận gần nhất được sử dụng để nội suy. Tìm kiếm chuỗi xác thực có thể giúp tìm kiếm các tham số tối ưu cho dữ liệu cụ thể của bạn.

```python
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Define pipeline and parameters
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'smote__sampling_strategy': [0.1, 0.2, 0.5, 0.75, 1.0],
    'smote__k_neighbors': [3, 5, 7],
    'classifier__n_estimators': [50, 100, 200]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best F1-score:", grid_search.best_score_)
```

Slide 12: Kết hợp SMOTE với các kỹ thuật khác

Để giải quyết những hạn chế của SMOTE, hãy cân nhắc việc kết hợp nó với các kỹ thuật khác. Ví dụ: SMOTEENN (SMOTE với Hàng xóm gần nhất đã được chỉnh sửa) hoặc SMOTETomek (SMOTE với Tomek Liên kết) áp dụng SMOTE, sau đó lấy mẫu dưới để loại bỏ các nhiễu xung. Phương pháp kết hợp này có thể giúp tạo ra sự cân bằng dữ liệu hơn là đồng thời giảm nguy cơ gây nhiễu hoặc tổng hợp các mẫu không thực tế.

```python
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Initialize resampling methods
smote_tomek = SMOTETomek(random_state=42)
smote_enn = SMOTEENN(random_state=42)

# Resample the dataset
X_resampled_tomek, y_resampled_tomek = smote_tomek.fit_resample(X, y)
X_resampled_enn, y_resampled_enn = smote_enn.fit_resample(X, y)

# Evaluate using cross-validation
clf = RandomForestClassifier(random_state=42)

scores_original = cross_val_score(clf, X, y, cv=5, scoring='f1')
scores_tomek = cross_val_score(clf, X_resampled_tomek, y_resampled_tomek, cv=5, scoring='f1')
scores_enn = cross_val_score(clf, X_resampled_enn, y_resampled_enn, cv=5, scoring='f1')

print(f"Mean F1-score (Original): {scores_original.mean():.3f}")
print(f"Mean F1-score (SMOTETomek): {scores_tomek.mean():.3f}")
print(f"Mean F1-score (SMOTEENN): {scores_enn.mean():.3f}")
```

Slide 13: Giám sát và xác nhận kết quả SMOTE

Sau khi áp dụng SMOTE, điều quan trọng phải được theo dõi và xác định kết quả để đảm bảo các mẫu tổng hợp có ý nghĩa và mang lại lợi ích. Các kỹ thuật như t-SNE hoặc UMAP có thể giúp trực quan hóa dữ liệu chiều cao trước và sau SMOTE. Ngoài ra, việc so sánh các hiệu suất dữ liệu trên cả dữ liệu gốc và dữ liệu được lấy lại SMOTE mẫu bằng cách sử dụng xác thực chéo có thể cung cấp thông tin chi tiết về hiệu quả kỹ thuật.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE

def visualize_smote_results(X, y, X_resampled, y_resampled):
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    X_resampled_tsne = tsne.fit_transform(X_resampled)

    # Plot original and resampled data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label='Majority', alpha=0.5)
    ax1.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label='Minority', alpha=0.5)
    ax1.set_title('Original Data')
    ax1.legend()

    ax2.scatter(X_resampled_tsne[y_resampled == 0, 0], X_resampled_tsne[y_resampled == 0, 1], label='Majority', alpha=0.5)
    ax2.scatter(X_resampled_tsne[y_resampled == 1, 0], X_resampled_tsne[y_resampled == 1, 1], label='Minority (SMOTE)', alpha=0.5)
    ax2.set_title('SMOTE-Resampled Data')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Example usage
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], n_clusters_per_class=1, n_features=20, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

visualize_smote_results(X, y, X_resampled, y_resampled)
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về chủ đề bộ dữ liệu mất cân bằng và SMOTE, đây là một số tài nguyên có giá trị:

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Kỹ thuật thu thập mẫu quá mức tổng hợp tối thiểu. Tạp chí Nghiên cứu Trí tuệ Nhân tạo, 16, 321-357. ArXiv: [https://arxiv.org/abs/1106.1813](https://arxiv.org/abs/1106.1813)
2. Anh ấy, H., & Garcia, E. A. (2009). Học từ dữ liệu không cân bằng. Giao dịch của IEEE về Kỹ thuật Kiến thức và Dữ liệu, 21(9), 1263-1284. DOI: 10.1109/TKDE.2008.239
3. Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Học mất cân bằng: Hộp công cụ Python để giải quyết lời nói về các bộ dữ liệu mất cân bằng trong máy học. Tạp chí Nghiên cứu Học máy, 18(17), 1-5. ArXiv: [https://arxiv.org/abs/1609.06570](https://arxiv.org/abs/1609.06570)

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về bộ dữ liệu mất cân bằng, SMOTE và nhiều kỹ thuật khác để xử lý tình trạng mất cân bằng lớp học trong máy học.
