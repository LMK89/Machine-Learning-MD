## Phân loại máy học
Slide 1: Giới thiệu về phân loại

Phân loại là một cơ sở nhiệm vụ trong máy học, nơi chúng tôi mong đợi các lớp nhãn riêng biệt cho đầu vào dữ liệu. Nó được sử dụng rộng rãi trong nhiều ứng dụng khác nhau, từ phát hiện thư rác đến dự đoán y tế.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

print(f"Accuracy: {clf.score(X_test, y_test):.2f}")
```

Slide 2: Phân loại nhị phân

Phân loại nhị phân phân loại liên quan đến việc phân loại các trường hợp thành một trong hai lớp. Nó thường được sử dụng trong các vấn đề như phát hiện thư rác hoặc dự đoán bệnh.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict for a new point
new_point = np.array([[1.5, 0.5]])
prediction = model.predict(new_point)
probability = model.predict_proba(new_point)

print(f"Predicted class: {prediction[0]}")
print(f"Probability of class 1: {probability[0][1]:.2f}")
```

Slide 3: Phân loại nhiều lớp

Phân loại nhiều lớp phân loại nhị phân mở rộng cho các vấn đề có nhiều hơn hai lớp. Nó được sử dụng trong các vấn đề như nhận dạng chữ số hoặc nhận dạng loại.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load iris dataset (3 classes)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM classifier
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_scaled, y)

# Predict for a new sample
new_sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
prediction = svm.predict(new_sample)

print(f"Predicted class: {iris.target_names[prediction[0]]}")
```

Trang trình bày 4: Phân loại nhiều nhãn

Phân loại đa nhãn cho phép mỗi phiên bản thuộc về nhiều lớp cùng một lúc. Nó hữu ích trong các vấn đề như gắn thẻ hình ảnh hoặc phân loại tài liệu.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic multilabel data
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.random.randint(2, size=(100, 3))

# Train multilabel classifier
forest = RandomForestClassifier(n_estimators=100, random_state=42)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(X, y)

# Predict for a new sample
new_sample = np.array([[0.5, 1.2, -0.3, 0.8, -1.5]])
prediction = multi_target_forest.predict(new_sample)

print(f"Predicted labels: {prediction[0]}")
```

Slide 5: Ma trận trộn lẫn

Ma trận hỗn loạn là một loại hiệu suất hóa trực tiếp của các loại mô hình, hiển thị số lượng kết quả dương tính thực, âm tính thực, dương tính giả và âm tính giả.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate synthetic predictions
y_true = np.array([0, 1, 2, 2, 1, 0, 1, 0, 2, 1])
y_pred = np.array([0, 2, 1, 2, 1, 0, 1, 0, 2, 1])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

Trang trình bày 6: Độ chính xác

Độ chính xác đo độ chính xác của cực kỳ vọng. Đó là tỷ lệ số lượng kết quả dương tính thực sự trên tổng số lượng dự kiến ​​​​cực.

```python
from sklearn.metrics import precision_score

# Binary classification example
y_true = [0, 1, 1, 0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.2f}")

# Calculate precision manually
true_positives = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
predicted_positives = sum(yp == 1 for yp in y_pred)
manual_precision = true_positives / predicted_positives
print(f"Manual Precision: {manual_precision:.2f}")
```

Trang trình bày 7: Nhớ lại

Thu thập các phép đo khả năng tìm thấy ở tất cả các trường hợp tích cực. Đó là tỷ lệ các trường hợp lý tính thực tế trên tổng số trường hợp dương tính thực tế.

```python
from sklearn.metrics import recall_score

# Binary classification example
y_true = [0, 1, 1, 0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.2f}")

# Calculate recall manually
true_positives = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
actual_positives = sum(yt == 1 for yt in y_true)
manual_recall = true_positives / actual_positives
print(f"Manual Recall: {manual_recall:.2f}")
```

Trang trình bày 8: Điểm F1

Điểm F1 là giá trị trung bình hài hòa của độ chính xác và khả năng thu hồi, cung cấp một điểm duy nhất cân bằng cả hai chỉ số. Nó đặc biệt hữu ích khi bạn có phân tích bổ sung lớp học không đồng đều.

```python
from sklearn.metrics import f1_score

# Binary classification example
y_true = [0, 1, 1, 0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.2f}")

# Calculate F1 score manually
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
manual_f1 = 2 * (precision * recall) / (precision + recall)
print(f"Manual F1 Score: {manual_f1:.2f}")
```

Trang trình bày 9: Đường công ROC và AUC

Đường đặc tính hoạt động của máy thu (ROC) và Zone under the path (AUC) được sử dụng để đánh giá hiệu suất của các loại nhị phân phân loại trên các ngưỡng cài đặt khác nhau.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
y_true = np.random.randint(2, size=100)
y_scores = np.random.rand(100)

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

Trang trình bày 10: Xác thực chéo

Xác thực chéo là một kỹ thuật được sử dụng để đánh giá hiệu suất của hình và ngăn chặn công việc bằng cách chia dữ liệu thành nhiều huấn luyện và xác thực.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.2f}")
print(f"Standard deviation of CV scores: {cv_scores.std():.2f}")
```

Trang trình bày 11: Ví dụ thực tế: Phân tích cảm xúc

Phân tích cảm xúc là một ứng dụng phổ biến của phân loại văn bản, được sử dụng để xác định sắc thái cảm xúc sau từ ngữ.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample tweets
tweets = [
    "I love this product! It's amazing!",
    "This is the worst experience ever.",
    "Neutral opinion about this service.",
    "Absolutely fantastic customer support!",
    "Disappointed with the quality."
]
sentiments = [1, 0, 2, 1, 0]  # 1: positive, 0: negative, 2: neutral

# Create a pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(tweets, sentiments)

# Predict sentiment for a new tweet
new_tweet = ["The product exceeded my expectations!"]
prediction = pipeline.predict(new_tweet)

sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
print(f"Predicted sentiment: {sentiment_map[prediction[0]]}")
```

Slide 12: Ví dụ thực tế: Phân loại hình ảnh

Phân loại hình ảnh được sử dụng rộng rãi trong các ứng dụng thị giác máy tính, từ việc nhận dạng khuôn mặt đến siêu hình ảnh y tế.

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess an image
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Print top 3 predictions
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}: {label} ({score:.2f})")
```

Slide 13: Lỗi phân tích

Phân tích lỗi bao gồm việc kiểm tra các trường hợp được phân loại sai để hiểu điểm yếu của mô hình và hướng dẫn cải tiến.

```python
from sklearn.metrics import classification_report
import pandas as pd

# Assuming we have true labels and predictions
y_true = [0, 1, 2, 2, 1, 0, 1, 0, 2, 1]
y_pred = [0, 2, 1, 2, 1, 0, 1, 0, 2, 1]

# Generate classification report
report = classification_report(y_true, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

print(df_report)

# Identify misclassified instances
misclassified = [(true, pred) for true, pred in zip(y_true, y_pred) if true != pred]
print("\nMisclassified instances (true label, predicted label):")
for true, pred in misclassified:
    print(f"True: {true}, Predicted: {pred}")
```

Slide 14: thích mô hình: Tầm quan trọng của tính năng

Hiểu tầm quan trọng của tính năng giúp diễn đàn giải quyết các quyết định về mô hình và có thể hướng dẫn các nỗ lực kỹ thuật tính năng.

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
feature_names = iris.feature_names

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về kỹ thuật phân loại và số hiệu suất, hãy xem xét các bài viết được bình duyệt sau:

1. "Khảo sát các kỹ thuật học sâu để phân loại hình ảnh" - arXiv:2009.09809
2. "Hiểu về ma trận hỗn loạn" - arXiv:2008.05786
3. "Giới thiệu về Phân tích ROC" - arXiv:2008.04635

Những tài nguyên này cung cấp các bài thảo luận chuyên sâu về các chủ đề nâng cao trong phân loại máy học.
