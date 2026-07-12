## Các trường hợp sử dụng phân loại Naive Bayes
Slide 1: Giới thiệu về Phân loại Naive Bayes

Bộ phân loại Naive Bayes là các thuật toán xác suất dựa trên định lý Bayes, giả sử tính độc lập của đặc trưng. Chúng vượt trội trong việc phân loại văn bản, lọc thư rác, phân tích cảm xúc và hệ thống đề xuất nhờ tính đơn giản và hiệu quả với dữ liệu nhiều chiều.

```python
# Basic structure of Naive Bayes classifier
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Mathematical representation of Bayes Theorem
"""
$$P(y|X) = \frac{P(X|y)P(y)}{P(X)}$$
Where:
- P(y|X) is posterior probability
- P(X|y) is likelihood
- P(y) is prior probability
- P(X) is evidence
"""

# Simple implementation
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

model = GaussianNB()
model.fit(X, y)
print(f"Prediction: {model.predict([[2.5, 3.5]])}")
```

Slide 2: Phân loại văn bản với các vịnh đa thức

Phân loại văn bản là một trong những ứng dụng phổ biến nhất của Naive Bayes, đặc biệt là sử dụng biến thể Đa thức. Việc triển khai này thể hiện việc phân loại tài liệu bằng cách sử dụng tần số từ làm đặc điểm.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample text documents
texts = [
    "This movie is fantastic",
    "Terrible waste of time",
    "Great film, highly recommended",
    "Awful movie, don't watch"
]
labels = [1, 0, 1, 0]  # 1: positive, 0: negative

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train classifier
clf = MultinomialNB()
clf.fit(X, labels)

# Predict new text
new_text = ["This film is amazing"]
new_X = vectorizer.transform(new_text)
print(f"Prediction: {clf.predict(new_X)}")
```

Slide 3: Hệ thống phát hiện thư rác

Naive Bayes vượt trội trong việc phát hiện thư rác qua email nhờ khả năng xử lý kích thước từ vựng lớn và thời gian đào tạo nhanh. Việc triển khai này cho thấy một hệ thống phát hiện thư rác hoàn chỉnh có tính năng xử lý trước văn bản.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample email data
emails = [
    "Get rich quick! Buy now!",
    "Meeting at 3pm tomorrow",
    "Win free prizes instantly",
    "Project deadline reminder"
]
labels = [1, 0, 1, 0]  # 1: spam, 0: not spam

# Create DataFrame
df = pd.DataFrame({'email': emails, 'spam': labels})

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['email'])
y = df['spam']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
```

Slide 4: Phân loại dữ liệu theo phân loại

Naive Bayes xử lý các đặc điểm phân loại một cách tự nhiên thông qua biến thể vịnh ngây thơ phân loại. Việc triển khai này thể hiện sự phân loại với các tính năng phân loại hỗn hợp.

```python
from sklearn.naive_bayes import CategoricalNB
import numpy as np

# Sample categorical data
X = np.array([
    [0, 1, 2],  # Feature 1: color (0:red, 1:blue, 2:green)
    [1, 0, 2],  # Feature 2: size (0:small, 1:medium, 2:large)
    [2, 1, 0],
    [0, 2, 1]
])
y = np.array([0, 1, 1, 0])  # Target classes

# Initialize and train model
cnb = CategoricalNB()
cnb.fit(X, y)

# Predict new instance
new_data = np.array([[1, 1, 2]])
print(f"Prediction: {cnb.predict(new_data)}")
print(f"Probability estimates: {cnb.predict_proba(new_data)}")
```

Slide 5: Phân loại chẩn đoán y khoa

Naive Bayes có thể phân loại các tình trạng bệnh lý một cách hiệu quả dựa trên các triệu chứng. Việc triển khai này cho thấy một hệ thống chẩn đoán y tế với nhiều tính năng và ước tính xác suất.

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Medical data (symptoms as features)
# Features: temperature, heart_rate, blood_pressure, pain_level
X = np.array([
    [38.5, 90, 140, 7],
    [37.0, 70, 120, 3],
    [39.0, 95, 150, 8],
    [36.8, 75, 125, 2]
])
y = np.array([1, 0, 1, 0])  # 1: condition present, 0: condition absent

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = GaussianNB()
model.fit(X_scaled, y)

# New patient data
new_patient = np.array([[38.2, 88, 135, 6]])
new_patient_scaled = scaler.transform(new_patient)

# Predict and get probabilities
prediction = model.predict(new_patient_scaled)
probabilities = model.predict_proba(new_patient_scaled)

print(f"Diagnosis: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Confidence: {max(probabilities[0])*100:.2f}%")
```

Trang trình bày 6: Thực hiện phân tích tình cảm

Phân tích tình cảm là một ứng dụng chính của Naive Bayes trong xử lý ngôn ngữ tự nhiên. Việc triển khai này thể hiện một bộ phân tích cảm tính hoàn chỉnh để đánh giá sản phẩm với các số liệu đánh giá và tiền xử lý.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

# Sample product reviews
reviews = [
    "This product exceeded my expectations",
    "Terrible quality, broke after one use",
    "Amazing value for money, highly satisfied",
    "Poor customer service, wouldn't recommend",
    "Great features and reliable performance"
]
sentiments = np.array([1, 0, 1, 0, 1])  # 1: positive, 0: negative

# Create pipeline
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', MultinomialNB())
])

# Train the model
sentiment_pipeline.fit(reviews, sentiments)

# Test new reviews
new_reviews = ["The product works perfectly", "Waste of money"]
predictions = sentiment_pipeline.predict(new_reviews)
probabilities = sentiment_pipeline.predict_proba(new_reviews)

for review, pred, prob in zip(new_reviews, predictions, probabilities):
    print(f"Review: {review}")
    print(f"Sentiment: {'Positive' if pred == 1 else 'Negative'}")
    print(f"Confidence: {max(prob)*100:.2f}%\n")
```

Trang trình bày 7: Gaussian Naive Bayes cho các tính năng liên tục

Gaussian Naive Bayes giả định các tính năng tuân theo phân phối chuẩn, làm cho nó phù hợp cho việc phân loại dữ liệu liên tục. Việc triển khai này thể hiện ứng dụng của nó với các tính năng số và ước tính mật độ xác suất.

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Generate synthetic continuous data
np.random.seed(42)
n_samples = 100

# Create two classes with different distributions
class1_x = np.random.normal(0, 1, (n_samples, 2))
class2_x = np.random.normal(2, 1, (n_samples, 2))
X = np.vstack([class1_x, class2_x])
y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

# Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X, y)

# Create grid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Make predictions on the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(class1_x[:, 0], class1_x[:, 1], label='Class 0')
plt.scatter(class2_x[:, 0], class2_x[:, 1], label='Class 1')
plt.legend()
plt.title('Gaussian Naive Bayes Decision Boundary')
plt.show()
```

Trang trình bày 8: Bernoulli Naive Bayes về các tính năng nhị phân

Bernoulli Naive Bayes chuyên về phân loại tính năng nhị phân, lý tưởng cho việc phân loại tài liệu dựa trên sự hiện diện của từ thay vì tần số. Việc triển khai này cho thấy ứng dụng của nó trong việc phân loại chủ đề.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import binarize

# Sample documents
documents = [
    "python programming code development",
    "machine learning algorithms data",
    "web development html css",
    "deep learning neural networks",
    "database sql queries"
]
topics = [0, 1, 0, 1, 0]  # 0: development, 1: ML

# Convert to binary features
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(documents)

# Train Bernoulli NB
bnb = BernoulliNB()
bnb.fit(X, topics)

# Test new documents
new_docs = ["python web development", "neural network training"]
X_new = vectorizer.transform(new_docs)

# Get predictions and probabilities
predictions = bnb.predict(X_new)
probabilities = bnb.predict_proba(X_new)

for doc, pred, prob in zip(new_docs, predictions, probabilities):
    print(f"Document: {doc}")
    print(f"Predicted topic: {'Development' if pred == 0 else 'Machine Learning'}")
    print(f"Confidence: {max(prob)*100:.2f}%\n")
```

Trang trình bày 9: Các vịnh ngây thơ bổ sung cho bộ dữ liệu không cân bằng

Naive Bayes bổ sung thích ứng với các tập dữ liệu không cân bằng bằng cách sử dụng xác suất có điều kiện của lớp bổ sung. Việc triển khai này chứng tỏ tính hiệu quả của nó đối với việc phân phối lớp sai lệch.

```python
from sklearn.naive_bayes import ComplementNB
from sklearn.datasets import make_imbalanced
import numpy as np

# Generate imbalanced dataset
n_samples = 1000
weights = [0.9, 0.1]  # 90% class 0, 10% class 1

# Create features and labels
X = np.random.rand(n_samples, 5)
y = np.random.choice([0, 1], size=n_samples, p=weights)

# Train Complement NB
cnb = ComplementNB()
cnb.fit(X, y)

# Compare with standard Multinomial NB
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X, y)

# Evaluate on balanced test set
X_test = np.random.rand(100, 5)
y_test = np.array([0, 1] * 50)

print("Complement NB accuracy:", cnb.score(X_test, y_test))
print("Multinomial NB accuracy:", mnb.score(X_test, y_test))

# Show class probabilities
new_sample = np.random.rand(1, 5)
print("\nComplement NB probabilities:", cnb.predict_proba(new_sample))
print("Multinomial NB probabilities:", mnb.predict_proba(new_sample))
```

Slide 10: Lựa chọn tính năng với Naive Bayes

Lựa chọn tính năng là rất quan trọng để cải thiện hiệu suất của Naive Bayes bằng cách loại bỏ các tính năng không liên quan. Việc triển khai này thể hiện thông tin lẫn nhau và các phương pháp lựa chọn tính năng chi bình phương với Naive Bayes.

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import numpy as np

# Generate dataset with irrelevant features
n_samples = 300
n_features = 20
n_informative = 5

# Create informative and noise features
X_informative = np.random.randn(n_samples, n_informative)
X_noise = np.random.randn(n_samples, n_features - n_informative)
X = np.hstack((X_informative, X_noise))
y = (X_informative[:, 0] + X_informative[:, 1] > 0).astype(int)

# Create pipeline with feature selection
pipeline_mi = Pipeline([
    ('feature_selection', SelectKBest(mutual_info_classif, k=5)),
    ('classification', GaussianNB())
])

# Train and evaluate
pipeline_mi.fit(X, y)

# Get selected feature indices
selected_features = pipeline_mi.named_steps['feature_selection'].get_support()
print("Selected features:", np.where(selected_features)[0])

# Evaluate model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline_mi, X, y, cv=5)
print(f"Average accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

Trang trình bày 11: Phân loại theo thời gian thực với Naive Bayes

Naive Bayes hoạt động hiệu quả đối với các nhiệm vụ phân loại theo thời gian thực nhờ khả năng dự đoán nhanh chóng. Việc triển khai này cho thấy một hệ thống phân loại phát trực tuyến đã được điều chỉnh một phần.

```python
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from time import sleep

class StreamingClassifier:
    def __init__(self):
        self.classifier = MultinomialNB(partial_fit_classes=[0, 1])
        self.batch_size = 10
        self.current_batch = []
        self.current_labels = []

    def process_sample(self, features, label):
        self.current_batch.append(features)
        self.current_labels.append(label)

        if len(self.current_batch) >= self.batch_size:
            X_batch = np.array(self.current_batch)
            y_batch = np.array(self.current_labels)

            # Partial fit on current batch
            self.classifier.partial_fit(X_batch, y_batch)

            # Clear batch
            self.current_batch = []
            self.current_labels = []

            return True
        return False

# Simulate streaming data
stream_clf = StreamingClassifier()
n_samples = 100

for i in range(n_samples):
    # Generate random sample
    feature = np.random.randint(0, 5, size=10)
    label = int(sum(feature) > 25)

    # Process sample
    batch_processed = stream_clf.process_sample(feature, label)

    if batch_processed:
        print(f"Processed batch {i//10}, Current accuracy: "
              f"{stream_clf.classifier.score(feature.reshape(1, -1), [label]):.3f}")

    sleep(0.1)  # Simulate real-time delay
```

Slide 12: Phân loại nhiều lớp

Naive Bayes mở rộng một cách tự nhiên đến các bài toán đa lớp mà không cần sửa đổi. Việc triển khai này thể hiện sự phân loại nhiều lớp với hiệu chuẩn xác suất.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Generate multi-class data
n_samples = 300
n_classes = 4

# Create features and labels
X = np.random.randn(n_samples, 5)
y = np.random.randint(0, n_classes, n_samples)

# Create and calibrate classifier
base_clf = GaussianNB()
calibrated_clf = CalibratedClassifierCV(base_clf, cv=5, method='isotonic')

# Train calibrated classifier
calibrated_clf.fit(X, y)

# Make predictions
new_samples = np.random.randn(5, 5)
predictions = calibrated_clf.predict(new_samples)
probabilities = calibrated_clf.predict_proba(new_samples)

# Print results
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"\nSample {i+1}:")
    print(f"Predicted class: {pred}")
    print("Class probabilities:")
    for class_idx, class_prob in enumerate(prob):
        print(f"Class {class_idx}: {class_prob:.3f}")
```

Trang trình bày 13: Tài nguyên bổ sung

* "Naive Bayes và Phân loại văn bản I - Lý thuyết" - [https://arxiv.org/abs/1410.5329](https://arxiv.org/abs/1410.5329)
* "So sánh thực tế về hiệu suất của các biến thể Naive Bayes khác nhau" - [https://www.sciencedirect.com/science/article/pii/S2352340920313202](https://www.sciencedirect.com/science/article/pii/S2352340920313202)
* "Về phân loại phân biệt đối xử và tạo sinh: So sánh hồi quy logistic và Bayes ngây thơ" - [https://proceedings.neurips.cc/apers/2001/file/7b7a53e239400a13bd6be6c91c4f6c4e-Paper.pdf] (https://proceedings.neurips.cc/apers/2001/file/7b7a53e239400a13bd6be6c91c4f6c4e-Paper.pdf)
* Để biết thêm tài liệu nghiên cứu và cách triển khai, hãy tìm kiếm "Phân loại Naive Bayes" trên Google Scholar hoặc arXiv
