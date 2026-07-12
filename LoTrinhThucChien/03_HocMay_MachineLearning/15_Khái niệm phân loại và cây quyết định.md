## Khái niệm phân loại và quyết định
Slide 1: Giới thiệu về phân loại

Loại phân vùng là một cơ sở nhiệm vụ trong máy học, trong đó chúng tôi mong đợi danh mục đầu vào dựa trên các tính năng của nó. Nó được sử dụng rộng rãi trong nhiều lĩnh vực khác nhau, từ y học đến công nghệ.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

print(f"Accuracy: {clf.score(X_test, y_test):.2f}")
```

Slide 2: Ứng dụng của phân loại

Phân loại loại có nhiều ứng dụng trong thế giới thực. Trong quá trình chăm sóc sức khỏe, nó được sử dụng để mong đợi bệnh. Trong môi trường khoa học, nó giúp phân loại các loài thực vật. Hãy cùng khám phá một ví dụ về phân loại email là thư rác hoặc không phải thư rác.

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
emails = [
    ("Free gift waiting for you", "spam"),
    ("Meeting at 3pm today", "not spam"),
    ("Win a luxury vacation now", "spam"),
    ("Project report due tomorrow", "not spam")
]

# Prepare the data
X, y = zip(*emails)
df = pd.DataFrame({'email': X, 'label': y})

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'])

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, df['label'])

# Predict a new email
new_email = ["Claim your prize today"]
X_new = vectorizer.transform(new_email)
prediction = clf.predict(X_new)

print(f"The email '{new_email[0]}' is classified as: {prediction[0]}")
```

Trang trình bày 3: Cây quyết định: Một kỹ thuật phân loại mạnh mẽ

Cây định nghĩa là mô hình phân loại quyết định trực quan và dễ hiểu. Họ đã quyết định cách phân tích dữ liệu dựa trên các giá trị cụ thể, tạo ra dạng cấu trúc.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create and train a decision tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

Slide 4: Xây dựng cây quyết định

Hãy cùng tìm hiểu quy trình xây dựng quyết định bằng Iris dữ liệu. Chúng tôi sẽ sử dụng entropy làm tiêu chí để phân tách các nút.

```python
from sklearn.tree import DecisionTreeClassifier

# Create a decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Print the importance of each feature
for name, importance in zip(iris.feature_names, clf.feature_importances_):
    print(f"{name}: {importance:.4f}")

# Make predictions
y_pred = clf.predict(X_test)

# Print the accuracy
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

Trang trình bày 5: Phân loại K-Láng sinh gần nhất (KNN)

KNN là một kỹ thuật phân loại phổ biến khác. Nó phân loại các cơ sở dữ liệu dựa trên lớp đa số của k hàng xóm gần nhất của chúng.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create and train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy:.4f}")

# Visualize decision boundaries (for 2D data)
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundaries(X, y, model, ax=None):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if ax is None:
        ax = plt.gca()
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    return ax

# Use only first two features for visualization
X_2d = X[:, :2]
X_train_2d, X_test_2d, y_train, y_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)

knn_2d = KNeighborsClassifier(n_neighbors=3)
knn_2d.fit(X_train_2d, y_train)

plt.figure(figsize=(10, 8))
plot_decision_boundaries(X_2d, y, knn_2d)
plt.title('KNN Decision Boundaries')
plt.show()
```

Slide 6: Xử lý bộ dữ liệu không cân bằng

Trong các vấn đề thực tế, chúng tôi thường gặp phải các bộ dữ liệu mất cân bằng trong đó một lớp đông hơn đáng kể so với các lớp khác. Hãy cùng khám phá các kỹ thuật để giải quyết những thử thách này.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a classifier on imbalanced data
clf_imbalanced = DecisionTreeClassifier(random_state=42)
clf_imbalanced.fit(X_train, y_train)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train a classifier on balanced data
clf_balanced = DecisionTreeClassifier(random_state=42)
clf_balanced.fit(X_train_balanced, y_train_balanced)

# Compare results
print("Imbalanced Dataset Results:")
print(classification_report(y_test, clf_imbalanced.predict(X_test)))

print("\nBalanced Dataset Results:")
print(classification_report(y_test, clf_balanced.predict(X_test)))
```

Slide 7: Lựa chọn tính năng và tầm quan trọng

Tính năng lựa chọn rất quan trọng để xây dựng các hiệu ứng phân loại. Hãy khám phá cách chọn các tính năng quan trọng nhất bằng cách sử dụng bộ phân loại Rừng ngẫu nhiên.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for f, idx in enumerate(indices):
    print(f"{f+1}. Feature {iris.feature_names[idx]}: {importances[idx]:.4f}")

# Select features using SelectFromModel
selector = SelectFromModel(rf, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train a new classifier with selected features
clf_selected = DecisionTreeClassifier(random_state=42)
clf_selected.fit(X_train_selected, y_train)

print(f"\nAccuracy with all features: {clf.score(X_test, y_test):.4f}")
print(f"Accuracy with selected features: {clf_selected.score(X_test_selected, y_test):.4f}")
```

Trang trình bày 8: Xác thực chéo: Đảm bảo độ tin cậy của màn hình

Xác thực chéo là một kỹ thuật được sử dụng để đánh giá độ đặc biệt của mô hình đối với dữ liệu chưa được tìm thấy. Hãy phát triển tính xác thực chéo k-Fold.

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)

print("Cross-validation scores:", scores)
print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Standard deviation: {scores.std():.4f}")

# Visualize the cross-validation process
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

kf = KFold(n_splits=5, shuffle=True, random_state=42)

plt.figure(figsize=(12, 4))
for i, (train_index, val_index) in enumerate(kf.split(X)):
    plt.subplot(1, 5, i+1)
    plt.scatter(X[train_index, 0], X[train_index, 1], c='blue', alpha=0.6, label='Train')
    plt.scatter(X[val_index, 0], X[val_index, 1], c='red', alpha=0.6, label='Validation')
    plt.title(f"Fold {i+1}")
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 9: Điều chỉnh siêu thông số

Tối ưu hóa các siêu tham số của mô hình có thể cải thiện hiệu suất đáng kể của mô hình. Vui lòng sử dụng GridSearchCV để tìm các thông số tốt nhất để quyết định.

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate on the test set
best_clf = grid_search.best_estimator_
test_score = best_clf.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.4f}")
```

Slide 10: Phương pháp tập hợp: Rừng ngẫu nhiên

Rừng ngẫu nhiên là một tập hợp các cây quyết định, thường hoạt động tốt hơn các cây riêng lẻ. Hãy phát triển bộ phân loại Rừng ngẫu nhiên.

```python
from sklearn.ensemble import RandomForestClassifier

# Create and train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Compare feature importances
importances = rf.feature_importances_
for name, importance in zip(iris.feature_names, importances):
    print(f"{name}: {importance:.4f}")

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(iris.feature_names, importances)
plt.title('Feature Importances in Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

Slide 11: Xử lý phân loại nhiều lớp

Trong khi chúng tôi ta tập trung vào phân loại nhị phân, nhiều vấn đề thực tế liên quan đến nhiều lớp. Hãy khám phá nhiều loại phân loại bằng Iris dữ liệu.

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Binarize the output
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)

# Create and train the multi-class classifier
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42))
clf.fit(X_train, y_train)

# Compute ROC curve and ROC area for each class
y_score = clf.decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()
```

Trang trình chiếu 12: Ví dụ thực tế: Phân tích cảm xúc

Vui lòng áp dụng phân loại để phân tích cảm tính của các sản phẩm đánh giá giá bằng cách sử dụng mô hình túi từ phân loại Naive Bayes.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample dataset
reviews = [
    ("This product is amazing!", "positive"),
    ("Worst purchase ever.", "negative"),
    ("Decent quality for the price.", "neutral"),
    ("I love it!", "positive"),
    ("Don't waste your money.", "negative"),
    ("It's okay, nothing special.", "neutral")
]

# Prepare the data
texts, labels = zip(*reviews)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Make predictions
y_pred = clf.predict(X_test_vec)

# Print the classification report
print(classification_report(y_test, y_pred))

# Test with a new review
new_review = ["This product exceeded my expectations!"]
new_review_vec = vectorizer.transform(new_review)
prediction = clf.predict(new_review_vec)
print(f"The sentiment of '{new_review[0]}' is predicted as: {prediction[0]}")
```

Slide 13: Ví dụ thực tế: Phân loại hình ảnh

Phân loại hình ảnh là một ứng dụng phổ biến của máy học. Vui lòng sử dụng Mạng thần kinh chuyển đổi (CNN) đơn giản để phân loại các chữ số viết tay từ bộ dữ liệu MNIST.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions on a few test images
predictions = model.predict(test_images[:5])
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predictions[i].argmax()}, Actual: {test_labels[i]}')
    plt.show()
```

Slide 14: Các công thức trong phân loại

Việc phân loại phải đối mặt với một số thức, bao gồm:

1. Trang bị quá trình: Mô hình có thể hoạt động tốt trên huấn luyện dữ liệu nhưng dữ liệu không được tìm thấy ở mức độ gần.
2. Mất cân bằng lớp: Khi một lớp đông hơn đáng kể so với các lớp khác, dẫn đến các mô hình sai lệch.
3. Lựa chọn tính năng: Chọn các tính năng phù hợp nhất để cải thiện hiệu suất của mô hình.
4. Khả năng mở rộng: Xử lý các dữ liệu lớn và không có hiệu quả nhiều chiều.

Để giải quyết các quy tắc này, chúng tôi có thể sử dụng các kỹ thuật như:

```python
# Pseudocode for addressing classification challenges

# 1. Overfitting: Use regularization and cross-validation
model = DecisionTreeClassifier(max_depth=3, min_samples_split=5)
scores = cross_val_score(model, X, y, cv=5)

# 2. Class Imbalance: Apply SMOTE (Synthetic Minority Over-sampling Technique)
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# 3. Feature Selection: Use SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 4. Scalability: Use algorithms that handle large datasets efficiently
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.partial_fit(X_batch, y_batch, classes=np.unique(y))
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về kỹ thuật phân loại và học máy:

1. Tài liệu scikit-learn: Hướng dẫn toàn diện về học máy bằng Python. [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
2. "Giới thiệu về thống kê học tập" của James, Witten, Hastie và Tibshirani: Nguồn tài liệu tuyệt vời để hiểu các phương pháp thống kê học tập. [https://www.statlearning.com/](https://www.statlearning.com/)
3. Cuộc thi Kaggle: Thực hành phân loại trên bộ dữ liệu trong thế giới thực. [https://www.kaggle.com/competitions](https://www.kaggle.com/competitions)
4. Bài viết về Machine Learning của ArXiv: Nghiên cứu mới nhất về phân loại và học máy. [https://arxiv.org/list/stat.ML/recent](https://arxiv.org/list/stat.ML/recent)

Những tài nguyên này cung cấp nhiều thông tin giúp bạn hiểu sâu hơn về các loại phân loại kỹ thuật và ứng dụng của chúng trong các lĩnh vực khác nhau.
