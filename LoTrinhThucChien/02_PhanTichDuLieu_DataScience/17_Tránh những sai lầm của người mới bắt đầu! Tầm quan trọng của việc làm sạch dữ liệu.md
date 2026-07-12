## Tránh những sai lầm của người mới bắt đầu! Tầm quan trọng của việc làm sạch dữ liệu

Trang trình bày 1: Tầm quan trọng của việc làm sạch dữ liệu

Làm sạch dữ liệu là một bước quan trọng trong quy trình khoa học dữ liệu, thường bị người mới bắt đầu bỏ qua. Nó liên quan đến việc xử lý các giá trị bị thiếu, loại bỏ các giá trị trùng lặp và giải quyết sự không nhất quán. Hãy cùng khám phá một ví dụ đơn giản về làm sạch dữ liệu bằng gấu trúc:

```python
import numpy as np

# Create a sample dataset with issues
data = {
    'name': ['John', 'Jane', 'Mike', 'John', np.nan],
    'age': [25, 30, np.nan, 25, 40],
    'salary': [50000, 60000, 55000, 50000, 70000]
}
df = pd.DataFrame(data)

print("Original dataset:")
print(df)

# Clean the data
df_cleaned = df.dropna()  # Remove rows with missing values
df_cleaned = df_cleaned.drop_duplicates()  # Remove duplicate rows

print("\nCleaned dataset:")
print(df_cleaned)
```

Mã này trình bày các kỹ thuật làm sạch dữ liệu cơ bản như loại bỏ các giá trị bị thiếu và trùng lặp.

Trang trình bày 2: Tránh trang bị quá mức

Quá khớp xảy ra khi một mô hình học dữ liệu huấn luyện quá tốt, bao gồm cả tiếng ồn và biến động của nó. Điều này dẫn đến việc khái quát hóa kém về dữ liệu chưa được nhìn thấy. Hãy minh họa việc trang bị quá mức bằng ví dụ hồi quy đa thức:

```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the models
degrees = [1, 3, 15]
plt.figure(figsize=(14, 4))

for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_poly, y_train)

    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    y_plot = model.predict(poly_features.transform(X_plot))

    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, color='r', s=10, label='Training data')
    plt.plot(X_plot, y_plot, color='b', label='Model')
    plt.title(f'Degree {degree}')
    plt.legend()

plt.tight_layout()
plt.show()
```

Ví dụ này cho thấy việc tăng bậc đa thức có thể dẫn đến tình trạng trang bị quá mức như thế nào.

Trang trình bày 3: Giá trị của phân tích dữ liệu thăm dò (EDA)

EDA giúp khám phá các mẫu, mối quan hệ và sự bất thường trong dữ liệu. Đây là một bước quan trọng trước khi xây dựng mô hình. Hãy thực hiện một EDA đơn giản trên tập dữ liệu Iris:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Pairplot to visualize relationships between features
sns.pairplot(df, hue='species', height=2.5)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Iris Dataset')
plt.show()
```

Mã này tạo ra một biểu đồ cặp và một bản đồ nhiệt tương quan, tiết lộ mối quan hệ giữa các đặc điểm và loài.

Slide 4: Xác thực mô hình phù hợp

Việc xác thực là rất quan trọng để đánh giá hiệu suất của mô hình trên dữ liệu chưa được nhìn thấy. Xác thực chéo là một kỹ thuật mạnh mẽ cho mục đích này. Hãy triển khai xác thực chéo k-Fold:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV score:", cv_scores.std())
```

Ví dụ này trình bày cách sử dụng xác thực chéo để có được ước tính mạnh mẽ hơn về hiệu suất mô hình.

Trang trình bày 5: Ngoài độ chính xác: Đánh giá mô hình toàn diện

Mặc dù độ chính xác rất quan trọng nhưng không phải lúc nào nó cũng là thước đo tốt nhất, đặc biệt đối với các tập dữ liệu mất cân bằng. Hãy khám phá các số liệu khác bằng ví dụ phân loại nhị phân:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"AUC-ROC: {auc_roc:.3f}")
```

Mã này tính toán các số liệu khác nhau để đưa ra đánh giá toàn diện hơn về hiệu suất của mô hình.

Trang trình bày 6: Bắt đầu đơn giản: Sức mạnh của các mô hình cơ bản

Trong khi các mô hình phức tạp có tác dụng mạnh mẽ thì các mô hình đơn giản hơn thường hoạt động tốt và dễ diễn giải hơn. Hãy so sánh hồi quy tuyến tính đơn giản với hồi quy đa thức phức tạp hơn:

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Fit linear regression
lr = LinearRegression()
lr.fit(X, y)

# Fit polynomial regression
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
pr = LinearRegression()
pr.fit(X_poly, y)

# Make predictions
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_lr = lr.predict(X_test)
y_pr = pr.predict(poly.transform(X_test))

# Calculate MSE
mse_lr = mean_squared_error(y, lr.predict(X))
mse_pr = mean_squared_error(y, pr.predict(X_poly))

# Plot results
plt.scatter(X, y, color='r', label='Data')
plt.plot(X_test, y_lr, color='b', label='Linear Regression')
plt.plot(X_test, y_pr, color='g', label='Polynomial Regression')
plt.legend()
plt.title('Linear vs Polynomial Regression')
plt.show()

print(f"MSE (Linear): {mse_lr:.4f}")
print(f"MSE (Polynomial): {mse_pr:.4f}")
```

Ví dụ này so sánh hồi quy tuyến tính đơn giản với hồi quy đa thức phức tạp hơn, cho thấy rằng đôi khi các mô hình đơn giản hơn có thể hoạt động tốt.

Trang trình bày 7: Ví dụ thực tế: Dự đoán giá nhà

Hãy áp dụng những gì chúng ta đã học vào tình huống thực tế: dự đoán giá nhà. Chúng tôi sẽ sử dụng phiên bản đơn giản của bộ dữ liệu Nhà ở Boston:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# Load the Boston Housing dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Select a few features for simplicity
features = ['RM', 'LSTAT', 'PTRATIO']
X = df[features]
y = df['PRICE']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
```

Ví dụ này trình bày cách xây dựng và đánh giá mô hình dự đoán giá nhà đơn giản bằng cách sử dụng dữ liệu thực tế.

Trang trình bày 8: Ví dụ thực tế: Dự đoán tỷ lệ rời bỏ khách hàng

Hãy cùng khám phá một tình huống thực tế khác: dự đoán tỷ lệ rời bỏ khách hàng đối với một công ty viễn thông. Chúng tôi sẽ sử dụng tập dữ liệu đơn giản hóa và tập trung vào việc làm sạch dữ liệu, phân tích dữ liệu khám phá và xây dựng mô hình:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'tenure': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 100, n_samples),
    'total_charges': np.random.uniform(100, 5000, n_samples),
    'contract_type': np.random.choice(['Monthly', 'One year', 'Two year'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Data cleaning
df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
df.dropna(inplace=True)

# Exploratory Data Analysis
plt.figure(figsize=(12, 5))
plt.subplot(121)
sns.boxplot(x='contract_type', y='monthly_charges', data=df)
plt.title('Monthly Charges by Contract Type')

plt.subplot(122)
sns.histplot(data=df, x='tenure', hue='churn', multiple='stack', bins=20)
plt.title('Tenure Distribution by Churn Status')
plt.show()

# Prepare data for modeling
X = pd.get_dummies(df.drop('churn', axis=1), drop_first=True)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Ví dụ này bao gồm việc làm sạch dữ liệu, phân tích dữ liệu khám phá và xây dựng mô hình dự đoán tỷ lệ rời bỏ bằng cách sử dụng bộ phân loại Rừng ngẫu nhiên.

Slide 9: Xử lý bộ dữ liệu không cân bằng

Bộ dữ liệu mất cân bằng thường xảy ra trong các tình huống thực tế, chẳng hạn như phát hiện gian lận hoặc chẩn đoán bệnh hiếm gặp. Hãy cùng khám phá các kỹ thuật xử lý dữ liệu mất cân bằng:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Generate an imbalanced dataset
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.95, 0.05], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define resampling strategies
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.5)

# Create a pipeline with SMOTE, undersampling, and Random Forest
pipeline = Pipeline([
    ('over', over),
    ('under', under),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
```

Ví dụ này trình bày cách sử dụng SMOTE (Kỹ thuật lấy mẫu quá mức thiểu số tổng hợp) và Lấy mẫu ngẫu nhiên để cân bằng tập dữ liệu trước khi đào tạo bộ phân loại Rừng ngẫu nhiên.

Slide 10: Kỹ thuật tính năng và lựa chọn

Kỹ thuật và lựa chọn tính năng là những bước quan trọng trong việc cải thiện hiệu suất mô hình. Hãy cùng khám phá một số kỹ thuật:

```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Engineering: Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Feature Selection: Select top k features
selector = SelectKBest(f_regression, k=10)
X_train_selected = selector.fit_transform(X_train_poly, y_train)
X_test_selected = selector.transform(X_test_poly)

# Train a model with selected features
model = LinearRegression()
model.fit(X_train_selected, y_train)

# Make predictions
y_pred = model.predict(X_test_selected)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
```

Ví dụ này thể hiện kỹ thuật tính năng đa thức và lựa chọn tính năng bằng cách sử dụng SelectKBest.

Slide 11: Xử lý dữ liệu bị thiếu

Thiếu dữ liệu là một vấn đề phổ biến trong các bộ dữ liệu trong thế giới thực. Hãy khám phá các kỹ thuật để xử lý các giá trị bị thiếu:

```python
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create a sample dataset with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, 7, 8, np.nan],
    'C': [9, 10, 11, np.nan, 13]
}
df = pd.DataFrame(data)

print("Original dataset:")
print(df)

# Simple Imputation (mean strategy)
imp_mean = SimpleImputer(strategy='mean')
df_mean_imputed = pd.DataFrame(imp_mean.fit_transform(df), columns=df.columns)

print("\nMean Imputation:")
print(df_mean_imputed)

# KNN Imputation
imp_knn = KNNImputer(n_neighbors=2)
df_knn_imputed = pd.DataFrame(imp_knn.fit_transform(df), columns=df.columns)

print("\nKNN Imputation:")
print(df_knn_imputed)

# Multiple Imputation by Chained Equations (MICE)
imp_mice = IterativeImputer(random_state=0)
df_mice_imputed = pd.DataFrame(imp_mice.fit_transform(df), columns=df.columns)

print("\nMICE Imputation:")
print(df_mice_imputed)
```

Ví dụ này thể hiện ba kỹ thuật quy định khác nhau: quy định trung bình, quy định K-Láng giềng gần nhất và Quy mô nhiều lần theo phương trình chuỗi (MICE).

Slide 12: Khả năng diễn giải mô hình

Khi các mô hình trở nên phức tạp hơn, khả năng diễn giải trở nên quan trọng. Hãy cùng khám phá một số kỹ thuật để diễn giải các mô hình học máy:

```python
from sklearn.inspection import partial_dependence, plot_partial_dependence
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target
feature_names = boston.feature_names

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Calculate feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Compute and plot partial dependence for the two most important features
fig, ax = plt.subplots(figsize=(10, 6))
plot_partial_dependence(rf_model, X, [indices[0], indices[1]],
                        feature_names=feature_names, ax=ax)
plt.tight_layout()
plt.show()
```

Ví dụ này trình bày cách tính toán và trực quan hóa tầm quan trọng của đối tượng và biểu đồ phụ thuộc một phần cho mô hình Rừng ngẫu nhiên.

Slide 13: Điều chỉnh siêu tham số

Tối ưu hóa siêu tham số của mô hình là rất quan trọng để đạt được hiệu suất tốt nhất. Hãy cùng khám phá tìm kiếm dạng lưới và tìm kiếm ngẫu nhiên để điều chỉnh siêu tham số:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters (Grid Search):", grid_search.best_params_)
print("Best score (Grid Search):", grid_search.best_score_)

# Random Search
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=20, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print("\nBest parameters (Random Search):", random_search.best_params_)
print("Best score (Random Search):", random_search.best_score_)
```

Ví dụ này trình bày cách sử dụng GridSearchCV và RandomizedSearchCV để điều chỉnh siêu tham số của bộ phân loại Rừng ngẫu nhiên.

Trang trình bày 14: Tài nguyên bổ sung

Để tìm hiểu và khám phá thêm về khoa học dữ liệu và học máy, hãy xem xét các tài nguyên sau:

1. ArXiv.org: Kho lưu trữ các bài báo khoa học, bao gồm nhiều bài về học máy và khoa học dữ liệu. URL: [https://arxiv.org/list/stat.ML/recent](https://arxiv.org/list/stat.ML/recent)
2. Tài liệu Scikit-learn: Hướng dẫn toàn diện về thư viện Scikit-learn. URL: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
3. Hướng tới khoa học dữ liệu: Một ấn phẩm trung bình bao gồm các bài viết về các chủ đề khoa học dữ liệu khác nhau. URL: [https://towardsdatascience.com/](https://towardsdatascience.com/)
4. Kaggle: Một nền tảng dành cho các cuộc thi và bộ dữ liệu về khoa học dữ liệu. URL: [https://www.kaggle.com/](https://www.kaggle.com/)
5. Machine Learning Mastery: Một blog có các hướng dẫn thực tế về machine learning. URL: [https://machinelearningmastery.com/](https://machinelearningmastery.com/)

Những tài nguyên này cung cấp nhiều thông tin cho người mới bắt đầu và những người thực hành ở trình độ trung cấp về khoa học dữ liệu và học máy.
