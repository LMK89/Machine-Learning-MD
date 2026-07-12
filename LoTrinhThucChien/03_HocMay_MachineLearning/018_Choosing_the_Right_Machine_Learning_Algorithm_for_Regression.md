##Chọn thuật toán học máy phù hợp cho hồi quy
Slide 1: Tổng quan về phân tích hồi phục

Phân tích phân tích tạo nền tảng của mô hình dự kiến, cho phép chúng tôi hiểu mối liên hệ giữa các biến thể và mức độ mong đợi được đưa ra. Chúng tôi sẽ khám phá việc phát triển nhiều kỹ thuật phục hồi bằng thư viện scikit-learn của Python, tập trung vào việc phát triển khai thực tế với các bộ dữ liệu trong thế giới thực.

```python
# Basic regression analysis setup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample dataset
np.random.seed(42)
X = np.random.randn(100, 3)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Mathematical representation of linear regression
'''
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$$
Where:
$$\beta_0$$ is the intercept
$$\beta_i$$ are the coefficients
$$\epsilon$$ is the error term
'''
```

Trang trình bày 2: Triển khai ngẫu nhiên giảm dần độ dốc

Giảm dần độ ngẫu nhiên ngẫu nhiên (SGD) là một kết quả hiệu ứng tối ưu hóa phương pháp để phù hợp với các tính năng tuyến tính phục hồi mô hình trên các dữ liệu lớn. Nó cập nhật các tham số mô hình đi lặp lại bằng cách sử dụng các ví dụ đào tạo riêng biệt, giúp tiết kiệm bộ nhớ và phù hợp với các vấn đề học tập trực tuyến.

```python
from sklearn.linear_model import SGDRegressor

# Initialize and train SGD regressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty='l2', eta0=0.01)
sgd_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_sgd = sgd_reg.predict(X_test_scaled)

# Evaluate performance
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

print(f"MSE: {mse_sgd:.4f}")
print(f"R2 Score: {r2_sgd:.4f}")
```

Trang trình bày 3: Triển khai thu hồi góc nhỏ nhất (LARS)

LARS cung cấp một phương pháp hiệu quả cao để tính toán toàn bộ đường Lasso cùng với chi phí tính toán như một phương pháp nhỏ nhất phù hợp. Nó đặc biệt hữu ích khi xử lý nhiều dữ liệu trong số lượng đối tượng vượt quá mức độ giám sát.

```python
from sklearn.linear_model import LarsCV
import matplotlib.pyplot as plt

# Initialize and train LARS with cross-validation
lars_cv = LarsCV(cv=5, max_iter=100)
lars_cv.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_lars = lars_cv.predict(X_test_scaled)

# Plot coefficients path
plt.figure(figsize=(10, 6))
plt.plot(lars_cv.coef_path_.T)
plt.xlabel('Step')
plt.ylabel('Coefficients')
plt.title('LARS Coefficient Path')
plt.show()

print(f"Best alpha: {lars_cv.alpha_}")
print(f"R2 Score: {r2_score(y_test, y_pred_lars):.4f}")
```

Trang trình bày 4: Triển khai Lasso và Elastic Net

Lasso và Elastic Net kết hợp chính xác hóa L1 và L2 để xử lý đa tuyến và thực hiện lựa chọn tính năng. Phương pháp này rất cần thiết cho các bộ dữ liệu nhiều chiều, trong đó việc lựa chọn tính năng và khả năng giải mô hình là rất quan trọng.

```python
from sklearn.linear_model import LassoCV, ElasticNetCV

# Initialize models with cross-validation
lasso_cv = LassoCV(cv=5, random_state=42)
elastic_cv = ElasticNetCV(cv=5, random_state=42)

# Train models
lasso_cv.fit(X_train_scaled, y_train)
elastic_cv.fit(X_train_scaled, y_train)

# Predictions
y_pred_lasso = lasso_cv.predict(X_test_scaled)
y_pred_elastic = elastic_cv.predict(X_test_scaled)

# Compare results
results = pd.DataFrame({
    'Method': ['Lasso', 'Elastic Net'],
    'R2 Score': [
        r2_score(y_test, y_pred_lasso),
        r2_score(y_test, y_pred_elastic)
    ],
    'Alpha': [lasso_cv.alpha_, elastic_cv.alpha_]
})
print(results)
```

Trang trình bày 5: Thực hiện phục hồi sườn núi

Hồi quy giải quyết hiện đa tuyến cộng đồng bằng cách bổ sung số phạt L2 vào hàm tiêu bình phương pháp nhỏ nhất thông thường. Kỹ thuật này giúp ngăn chặn việc trang bị quá mạnh và ổn định mô hình khi các yếu tố dự đoán có mối tương quan cao.

```python
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

# Initialize Ridge regression with cross-validation
alphas = np.logspace(-6, 6, 100)
ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

# Predictions
y_pred_ridge = ridge_cv.predict(X_test_scaled)

# Plot alpha vs MSE
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, ridge_cv.cv_values_.mean(axis=0) * -1)
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression: Alpha vs MSE')
plt.grid(True)
plt.show()

print(f"Best alpha: {ridge_cv.alpha_}")
print(f"R2 Score: {r2_score(y_test, y_pred_ridge):.4f}")
```

Trình bày 6: Hỗ trợ bộ thu hồi với nhân tuyến tính

SVR with nhân tuyến tính thực hiện khôi phục quy trình bằng cách sử dụng tính năng hỗ trợ tuyến tính, giúp giải quyết hiệu quả này đối với các vấn đề trong đó mối liên hệ giữa các đối tượng và mục tiêu gần như tuyến tính trong khi vẫn duy trì khả năng dự đoán mạnh mẽ.

```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Initialize Linear SVR
linear_svr = SVR(kernel='linear')

# Parameter grid for optimization
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.3]
}

# Grid search with cross-validation
grid_search = GridSearchCV(linear_svr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model predictions
y_pred_linear_svr = grid_search.predict(X_test_scaled)

print("Best parameters:", grid_search.best_params_)
print(f"R2 Score: {r2_score(y_test, y_pred_linear_svr):.4f}")
```

Slide 7: Hỗ trợ Vector Regressor với RBF Kernel

Biến RBF nhân không có một tính chất tuyến tính cụ thể, cho phép SVR thu thập các mẫu phức tạp trong dữ liệu. Việc phát triển này trình bày cách siêu tham số tối ưu hóa cho các tính năng phục hồi quy tuyến của các tác vụ.

```python
# Initialize RBF SVR
rbf_svr = SVR(kernel='rbf')

# Extended parameter grid for RBF kernel
param_grid_rbf = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1],
    'epsilon': [0.1, 0.2, 0.3]
}

# Grid search for RBF kernel
grid_search_rbf = GridSearchCV(rbf_svr, param_grid_rbf, cv=5, scoring='neg_mean_squared_error')
grid_search_rbf.fit(X_train_scaled, y_train)

# Predictions with best model
y_pred_rbf_svr = grid_search_rbf.predict(X_test_scaled)

# Compare performance metrics
print("Best parameters:", grid_search_rbf.best_params_)
print(f"R2 Score: {r2_score(y_test, y_pred_rbf_svr):.4f}")

# Mathematical representation of RBF kernel
'''
$$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
Where:
$$\gamma$$ is the kernel coefficient
$$||x - x'||^2$$ is the squared Euclidean distance
'''
```

Slide 8: Cây quyết định và tập hợp phương pháp

Cây định và phương pháp tập hợp quyết định kết hợp nhiều mô hình để tạo ra các yếu tố dự phòng mạnh mẽ. Triển khai giới thiệu Rừng ngẫu nhiên và Tăng cường độ dốc, hai kỹ thuật tổng hợp mạnh mẽ cho các nhiệm vụ phục hồi quy mô.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Initialize models
dt_reg = DecisionTreeRegressor(random_state=42)
rf_reg = RandomForestRegressor(random_state=42)
gb_reg = GradientBoostingRegressor(random_state=42)

# Train models
models = {
    'Decision Tree': dt_reg,
    'Random Forest': rf_reg,
    'Gradient Boosting': gb_reg
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'R2 Score': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred)
    }

# Display results
results_df = pd.DataFrame(results).T
print(results_df)
```

Trang trình bày 9: Thực hiện phương pháp tối thiểu thông tin

Bình phương pháp tối thiểu thông thường (OLS) cung cấp nền tảng cho tính toán tuyến tính bằng cách giảm thiểu tổng phương pháp dư. Việc phát triển này bao gồm các công cụ mong đợi và kiểm tra thống kê để đánh giá các giá trị giả định của mô hình và chất lượng phù hợp.

```python
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

# Implement OLS using both scikit-learn and statsmodels
# Scikit-learn implementation
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_ols = lr.predict(X_test_scaled)

# Statsmodels implementation for detailed statistics
X_train_sm = sm.add_constant(X_train_scaled)
model = sm.OLS(y_train, X_train_sm)
results = model.fit()

# Calculate residuals and perform diagnostic tests
residuals = y_test - y_pred_ols
residuals_standardized = (residuals - residuals.mean()) / residuals.std()

# Diagnostic plots and tests
normality_test = stats.normaltest(residuals_standardized)
print(results.summary())
print(f"\nNormality test p-value: {normality_test.pvalue:.4f}")
```

Trình bày 10: Phân loại tuyến hỗ trợ thuộc tính

Hỗ trợ phát triển tuyến tính SVC khai báo loại hỗ trợ bằng cách sử dụng tính năng tuyến nhân hạt, cung cấp khả năng phân loại kết quả hiệu quả cho dữ liệu có thể phân tích tuyến tính với khả năng chuẩn hóa và hợp lý trang web tối ưu.

```python
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Generate classification dataset
np.random.seed(42)
X_class = np.random.randn(200, 2)
y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)

# Split and scale data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class)
scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# Train Linear SVC
linear_svc = LinearSVC(dual=False, random_state=42)
linear_svc.fit(X_train_c_scaled, y_train_c)

# Predictions and evaluation
y_pred_svc = linear_svc.predict(X_test_c_scaled)

print("Classification Report:")
print(classification_report(y_test_c, y_pred_svc))

# Mathematical representation
'''
$$\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \max(0, 1 - y_i(w^Tx_i + b))$$
Where:
$$w$$ is the normal vector to the hyperplane
$$b$$ is the bias term
$$C$$ is the penalty parameter
'''
```

Trang trình bày 11: Triển khai Naive Bayes

Các bộ phân loại Naive Bayes thực hiện định lý Bayes với các giả định độc lập mạnh mẽ giữa các đặc điểm. Việc phát triển này hiển thị các biến Gaussian, Multinomial và Bernoulli cho các phân phối dữ liệu khác nhau.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler

# Initialize different Naive Bayes classifiers
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# For MultinomialNB and BernoulliNB, we need non-negative features
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train_c)
X_test_minmax = minmax_scaler.transform(X_test_c)

# Train and evaluate each classifier
classifiers = {
    'Gaussian NB': (gnb, X_train_c_scaled),
    'Multinomial NB': (mnb, X_train_minmax),
    'Bernoulli NB': (bnb, X_train_minmax)
}

results = {}
for name, (clf, X_train_transformed) in classifiers.items():
    clf.fit(X_train_transformed, y_train_c)
    y_pred = clf.predict(X_test_minmax if name != 'Gaussian NB' else X_test_c_scaled)
    results[name] = classification_report(y_test_c, y_pred, output_dict=True)

print(pd.DataFrame(results).round(3))
```

Trang trình bày 12: Bộ phân loại hàng xóm gần nhất K

K-Nearest Neighbors là một công cụ phân loại phi tham số linh hoạt, đưa ra các kỳ vọng dựa trên lớp đa số của k mẫu đào tạo gần nhất. Việc phát triển này bao gồm các khoảng cách vật liệu tối ưu hóa và các sơ đồ kỹ thuật số cận cảnh.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Initialize arrays for storing cross-validation scores
k_range = range(1, 31)
cv_scores = []
cv_std = []

# Evaluate different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_c_scaled, y_train_c, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    cv_std.append(scores.std())

# Find optimal k
optimal_k = k_range[np.argmax(cv_scores)]
best_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
best_knn.fit(X_train_c_scaled, y_train_c)

# Mathematical representation
'''
$$d(x, x') = \sqrt{\sum_{i=1}^n (x_i - x'_i)^2}$$
For weighted voting:
$$w_i = \frac{1}{d(x, x_i)^2}$$
'''

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.errorbar(k_range, cv_scores, yerr=cv_std, capsize=5)
plt.xlabel('k value')
plt.ylabel('Cross-validation accuracy')
plt.title('KNN: k vs Classification Accuracy')
print(f"Optimal k: {optimal_k}")
print(f"Best cross-validation score: {max(cv_scores):.4f}")
```

Trang trình bày 13: SVC với phát triển hạt nhân RBF

Phân loại hỗ trợ RBF nhân cho phép các ranh giới quyết định tuyến tính thông qua chuyển đổi không ẩn biểu tượng gian lận. Việc phát triển tập trung vào hạt nhân tham số tối ưu hóa và trực quan hóa ranh giới.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize SVC with RBF kernel
svc_rbf = SVC(kernel='rbf', probability=True)

# Parameter grid for optimization
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
}

# Grid search with cross-validation
grid_search = GridSearchCV(svc_rbf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_c_scaled, y_train_c)

# Best model evaluation
best_svc = grid_search.best_estimator_
y_pred_rbf = best_svc.predict(X_test_c_scaled)
y_prob_rbf = best_svc.predict_proba(X_test_c_scaled)

# Calculate and plot decision boundary
def plot_decision_boundary(model, X, y):
    h = 0.02  # step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('RBF SVC Decision Boundary')

plot_decision_boundary(best_svc, X_test_c_scaled, y_test_c)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Accuracy score: {grid_search.best_score_:.4f}")
```

Trang trình bày 14: Tài nguyên bổ sung

* " Hướng dẫn sử dụng máy hỗ trợ để nhận dạng mẫu" - [https://www.research.microsoft.com/pubs/67119/svmtutorial.pdf](https://www.research.microsoft.com/pubs/67119/svmtutorial.pdf)
* "Những khu rừng ngẫu nhiên" của Leo Breiman - [https://link.springer.com/article/10.1023/A:1010933404324](https://link.springer.com/article/10.1023/A:1010933404324)
* "Máy tăng cường độ dốc: Hướng dẫn" - [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
* "Giới thiệu về Học Thống kê" - [https://www.statlearning.com/](https://www.statlearning.com/)
* "Nhận dạng mẫu và máy học" - [https://www.springer.com/gp/book/9780387310732](https://www.springer.com/gp/book/9780387310732)
