## Tương quan, hồi quy và khớp đường cong trong Machine Learning with Python

Slide 1: Giới thiệu về Tương quan

Giới thiệu về tương quan

Mối tương quan đo cường độ và hướng của mối quan hệ giữa các biến thể. Đó là một khái niệm cơ bản trong thống kê và máy học, đặc biệt hữu ích trong phân tích dữ liệu khám phá và lựa chọn tính năng.

Mã số:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate correlated data
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5

# Calculate correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]

# Plot the data
plt.scatter(x, y)
plt.title(f"Correlation: {correlation:.2f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 2: Các loại tương quan

Các loại tương quan

Có ba loại tương quan chính: tích cực, tiêu cực và không tương quan. Tương quan dương có nghĩa là khi một biến tăng thì biến kia có xu hướng tăng. Tương quan âm nghĩa có nghĩa là khi một biến tăng thì biến kia có xu hướng giảm. Không có mối quan hệ nào có nghĩa là không có mối quan hệ rõ ràng giữa các biến.

Mã số:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data for different types of correlation
x = np.linspace(0, 10, 100)
y_positive = x + np.random.randn(100)
y_negative = -x + np.random.randn(100)
y_no_corr = np.random.randn(100)

# Plot the data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(x, y_positive)
ax1.set_title("Positive Correlation")

ax2.scatter(x, y_negative)
ax2.set_title("Negative Correlation")

ax3.scatter(x, y_no_corr)
ax3.set_title("No Correlation")

plt.tight_layout()
plt.show()
```

Trang trình bày 3: Hệ số tương quan Pearson

Tương thích Pearson System

Hệ thống tương số Pearson là thước đo tương quan phổ biến nhất. Nó dao động từ -1 (tương quan âm hoàn hảo) đến 1 (tương quan dương hoàn hảo), với 0 biểu thị không có tương quan tuyến tính tính.

Mã số:

```python
import numpy as np
from scipy import stats

# Generate data
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5

# Calculate Pearson correlation coefficient
pearson_corr, _ = stats.pearsonr(x, y)

print(f"Pearson correlation coefficient: {pearson_corr:.2f}")
```

Trang trình bày 4: Spearman cấp bậc tương tự

Spearman tự xếp hạng tương tự

Tương quan xếp hạng Spearman đánh giá mối quan hệ quan hệ đơn phương giữa các biến thể. Nó hữu ích khi kết nối giữa các biến không nhất thiết phải tuyến tính mà kèm theo một chế độ đơn năng chức năng.

Mã số:

```python
import numpy as np
from scipy import stats

# Generate non-linear but monotonic data
x = np.random.rand(100)
y = np.exp(x) + np.random.randn(100)*0.1

# Calculate Spearman rank correlation
spearman_corr, _ = stats.spearmanr(x, y)

print(f"Spearman rank correlation: {spearman_corr:.2f}")
```

Slide 5: Ma trận tương tự

Ma trận tương thích

Ma trận tương thích hiển thị các hệ tương quan giữa nhiều biến. Nó đặc biệt hữu ích trong việc phân tích đa biến và lựa chọn tính năng cho các mô hình học.

Mã số:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate multivariate data
data = np.random.randn(100, 4)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])

# Calculate correlation matrix
corr_matrix = df.corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

Slide 6: Giới thiệu về quá trình phục hồi

Giới thiệu về quá trình phục hồi

Phân tích được phục hồi là một thống kê phương pháp được sử dụng để mô hình hóa mối quan hệ giữa một biến phụ thuộc và một hoặc nhiều biến độc lập. Nó được sử dụng rộng rãi trong các mô hình dự đoán và máy học.

Mã số:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = 2*X + 1 + np.random.randn(5, 1)*0.5

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Plot data and regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 7: Hồi quy tuyến tính đơn giản

Hồi quy tuyến tính đơn giản

Hồi quy tuyến tính đơn giản hoá hóa mối quan hệ giữa các biến bằng phương pháp tuyến tính. Đây là hình thức phục hồi đơn giản nhất và đóng vai trò là nền tảng cho các kỹ thuật phục hồi phức tạp hơn.

Mã số:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = 2*X + 1 + np.random.randn(5, 1)*0.5

# Fit model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")
```

Slide 8: Hồi quy tuyến tính bội

Hồi quy tính bội tuyến

Hồi quy tuyến tính bội mở rộng hồi quy tuyến tính đơn giản để bao gồm nhiều biến độc lập. Nó hữu ích khi cố gắng hy vọng một biến phụ thuộc dựa trên nhiều yếu tố.

Mã số:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate data
X = np.random.rand(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1

# Fit model
model = LinearRegression()
model.fit(X, y)

# Print coefficients
for i, coef in enumerate(model.coef_):
    print(f"Coefficient for X{i+1}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
```

Slide 9: Hồi quy đa thức

Hồi phục đa thức

Hồi quy đa thức được sử dụng khi kết nối quan hệ giữa các biến là phi tuyến tính. Nó phù hợp với một phương thức đa phương thức cho dữ liệu, cho phép mô hình hóa các mối liên hệ phức tạp hơn.

Mã số:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate non-linear data
X = np.linspace(0, 5, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 0.5

# Create and fit the polynomial regression model
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X, y)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Trang trình bày 10: Hồi quy logistic

Hồi quy logistic

Hồi quy logistic được sử dụng cho vấn đề phân loại phân loại. Mặc dù tên của nó là như vậy nhưng nó là một loại phân tích thuật toán chứ không phải thu hồi thuật toán. Nó được mong đợi về hiệu suất của một công cụ thuộc về một lớp.

Mã số:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate binary classification data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 11: Giới thiệu về Đường cong

Giới thiệu về đường cong

So khớp đường dẫn được xây dựng một đường cong hoặc hàm học thuật phù hợp nhất với một tập dữ liệu. Nó được sử dụng trong nhiều lĩnh vực khác nhau, bao gồm cả học máy, để mô hình hóa các mối liên hệ phức tạp.

Mã số:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the function to fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate noisy data
x = np.linspace(0, 4, 50)
y = func(x, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x))

# Fit the function
popt, _ = curve_fit(func, x, y)

# Plot the results
plt.scatter(x, y, label='data')
plt.plot(x, func(x, *popt), 'r-', label='fit')
plt.legend()
plt.show()

print(f"Optimal parameters: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}")
```

Trang trình bày 12: Khối bình phương nhỏ nhất phi tuyến tính

Phương pháp tối thiểu tính toán

Matching phương pháp nhỏ nhất phi tuyến tính tính là một đường cong phù hợp trong đó hàm không bắt buộc phải tuyến tính trong các tham số. Nó được sử dụng khi kết nối giữa các biến được biết là phi tuyến tính.

Mã số:

```python
import numpy as np
from scipy.optimize import least_squares

# Define the model function
def model(x, params):
    a, b, c = params
    return a * np.exp(-b * x) + c

# Define the residual function
def residual(params, x, y):
    return model(x, params) - y

# Generate synthetic data
x = np.linspace(0, 10, 100)
true_params = [2.5, 0.5, 1.0]
y_true = model(x, true_params)
y = y_true + 0.1 * np.random.randn(len(x))

# Perform the fit
initial_guess = [1.0, 1.0, 0.0]
result = least_squares(residual, initial_guess, args=(x, y))

print("Fitted parameters:", result.x)
```

Slide 13: Ví dụ thực tế: Dự đoán giá nhà ở

Ví dụ thực tế: Dự đoán giá nhà ở

Vui lòng sử dụng tính năng phục hồi tuyến tính để dự đoán giá đất dựa trên nhiều đặc điểm khác nhau như quy mô, số phòng ngủ và vị trí.

Mã số:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data (assuming we have a CSV file with housing data)
data = pd.read_csv('housing_data.csv')

# Prepare the features and target
X = data[['size', 'bedrooms', 'location']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")

# Example prediction
new_house = [[2000, 3, 1]]  # size: 2000 sq ft, 3 bedrooms, location code: 1
predicted_price = model.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:,.2f}")
```

Trang trình bày 14: Ví dụ thực tế: Tỷ lệ dự kiến ​​khi bỏ hàng

Ví dụ thực tế: Tỷ lệ được mong đợi khi bỏ hàng

Vui lòng sử dụng phương pháp hồi phục logistic để dự đoán tỷ lệ loại bỏ khách hàng dựa trên các tính năng như khả năng sử dụng, cuộc gọi dịch vụ khách hàng và thời hạn hợp lý.

Mã số:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the data (assuming we have a CSV file with customer data)
data = pd.read_csv('customer_data.csv')

# Prepare the features and target
X = data[['usage', 'customer_service_calls', 'contract_length']]
y = data['churned']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example prediction
new_customer = [[100, 2, 12]]  # usage: 100, customer service calls: 2, contract length: 12 months
churn_probability = model.predict_proba(new_customer)[0][1]
print(f"Churn probability: {churn_probability:.2f}")
```

Trang trình bày 15: Tài nguyên bổ sung

Tài nguyên bổ sung

Để nghiên cứu sâu hơn về mối tương quan, phục hồi và kết hợp đường cong trong máy học, hãy cân nhắc khám phá các tài nguyên sau:

1. "Giới thiệu về thống kê tập học" của Gareth James và cộng đồng. (Có trên ArXiv: [https://arxiv.org/abs/1501.07274](https://arxiv.org/abs/1501.07274))
2. “Các yếu tố của việc học thống kê” của Trevor Hastie et al. (Có trên ArXiv: [https://arxiv.org/abs/2001.00323](https://arxiv.org/abs/2001.00323))
3. Tài liệu về Scikit-learn: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
4. Tài liệu SciPy: [https://docs.scipy.org/doc/scipy/reference/](https://docs.scipy.org/doc/scipy/reference/)

Tài nguyên này cung cấp thông tin toàn diện về các chủ đề được thảo luận trong bài trình bày này và có thể giúp bạn hiểu sâu hơn về các khái niệm máy học cơ bản này.
