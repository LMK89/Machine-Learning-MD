## Chế độ của phương R trong Đánh giá mô hình hồi phục:
Slide 1: Giới thiệu về R bình phương (R²)

Hiểu R bình phương (R²) trong Phân tích hồi quy

R-squared, còn được gọi là hệ số xác định, là thống kê đo lường được sử dụng để đánh giá mức độ phù hợp của mô hình định nghĩa. Nó có thể đưa ra sai số tỷ lệ của phương pháp trong các biến phụ thuộc có thể được dự đoán từ (các) biến độc lập. Mặc dù R bình luận được sử dụng rộng rãi nhưng không có chế độ giới hạn nào có thể dẫn đến hiểu sai về hiệu suất của mô hình.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate R-squared
r2 = r2_score(y, model.predict(X))
print(f"R-squared: {r2:.4f}")
```

Trang trình bày 2: Chế độ 1 - Độ nhạy đối với kích thước mẫu

Bình phương R và kích thước mẫu

R thảo luận có xu hướng tăng khi có nhiều biến hơn được bổ sung vào mô hình, ngay cả khi các biến này không cải thiện đáng kể khả năng được mong đợi của mô hình. Điều này có thể dẫn đến trạng thái trang quá trình độ, đặc biệt với kích thước nhỏ. Để chứng minh điều này, chúng tôi sẽ tạo ra một hàm tạo ngẫu nhiên dữ liệu và tính R bình phương cho các mẫu có kích thước khác nhau.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def r2_vs_sample_size(n_samples, n_features):
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    model = LinearRegression().fit(X, y)
    return r2_score(y, model.predict(X))

sample_sizes = [10, 50, 100, 500, 1000]
r2_values = [r2_vs_sample_size(n, 5) for n in sample_sizes]

for n, r2 in zip(sample_sizes, r2_values):
    print(f"Sample size: {n}, R-squared: {r2:.4f}")
```

Slide 3: Han mode 2 - Không nhạy cảm với thành kiến

R bình phương và mô phỏng trôi

R bình phương không tính đến hệ thống sai lệch trong các mô hình dự kiến. Một mô hình có thể có giá trị R bình ổn cao ngay cả khi các kỳ vọng của nó luôn lệch một khoảng lớn. Chế độ này nêu bật tầm quan trọng của việc xem xét các số liệu khác cùng với bình luận R khi đánh giá hiệu suất mô hình.

```python
import numpy as np
from sklearn.metrics import r2_score

# Generate sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Create biased predictions
y_pred_biased = y + 5  # Add a constant bias of 5

# Calculate R-squared for biased predictions
r2_biased = r2_score(y, y_pred_biased)
print(f"R-squared (biased): {r2_biased:.4f}")

# Calculate mean absolute error
mae = np.mean(np.abs(y - y_pred_biased))
print(f"Mean Absolute Error: {mae:.4f}")
```

Trang trình bày 4: Chế độ 3 - Thiếu thông tin về kiến ​​trúc chính xác

Bình phương R và độ chính xác được mong đợi

R-squared không cung cấp thông tin trực tiếp về tính chính xác của dự đoán. Bình phương R cao không có nghĩa là mô hình đưa ra chính xác được mong đợi. Để minh họa điều này, chúng tôi sẽ tạo ra một mô hình có R bình phương cao nhưng hiệu suất được mong đợi trên dữ liệu mới.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Generate training data
X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_train = np.array([2, 4, 6, 8, 10])

# Fit model
model = LinearRegression().fit(X_train, y_train)

# Calculate R-squared on training data
r2_train = r2_score(y_train, model.predict(X_train))
print(f"R-squared (train): {r2_train:.4f}")

# Generate test data
X_test = np.array([6, 7, 8, 9, 10]).reshape(-1, 1)
y_test = np.array([11, 9, 13, 15, 12])

# Calculate R-squared and MAE on test data
r2_test = r2_score(y_test, model.predict(X_test))
mae_test = mean_absolute_error(y_test, model.predict(X_test))
print(f"R-squared (test): {r2_test:.4f}")
print(f"MAE (test): {mae_test:.4f}")
```

Trang trình bày 5: Chế độ 4 - Độ nhạy cảm với các ngoại lệ

Bình phương R và ngoại lệ

R bình phương có thể bị ảnh hưởng nặng nề bởi các giá trị ngoại lệ trong dữ liệu. Một giá trị cực trị duy nhất có thể tác động đáng kể đến giá trị bình phương R, có khả năng dẫn đến đánh giá quá lạc quan hoặc bi quan về hiệu suất của mô hình. Hãy chứng minh điều này bằng cách so sánh các phương pháp giá trị R có và không có giá trị ngoại lệ.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Add an outlier
X_outlier = np.vstack([X, [11]])
y_outlier = np.append(y, [100])

# Fit models and calculate R-squared
model = LinearRegression()
model.fit(X, y)
r2_normal = r2_score(y, model.predict(X))

model_outlier = LinearRegression()
model_outlier.fit(X_outlier, y_outlier)
r2_with_outlier = r2_score(y_outlier, model_outlier.predict(X_outlier))

print(f"R-squared (normal): {r2_normal:.4f}")
print(f"R-squared (with outlier): {r2_with_outlier:.4f}")
```

Trình bày 6: Chế độ 5 - Không thể xác định quan hệ nhân vật

Bình phương R và nhân

Giá trị R bình luận cao không có ý nghĩa quan hệ nhân quả giữa các biến. Nó chỉ tìm thấy mối tương quan. Điều này rất quan trọng để hiểu khi diễn ra giải kết quả hồi quy, đặc biệt là trong các lĩnh vực như kinh tế hoặc khoa học xã hội. Hãy tạo một ví dụ trong đó hai biến không liên quan có giá trị R bình luận cao.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate two unrelated variables
np.random.seed(42)
X = np.random.rand(100, 1)
y = np.random.rand(100)

# Fit model and calculate R-squared
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

print(f"R-squared: {r2:.4f}")

# Calculate correlation coefficient
corr = np.corrcoef(X.flatten(), y)[0, 1]
print(f"Correlation coefficient: {corr:.4f}")
```

Slide 7: Phần 6 - Sự phụ thuộc vào mối quan hệ tuyến tính

Mối quan hệ R bình phương và phi tuyến tính

R-squared giả định mối quan hệ tuyến tính giữa các biến. Đối với các mối quan hệ phi tuyến tính, bình phương R có thể đánh giá thấp sức mạnh của mối quan hệ. Chế độ này nhấn mạnh tầm quan trọng của công việc trực tuyến hóa dữ liệu và xem xét các mô hình phi tuyến tính khi thích hợp. Vui lòng so sánh phương pháp R để chọn mô hình tuyến tính và mối quan hệ phi tuyến tính.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Generate non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).flatten() + np.random.normal(0, 0.1, 100)

# Fit linear model
linear_model = LinearRegression().fit(X, y)
r2_linear = r2_score(y, linear_model.predict(X))

# Calculate R-squared for non-linear relationship
r2_nonlinear = r2_score(y, np.sin(X).flatten())

print(f"R-squared (linear model): {r2_linear:.4f}")
print(f"R-squared (true non-linear relationship): {r2_nonlinear:.4f}")

# Plot data and models
plt.scatter(X, y, alpha=0.5)
plt.plot(X, linear_model.predict(X), color='red', label='Linear model')
plt.plot(X, np.sin(X), color='green', label='True relationship')
plt.legend()
plt.title("Linear vs Non-linear Relationship")
plt.show()
```

Slide 8: Mode 7 - Missing information về dư lượng

Phương pháp phân tích R và dư thừa

R-squared không cung cấp thông tin về Bố cục dư, điều này rất quan trọng để đánh giá các giả định của mô hình. Bình phương R cao không đảm bảo rằng phần dư thừa có phân phối chuẩn hoặc phương pháp sai không thay đổi. Hãy tạo một ví dụ trong đó R bình phương cao nhưng phần dư thừa hiện không nhất thiết phải thay đổi.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Generate heteroscedastic data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + np.random.normal(0, X.flatten(), 100)

# Fit model and calculate R-squared
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

# Calculate residuals
residuals = y - model.predict(X).flatten()

print(f"R-squared: {r2:.4f}")

# Plot residuals
plt.scatter(X, residuals)
plt.title("Residuals vs. Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()
```

Trình bày 9: Mode 8 - So sánh giữa các dữ liệu khác nhau

So sánh R bình luận và dữ liệu

Giá trị phương pháp R không thể so sánh trực tiếp giữa các dữ liệu hoặc các biến phụ thuộc khác nhau. Một mô hình có R bình phương thấp hơn có thể hoạt động tốt hơn trên dữ liệu mới nên mô hình có R bình phương cao hơn được đào tạo trên một dữ liệu khác. Chế độ này nhấn mạnh tầm quan trọng của công việc xem bối cảnh và mục tiêu cụ thể của phân tích.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Generate two datasets
np.random.seed(42)
X1 = np.random.rand(100, 1)
y1 = 2 * X1 + np.random.normal(0, 0.5, (100, 1))

X2 = np.random.rand(100, 1)
y2 = 5 * X2 + np.random.normal(0, 2, (100, 1))

# Fit models and calculate R-squared and MSE
model1 = LinearRegression().fit(X1, y1)
r2_1 = r2_score(y1, model1.predict(X1))
mse_1 = mean_squared_error(y1, model1.predict(X1))

model2 = LinearRegression().fit(X2, y2)
r2_2 = r2_score(y2, model2.predict(X2))
mse_2 = mean_squared_error(y2, model2.predict(X2))

print(f"Dataset 1 - R-squared: {r2_1:.4f}, MSE: {mse_1:.4f}")
print(f"Dataset 2 - R-squared: {r2_2:.4f}, MSE: {mse_2:.4f}")
```

Trang trình bày 10: Hàm chế 9 - Không nhạy cảm với tầm quan trọng của yếu tố dự đoán

Bình phương R và tầm quan trọng của tính năng

R bình phương không cung cấp thông tin về tầm quan trọng tương đối của từng yếu tố dự đoán. Bình phương R cao không biết biến nào có ảnh hưởng nhiều nhất trong mô hình. Để giải quyết vấn đề này, chúng tôi có thể sử dụng các kỹ thuật như biểu đồ tầm quan trọng của đặc điểm hoặc biểu đồ phụ thuộc của một phần. Hãy chứng minh điều này bằng cách sử dụng mô hình hồi quy bội đơn giản.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# Generate sample data
X = np.random.rand(100, 3)
y = 2*X[:, 0] + 0.5*X[:, 1] + 0.1*X[:, 2] + np.random.normal(0, 0.1, 100)

# Fit model and calculate R-squared
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

print(f"R-squared: {r2:.4f}")

# Calculate feature importance
perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)

for i, importance in enumerate(perm_importance.importances_mean):
    print(f"Feature {i+1} importance: {importance:.4f}")
```

Slide 11: Mode 10 - Giả định sai phương pháp

R-bình phương và tính đồng nhất

R bình định phương pháp tính toán tối đa (phương pháp không đổi) của dư phần. Khi giả định điều này là phạm vi, phương pháp R có thể không thể xác định chính xác mức độ phù hợp của mô hình. Hãy tạo một ví dụ trong đó R bình phương cao nhưng giả định về tính đồng nhất bị vi phạm.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Generate heteroscedastic data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + np.random.normal(0, 0.5 * X.flatten(), 100)

# Fit model and calculate R-squared
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

print(f"R-squared: {r2:.4f}")

# Plot residuals
residuals = y - model.predict(X).flatten()
plt.scatter(model.predict(X), residuals)
plt.title("Residuals vs. Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()
```

Slide 12: Phần chế độ 11 - Nhạy cảm với các ảnh hưởng

R bình phương và ảnh hưởng

Phương pháp có thể bị ảnh hưởng không tương thích với các ảnh hưởng, đó là những hoạt động quan trọng để phục hồi đường phục hồi. Những điểm này có thể dẫn đến sai lệch giá trị R bình phương. Hãy chứng minh điều này bằng cách so sánh phương pháp R có và không có ảnh hưởng.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Add an influential point
X_influential = np.vstack([X, [20]])
y_influential = np.append(y, [60])

# Fit models and calculate R-squared
model = LinearRegression().fit(X, y)
r2_normal = r2_score(y, model.predict(X))

model_influential = LinearRegression().fit(X_influential, y_influential)
r2_influential = r2_score(y_influential, model_influential.predict(X_influential))

print(f"R-squared (normal): {r2_normal:.4f}")
print(f"R-squared (with influential point): {r2_influential:.4f}")
```

Slide 13: Mode 12 - Thiếu thông tin về mô hình phức tạp

Bình phương R và phức tạp mô hình

R-squared không cung cấp thông tin về mô hình phức tạp. Một mô hình phức tạp hơn có thể có R bình phương cao hơn nhưng có thể làm quá khớp dữ liệu. Để giải quyết vấn đề này, chúng tôi có thể sử dụng R bình phương đã điều chỉnh, điều này sẽ loại bỏ việc bổ sung các yếu tố dự đoán không cần thiết. Hãy so sánh bình phương R và bình phương R đã điều chỉnh cho các mô hình có lượng yếu tố dự đoán khác nhau.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 5)
y = 2*X[:, 0] + 0.5*X[:, 1] + np.random.normal(0, 0.1, 100)

# Fit models with different numbers of predictors
r2_values = []
adj_r2_values = []

for i in range(1, 6):
    model = LinearRegression().fit(X[:, :i], y)
    r2 = r2_score(y, model.predict(X[:, :i]))
    adj_r2 = adjusted_r2(r2, len(y), i)

    r2_values.append(r2)
    adj_r2_values.append(adj_r2)

    print(f"Predictors: {i}, R-squared: {r2:.4f}, Adjusted R-squared: {adj_r2:.4f}")
```

Slide 14: Ví dụ 1 thực tế - Dự đoán giá nhà

Bình phương R trong dự đoán giá

Trong bất kỳ sản phẩm nào, R bình luận thường được sử dụng để đánh giá giá các nhà sản xuất được mong đợi. Tuy nhiên, chỉ dựa vào R có thể gây nhầm lẫn. Vui lòng tạo một mô hình mong đợi giá đơn giản và xem xét giới hạn của nó.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Generate synthetic house data
np.random.seed(42)
size = np.random.randint(1000, 5000, 1000)
age = np.random.randint(0, 50, 1000)
location = np.random.randint(1, 10, 1000)
price = 100000 + 100 * size - 2000 * age + 50000 * location + np.random.normal(0, 50000, 1000)

X = np.column_stack((size, age, location))
y = price

# Split data and fit model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# Calculate metrics
r2 = r2_score(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))

print(f"R-squared: {r2:.4f}")
print(f"Mean Absolute Error: ${mae:.2f}")
```

Slide 15: Ví dụ 2 thực tế - Dự đoán thị trường chứng khoán

Chế độ của phương thức R trong trường chứng khoán được mong đợi

Trong tài chính, R bình phương trùng lặp khi được sử dụng để đánh giá những kỳ vọng về thị trường chứng khoán. Tuy nhiên, những giới hạn của nó sẽ trở nên rõ ràng trong lĩnh vực đầy biến động này. Hãy tạo một mô hình dự đoán giá cổ phiếu đơn giản để minh họa tại sao chỉ R bình phương là chưa đủ.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

# Generate synthetic stock data
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2022-12-31')
price = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
volume = np.random.randint(1000000, 10000000, len(dates))

df = pd.DataFrame({'Date': dates, 'Price': price, 'Volume': volume})
df['PreviousPrice'] = df['Price'].shift(1)
df['PriceChange'] = df['Price'] - df['PreviousPrice']
df = df.dropna()

X = df[['PreviousPrice', 'Volume']]
y = df['Price']

# Fit model and calculate R-squared
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))

print(f"R-squared: {r2:.4f}")

# Calculate daily returns
df['DailyReturn'] = df['PriceChange'] / df['PreviousPrice']
print(f"Volatility (std of daily returns): {df['DailyReturn'].std():.4f}")
```

Trang trình bày 16: Tài nguyên bổ sung

Đọc thêm về giới hạn của phương R

Để hiểu sâu hơn về các giới hạn bình luận phương pháp R và các số liệu thay thế, hãy xem xét khám phá các tài nguyên sau:

1. "Sự nguy hiểm của bình R" của Frost, J. (2020) - Thảo luận toàn diện về những nguy hiểm của bình R.
2. "Beyond R-squared: Metrics New for Regression Models" của Kvålseth, T. O. (2015) - Khám phá các biện pháp thay thế mức độ phù hợp.
3. "Hệ số xác định R-Squared có nhiều thông tin hơn SMAPE, MAE, MAPE, MSE và RMSE trong đánh giá phân tích phục hồi" của Alexander, D. L. J., Tropsha, A., & Winkler, D. A. (2015) - ArXiv:1511.02513 \[stat.ML\]

Tài nguyên này cung cấp các phân tích chuyên sâu về giới hạn bình phương R và đề xuất các phương pháp tiếp cận khác để đánh giá mô hình trong các bối cảnh khác nhau.
