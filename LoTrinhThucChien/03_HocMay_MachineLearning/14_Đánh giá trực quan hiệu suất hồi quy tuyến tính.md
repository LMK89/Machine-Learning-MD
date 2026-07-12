## Đánh giá trực quan hiệu suất hồi quy tuyến tính
Trang trình bày 1: Đánh giá hiệu suất hồi quy tuyến tính với các ô phân phối dư

Hồi quy tuyến tính là một kỹ thuật thống kê cơ bản được sử dụng để mô hình hóa mối quan hệ giữa các biến. Mặc dù bản thân đường hồi quy cung cấp những hiểu biết có giá trị nhưng việc đánh giá hiệu suất của mô hình đòi hỏi phải có cái nhìn sâu hơn. Một công cụ mạnh mẽ nhưng bị đánh giá thấp cho mục đích này là biểu đồ phân phối phần dư.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)

# Perform linear regression
coeffs = np.polyfit(X, y, 1)
y_pred = np.polyval(coeffs, X)

# Calculate residuals
residuals = y - y_pred

# Plot residual distribution
plt.hist(residuals, bins=20, edgecolor='black')
plt.title('Residual Distribution Plot')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Tìm hiểu phần dư trong hồi quy tuyến tính

Phần dư là sự khác biệt giữa giá trị quan sát được và giá trị dự đoán trong mô hình hồi quy. Chúng đóng một vai trò quan trọng trong việc đánh giá mức độ phù hợp của mô hình với dữ liệu. Trong một kịch bản lý tưởng, phần dư phải được phân phối ngẫu nhiên xung quanh số 0, cho thấy rằng mô hình nắm bắt tốt mối quan hệ cơ bản.

```python
# Visualize residuals
plt.scatter(X, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Independent Variable')
plt.xlabel('X')
plt.ylabel('Residual')
plt.show()
```

Slide 3: The Importance of Normally Distributed Residuals

One key assumption of linear regression is that the residuals follow a normal distribution. This assumption is crucial because it underpins the validity of statistical inferences drawn from the model. A normal distribution of residuals suggests that the model's errors are random and not systematically biased.

```python
import scipy.stats as stats

# Q-Q plot to check normality
fig, ax = plt.subplots()
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q plot of residuals")
plt.show()
```

Trang trình bày 4: Đặc điểm của lô phân phối thặng dư tốt

Một mô hình hồi quy tuyến tính hoạt động tốt sẽ tạo ra một biểu đồ phân phối phần dư:

1. Tuân theo phân phối chuẩn, có dạng đối xứng và hình chuông.
2. Căn giữa quanh số 0, biểu thị những dự đoán không thiên vị.
3. Không hiển thị mô hình hoặc xu hướng rõ ràng khi được vẽ dựa trên các giá trị dự đoán hoặc các biến độc lập.

```python
# Residual plot against predicted values
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
```

Slide 5: Red Flags in Residual Distribution Plots

Certain patterns in residual plots can indicate issues with the model:

1.  Skewness: Asymmetry in the distribution suggests non-linearity or the presence of outliers.
2.  Heavy tails: Excess kurtosis may indicate the presence of outliers or heteroscedasticity.
3.  Multimodality: Multiple peaks in the distribution could suggest the need for additional predictors or non-linear terms.

```python
# Generate non-linear data
X_nl = np.linspace(0, 10, 100)
y_nl = 2 * X_nl**2 + 1 + np.random.normal(0, 5, 100)

# Fit linear model to non-linear data
coeffs_nl = np.polyfit(X_nl, y_nl, 1)
y_pred_nl = np.polyval(coeffs_nl, X_nl)
residuals_nl = y_nl - y_pred_nl

# Plot residual distribution for non-linear data
plt.hist(residuals_nl, bins=20, edgecolor='black')
plt.title('Residual Distribution Plot (Non-linear Data)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()
```

Trang trình bày 6: Phát hiện phương sai thay đổi

Tính không đồng nhất xảy ra khi độ biến thiên của phần dư không phải là hằng số ở tất cả các mức của các biến độc lập. Sự vi phạm này có thể dẫn đến sai số chuẩn và khoảng tin cậy không đáng tin cậy. Biểu đồ dư có thể giúp phát hiện vấn đề này bằng cách hiển thị hình quạt hoặc hình nón.

```python
# Generate heteroscedastic data
X_hetero = np.linspace(0, 10, 100)
y_hetero = 2 * X_hetero + np.random.normal(0, 0.5 * X_hetero, 100)

# Fit linear model
coeffs_hetero = np.polyfit(X_hetero, y_hetero, 1)
y_pred_hetero = np.polyval(coeffs_hetero, X_hetero)
residuals_hetero = y_hetero - y_pred_hetero

# Plot residuals
plt.scatter(X_hetero, residuals_hetero)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. X (Heteroscedastic)')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.show()
```

Slide 7: Xử lý sự bất thường: Sự biến đổi

Khi phần dư không có phân phối chuẩn, việc biến đổi các biến phụ thuộc hoặc độc lập đôi khi có thể hữu ích. Các phép biến đổi phổ biến bao gồm phép biến đổi logarit, căn bậc hai và Box-Cox. Những điều này có thể giúp tuyến tính hóa các mối quan hệ và ổn định sự khác biệt.

```python
# Log transformation example
y_log = np.log(y_nl)
coeffs_log = np.polyfit(X_nl, y_log, 1)
y_pred_log = np.polyval(coeffs_log, X_nl)
residuals_log = y_log - y_pred_log

# Plot transformed residuals
plt.hist(residuals_log, bins=20, edgecolor='black')
plt.title('Residual Distribution Plot (Log-transformed)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()
```

Trang trình bày 8: Các ô dư cho dữ liệu chiều cao

Trong các bộ dữ liệu nhiều chiều, việc trực quan hóa đường hồi quy trở nên khó khăn. Tuy nhiên, biểu đồ phân phối phần dư vẫn là một công cụ mạnh mẽ vì nó cô đọng hiệu suất của mô hình thành biểu diễn một chiều, bất kể số lượng yếu tố dự đoán.

```python
# Generate high-dimensional data
np.random.seed(42)
X_high_dim = np.random.rand(100, 5)  # 5 predictors
y_high_dim = np.sum(X_high_dim, axis=1) + np.random.normal(0, 0.5, 100)

# Fit linear model
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_high_dim, y_high_dim)
y_pred_high_dim = model.predict(X_high_dim)
residuals_high_dim = y_high_dim - y_pred_high_dim

# Plot residual distribution
plt.hist(residuals_high_dim, bins=20, edgecolor='black')
plt.title('Residual Distribution Plot (High-Dimensional Data)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()
```

Trang trình bày 9: Giải thích các ô dư: Một nghiên cứu điển hình

Chúng ta hãy xem xét một kịch bản trong thế giới thực trong đó các ô dư cho thấy những bất cập của mô hình. Hãy xem xét một nghiên cứu về mối quan hệ giữa dân số của một thành phố và tỷ lệ tội phạm ở đó. Hồi quy tuyến tính ban đầu có vẻ thỏa đáng, nhưng phân tích phần dư lại kể một câu chuyện khác.

```python
# Simulated city data
np.random.seed(42)
population = np.linspace(10000, 1000000, 100)
crime_rate = 0.05 * np.sqrt(population) + np.random.normal(0, 2, 100)

# Linear regression
coeffs = np.polyfit(population, crime_rate, 1)
crime_rate_pred = np.polyval(coeffs, population)
residuals = crime_rate - crime_rate_pred

# Residual plot
plt.scatter(population, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Population')
plt.xlabel('Population')
plt.ylabel('Residuals')
plt.show()
```

Trang trình chiếu 10: Diễn giải kết quả nghiên cứu điển hình

Biểu đồ còn lại từ ví dụ về tỷ lệ tội phạm trong thành phố của chúng tôi cho thấy một mô hình đường cong rõ ràng, cho thấy mối quan hệ giữa dân số và tỷ lệ tội phạm là không tuyến tính. Điều này cho thấy mô hình tuyến tính ban đầu của chúng tôi không đầy đủ và không nắm bắt được mối quan hệ thực sự giữa các biến.

```python
# Histogram of residuals
plt.hist(residuals, bins=20, edgecolor='black')
plt.title('Residual Distribution (Crime Rate Model)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# Q-Q plot
fig, ax = plt.subplots()
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q plot of residuals (Crime Rate Model)")
plt.show()
```

Trang trình bày 11: Cải thiện mô hình dựa trên phân tích phần dư

Dựa trên phân tích phần dư, chúng tôi có thể cải thiện mô hình của mình bằng cách xem xét mối quan hệ phi tuyến tính. Trong trường hợp này, phép biến đổi căn bậc hai của tổng thể có thể phù hợp.

```python
# Improved model with square root transformation
population_sqrt = np.sqrt(population)
coeffs_improved = np.polyfit(population_sqrt, crime_rate, 1)
crime_rate_pred_improved = np.polyval(coeffs_improved, population_sqrt)
residuals_improved = crime_rate - crime_rate_pred_improved

# Residual plot for improved model
plt.scatter(population, residuals_improved)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Population (Improved Model)')
plt.xlabel('Population')
plt.ylabel('Residuals')
plt.show()
```

Slide 12: So sánh mẫu gốc và mẫu cải tiến

Bằng cách so sánh các biểu đồ phần dư của mô hình ban đầu và mô hình cải tiến, chúng ta có thể thấy sự cải thiện đáng kể trong việc phân bổ phần dư. Mô hình cải tiến cho thấy độ phân tán ngẫu nhiên hơn quanh mức 0, cho thấy mức độ phù hợp tốt hơn với dữ liệu.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original model residuals
ax1.hist(residuals, bins=20, edgecolor='black')
ax1.set_title('Original Model Residuals')
ax1.set_xlabel('Residual Value')
ax1.set_ylabel('Frequency')

# Improved model residuals
ax2.hist(residuals_improved, bins=20, edgecolor='black')
ax2.set_title('Improved Model Residuals')
ax2.set_xlabel('Residual Value')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

Slide 13: Ví dụ thực tế: Dự đoán giá nhà

Hãy xem xét một ví dụ thực tế khác: dự đoán giá nhà dựa trên mét vuông. Ví dụ này cho thấy cách phân tích phần dư có thể cho thấy sự cần thiết của các yếu tố dự đoán bổ sung hoặc các thuật ngữ phi tuyến tính trong mô hình.

```python
# Simulated house price data
np.random.seed(42)
sqft = np.linspace(1000, 5000, 200)
price = 100000 + 150 * sqft + 0.05 * sqft**2 + np.random.normal(0, 50000, 200)

# Linear regression
coeffs = np.polyfit(sqft, price, 1)
price_pred = np.polyval(coeffs, sqft)
residuals = price - price_pred

# Residual plot
plt.scatter(sqft, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Square Footage (House Prices)')
plt.xlabel('Square Footage')
plt.ylabel('Residuals')
plt.show()
```

Slide 14: Giải thích phần dư của mô hình giá nhà

Biểu đồ phần dư cho mô hình giá nhà của chúng tôi thể hiện một mô hình bậc hai rõ ràng, cho thấy rằng mô hình tuyến tính đơn giản là không đủ. Điều này cho thấy mối quan hệ giữa mét vuông và giá cả là phi tuyến tính, có thể do các yếu tố như vị trí hoặc động lực thị trường nhà ở.

```python
# Improved model with quadratic term
coeffs_quad = np.polyfit(sqft, price, 2)
price_pred_quad = np.polyval(coeffs_quad, sqft)
residuals_quad = price - price_pred_quad

# Residual plot for improved model
plt.scatter(sqft, residuals_quad)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Square Footage (Improved Model)')
plt.xlabel('Square Footage')
plt.ylabel('Residuals')
plt.show()
```

Trang trình bày 15: Kết luận và các phương pháp hay nhất

Biểu đồ phân phối phần dư là công cụ mạnh mẽ để đánh giá hiệu suất hồi quy tuyến tính. Chúng giúp xác định những vi phạm đối với các giả định của mô hình và hướng dẫn cải tiến. Các phương pháp hay nhất bao gồm:

1. Luôn vẽ đồ thị phần dư theo các giá trị dự đoán và các biến độc lập.
2. Sử dụng đồ thị Q-Q để đánh giá tính chuẩn.
3. Xem xét các phép biến đổi hoặc các yếu tố dự đoán bổ sung khi phần dư hiển thị các mẫu.
4. Hãy nhớ rằng một đồ thị dư tốt không đảm bảo một mô hình hoàn hảo, nhưng một đồ thị xấu hầu như luôn chỉ ra vấn đề.

Bằng cách kết hợp phân tích phần dư vào quy trình hồi quy của mình, bạn có thể xây dựng các mô hình chính xác và đáng tin cậy hơn, mang lại những hiểu biết và dự đoán tốt hơn.

Trang trình bày 16: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về phân tích dư lượng và chẩn đoán hồi quy tuyến tính, các tài nguyên sau được khuyến nghị:

1. Gelman, A., & Hill, J. (2006). Phân tích dữ liệu bằng cách sử dụng mô hình hồi quy và đa cấp/phân cấp. Nhà xuất bản Đại học Cambridge.
2. Cook, R. D., & Weisberg, S. (1982). Phần dư và ảnh hưởng trong hồi quy. Chapman và Hall.
3. Bài báo ArXiv: "Các sơ đồ chẩn đoán chất lượng của mô hình hồi quy tuyến tính" của M. Friendly và D. Denis. Có tại: [https://arxiv.org/abs/stat.AP/0406049](https://arxiv.org/abs/stat.AP/0406049)

Những tài nguyên này cung cấp các cuộc thảo luận chuyên sâu về lý thuyết và ứng dụng phân tích số dư trong hồi quy tuyến tính và các mô hình thống kê khác.
