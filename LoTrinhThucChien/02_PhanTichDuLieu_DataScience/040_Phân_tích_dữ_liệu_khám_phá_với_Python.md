## Phân tích dữ liệu khám phá bằng Python
Trang trình bày 1: Giới thiệu về Phân tích dữ liệu thăm dò (EDA)

Khám phá phân tích dữ liệu là một bước quan trọng trong quy trình nghiên cứu dữ liệu, cho phép chúng tôi hiểu cấu trúc, mẫu và đặc điểm của dữ liệu trước khi thiết lập mô hình chính. EDA giúp chúng tôi xác định xu hướng, phát hiện các ngoại lệ và hình thành các giả thuyết về dữ liệu.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load a sample dataset
df = pd.read_csv('sample_data.csv')

# Display basic information about the dataset
print(df.info())

# Show the first few rows
print(df.head())
```

Slide 2: Tải và kiểm tra dữ liệu

Bước đầu tiên trong EDA là tải dữ liệu và xem tổng quan nhanh. Chúng tôi sẽ sử dụng gấu trúc để tải tệp CSV và hiển thị cơ sở dữ liệu thông tin.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('iris.csv')

# Display basic information
print(df.info())

# Show the first few rows
print(df.head())

# Display summary statistics
print(df.describe())
```

Slide 3: Xử lý các giá trị bị thiếu

Xác định và xử lý các giá trị còn thiếu là rất quan trọng trong EDA. Chúng tôi sẽ khám phá các cách thiếu dữ liệu được phát hiện và trực tuyến hóa.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load a dataset with missing values
df = pd.read_csv('dataset_with_missing_values.csv')

# Calculate percentage of missing values
missing_percentage = df.isnull().mean() * 100

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

print("Percentage of missing values:\n", missing_percentage)
```

Slide 4: Phân tích dữ liệu phân tích

Hiểu biết về việc phân phối các biến là điều cần thiết. Chúng tôi sẽ sử dụng biểu đồ và biểu tượng hạt nhân mật khẩu để trực tiếp hóa các phân bố.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('iris.csv')

# Create histograms for numerical columns
df.hist(figsize=(12, 8))
plt.suptitle('Histograms of Numerical Variables')
plt.tight_layout()
plt.show()

# Create kernel density plots
plt.figure(figsize=(12, 8))
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    sns.kdeplot(data=df[column], shade=True, label=column)
plt.title('Kernel Density Plots of Numerical Variables')
plt.legend()
plt.show()
```

Slide 5: Phân tích tương quan

Khám phá mối quan hệ giữa các biến là rất quan trọng. Chúng ta sẽ sử dụng ma trận tương quan và bản đồ nhiệt để trực tiếp hóa quan hệ các mối quan hệ này.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Calculate correlation matrix
corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for visualizing relationships
sns.pairplot(df, hue='species')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()
```

Trang trình bày 6: Ngoại lệ phát triển

Việc xác định các ngoại lệ rất quan trọng để hiểu được chất lượng của dữ liệu và các ẩn ẩn bất ngờ. Chúng tôi sẽ sử dụng biểu tượng hộp và biểu đồ phân tán để phát hiện ngoại lệ.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('iris.csv')

# Create box plots
plt.figure(figsize=(12, 6))
df.boxplot()
plt.title('Box Plots for Numerical Variables')
plt.show()

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot: Sepal Length vs Sepal Width')
plt.show()
```

Slide 7: Phân tích loại dữ liệu

Việc phân tích các loại biến là điều cần thiết để hiểu các thành phần của dữ liệu của chúng tôi. Chúng tôi sẽ sử dụng biểu đồ và biểu đồ tròn cho mục đích này.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Create a bar plot
plt.figure(figsize=(10, 6))
df['species'].value_counts().plot(kind='bar')
plt.title('Distribution of Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Create a pie chart
plt.figure(figsize=(8, 8))
df['species'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Iris Species')
plt.ylabel('')
plt.show()
```

Slide 8: Phân tích chuỗi thời gian

Đối với thời gian chuỗi dữ liệu, chúng tôi cần phân tích xu hướng, tính thời vụ và mô hình theo thời gian. Chúng tôi sẽ sử dụng biểu đồ và cuộn thống kê dữ liệu cho mục tiêu này.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load a time series dataset
df = pd.read_csv('time_series_data.csv', parse_dates=['date'], index_col='date')

# Plot the time series
plt.figure(figsize=(12, 6))
df['value'].plot()
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Calculate and plot rolling mean and standard deviation
rolling_mean = df['value'].rolling(window=30).mean()
rolling_std = df['value'].rolling(window=30).std()

plt.figure(figsize=(12, 6))
df['value'].plot(label='Original')
rolling_mean.plot(label='Rolling Mean', color='red')
rolling_std.plot(label='Rolling Std', color='green')
plt.title('Time Series with Rolling Statistics')
plt.legend()
plt.show()
```

Trang trình bày 9: Kỹ thuật tính năng

Kỹ thuật tính năng là quá trình tạo ra các tính năng mới từ những tính năng hiện có. Chúng tôi sẽ trình bày cách tạo các thuật ngữ tương tác và các tính năng đa thức.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
df = pd.read_csv('iris.csv')

# Create interaction terms
df['sepal_area'] = df['sepal_length'] * df['sepal_width']
df['petal_area'] = df['petal_length'] * df['petal_width']

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['sepal_length', 'sepal_width']])
poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(['sepal_length', 'sepal_width']))

# Combine original and new features
df_engineered = pd.concat([df, poly_features_df], axis=1)

print(df_engineered.head())
```

Slide 10: Giảm kích thước

Khi xử lý dữ liệu nhiều chiều, các kỹ thuật giảm kích thước như PCA có thể hữu ích cho việc trực quan hóa và lựa chọn tính năng.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('species', axis=1))

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Plot the results
plt.figure(figsize=(10, 8))
for species in df['species'].unique():
    mask = df['species'] == species
    plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=species)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.title('PCA of Iris Dataset')
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 11: Kiểm tra thống kê

Việc thực hiện danh sách thống kê bài kiểm tra có thể giúp chúng tôi hiểu được tầm quan trọng của những phát hiện của chúng tôi. Chúng tôi sẽ trình bày cách thực hiện bài kiểm tra và bài kiểm tra chi bình phương.

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
df = pd.read_csv('iris.csv')

# Perform t-test
setosa = df[df['species'] == 'setosa']['sepal_length']
versicolor = df[df['species'] == 'versicolor']['sepal_length']
t_stat, p_value = stats.ttest_ind(setosa, versicolor)

print("T-test results:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Perform chi-square test
observed = pd.crosstab(df['species'], df['sepal_length'] > df['sepal_length'].mean())
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print("\nChi-square test results:")
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p_value}")
```

Slide 12: Dữ liệu trực quan với Seaborn

Seaborn là một thư mạnh mẽ để trực tuyến hóa dữ liệu thống kê. Chúng tôi sẽ sử dụng nó để tạo các bản nâng cấp sơ đồ cao hơn cho EDA của mình.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Create a pair plot
sns.pairplot(df, hue='species', height=2.5)
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.show()

# Create a violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='sepal_length', data=df)
plt.title('Violin Plot of Sepal Length by Species')
plt.show()

# Create a joint plot
sns.jointplot(x='sepal_length', y='sepal_width', data=df, kind='kde', hue='species')
plt.suptitle('Joint Plot of Sepal Length vs Sepal Width', y=1.02)
plt.show()
```

Trang trình bày 13: Ví dụ thực tế: Phân tích dữ liệu thời gian

Vui lòng phân tích dữ liệu thời gian để chứng minh EDA trong thế giới kịch bản thực tế. Chúng tôi sẽ khám phá xu hướng nhiệt độ và lượng mưa.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load weather data
weather_df = pd.read_csv('weather_data.csv', parse_dates=['date'])

# Plot temperature trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='temperature', data=weather_df)
plt.title('Temperature Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()

# Analyze precipitation patterns
plt.figure(figsize=(12, 6))
sns.boxplot(x=weather_df['date'].dt.month, y='precipitation', data=weather_df)
plt.title('Monthly Precipitation Distribution')
plt.xlabel('Month')
plt.ylabel('Precipitation (mm)')
plt.show()

# Correlation between temperature and precipitation
plt.figure(figsize=(10, 8))
sns.scatterplot(x='temperature', y='precipitation', data=weather_df)
plt.title('Temperature vs Precipitation')
plt.xlabel('Temperature (°C)')
plt.ylabel('Precipitation (mm)')
plt.show()
```

Trang trình bày 14: Ví dụ thực tế: Tỷ lệ phân chia bỏ khách hàng

Trong ví dụ này, chúng tôi sẽ khám phá tập dữ liệu liên quan đến tình trạng bỏ khách hàng trong một công ty Viễn thông, bằng chứng minh là EDA có thể cung cấp thông tin chuyên sâu về các vấn đề kinh doanh.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load customer churn data
churn_df = pd.read_csv('customer_churn_data.csv')

# Visualize churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=churn_df)
plt.title('Distribution of Customer Churn')
plt.show()

# Analyze relationship between tenure and churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=churn_df)
plt.title('Customer Tenure by Churn Status')
plt.show()

# Explore correlation between numerical features
correlation = churn_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về khám phá các kỹ thuật phân tích dữ liệu và các phương pháp hay nhất, hãy xem xét các tài nguyên sau:

1. "Khám phá dữ liệu phân tích" của John W. Tukey (1977) - Cuốn sách nền tảng về EDA.
2. "Python to parsing data" của Wes McKinney - Bao gồm gấu trúc và numpy để thao tác và phân tích dữ liệu.
3. "Khoa học dữ liệu từ đầu" của Joel Grus - Cung cấp phần giới thiệu toàn diện về các khái niệm khoa học dữ liệu.
4. "Suy nghĩ lại về thống kê" của Richard McElreath - Đưa ra quan điểm Bayes về phân tích dữ liệu.
5. Bài viết ArXiv: "Khảo sát về Kỹ thuật phân tích và trình bày dữ liệu khám phá cho dữ liệu lớn" ([https://arxiv.org/abs/2005.02218](https://arxiv.org/abs/2005.02218)) - Thảo luận về kỹ thuật EDA hiện đại cho các tập dữ liệu lớn.

Hãy nhớ điều chỉnh các kỹ thuật này cho phù hợp với dữ liệu và câu hỏi nghiên cứu của bạn. EDA là một quá trình lặp đi lặp lại và những hiểu biết sâu sắc thường được dẫn đến những câu hỏi và phân tích sâu hơn.
