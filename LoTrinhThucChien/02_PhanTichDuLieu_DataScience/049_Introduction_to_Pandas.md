## Giới thiệu về gấu trúc

Slide 1: Giới thiệu về Pandas

Pandas là thư viện Python mã nguồn mở mạnh mẽ để phân tích và thao tác dữ liệu. Nó cung cấp các cấu trúc dữ liệu và công cụ phân tích dữ liệu dễ sử dụng để làm việc với dữ liệu chuỗi thời gian và dữ liệu có cấu trúc (dạng bảng, đa chiều, có khả năng không đồng nhất).

Trang trình bày 2: Nhập gấu trúc

```python
import pandas as pd
```

This line imports the Pandas library and assigns it the conventional abbreviation 'pd'.

Slide 3: Series

A Pandas Series is a one-dimensional labeled array capable of holding any data type.

```python
data = pd.Series([1, 2, 3, 4, 5])
print(data)
```

Đầu ra:

```
0    1
1    2
2    3
3    4
4    5
dtype: int64
```

Slide 4: DataFrames

A Pandas DataFrame is a 2-dimensional labeled data structure, like a 2D array, with columns of potentially different data types.

```python
data = {'Name': ['John', 'Jane', 'Jim', 'Joan'],
        'Age': [25, 32, 19, 27]}
df = pd.DataFrame(data)
print(df)
```

Đầu ra:

```
   Name  Age
0  John   25
1  Jane   32
2   Jim   19
3  Joan   27
```

Slide 5: Reading Data

Pandas can read data from various file formats like CSV, Excel, SQL databases, and more.

```python
df = pd.read_csv('data.csv')
```

Slide 6: Lựa chọn dữ liệu

Việc chọn dữ liệu từ DataFrame thật dễ dàng với tính năng lập chỉ mục của Pandas.

```python
print(df['Name'])    # Select a column
print(df.loc[0])     # Select a row by label
print(df.iloc[0, 1]) # Select a value by row/column number
```

Slide 7: Data Manipulation

Pandas provides powerful tools for reshaping, merging, and cleaning data.

```python
df['Age_months'] = df['Age'] * 12  # Add a new column
df.dropna(inplace=True)             # Drop rows with missing values
df.rename(columns={'Age': 'Years'}, inplace=True) # Rename a column
```

Slide 8: Phân nhóm và tổng hợp

Nhóm và tổng hợp dữ liệu là một thao tác phổ biến trong phân tích dữ liệu.

```python
grouped = df.groupby('Name')['Age'].sum()
print(grouped)
```

Output:

```
Name
Jane    32
Jim     19
Joan    27
John    25
Name: Age, dtype: int64
```

Slide 9: Vẽ đồ thị

Pandas tích hợp tốt với Matplotlib và các thư viện trực quan hóa dữ liệu khác.

```python
import matplotlib.pyplot as plt
df.plot(kind='scatter', x='Age', y='Height')
plt.show()
```

Slide 11: Data Cleaning

Pandas provides utilities for cleaning and preprocessing data.

```python
import numpy as np

# Replace values
df['Age'].replace([19, 27], np.nan, inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
```

Slide 12: Sáp nhập và tham gia

Pandas giúp dễ dàng kết hợp các tập dữ liệu bằng cách hợp nhất và nối.

```python
# Merge two DataFrames
pd.merge(df1, df2, on='key', how='inner')

# Join on indexes
df1.join(df2, lsuffix='_left', rsuffix='_right')
```

Trang trình bày 13: Dữ liệu chuỗi thời gian

Pandas có sự hỗ trợ tuyệt vời để làm việc với dữ liệu chuỗi thời gian.

```python
# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set index
df = df.set_index('Date')

# Resample
df.resample('M').mean()
```

Slide 14: Xử lý bộ dữ liệu lớn

Pandas cung cấp các công cụ để xử lý hiệu quả các bộ dữ liệu lớn.

```python
# Chunking data
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process_data(chunk)

# Data types and memory usage
df.info(memory_usage='deep')
```

Slide 15: Tích hợp với các thư viện khác

Pandas tích hợp tốt với các thư viện khoa học dữ liệu khác trong Python.

```python
# NumPy for numerical operations
df['New_Col'] = np.sqrt(df['Col1'] ** 2 + df['Col2'] ** 2)

# Scikit-learn for machine learning
from sklearn.linear_model import LinearRegression
X = df[['Col1', 'Col2']]
y = df['Target']
model = LinearRegression().fit(X, y)
```

Các trang trình bày bổ sung này đề cập đến các chủ đề nâng cao hơn trong Pandas, chẳng hạn như làm sạch dữ liệu, hợp nhất và nối các tập dữ liệu, làm việc với dữ liệu chuỗi thời gian, xử lý các tập dữ liệu lớn và tích hợp Pandas với các thư viện Python khác như NumPy và Scikit-learn.

## Meta
Đây là tiêu đề, mô tả và hashtag cho TikTok về các nguyên tắc cơ bản của Pandas, với giọng điệu mang tính thể chế:

Làm chủ Pandas: Hướng dẫn toàn diện về phân tích dữ liệu

Nâng cao kỹ năng phân tích dữ liệu của bạn với Pandas, thư viện Python mạnh mẽ để thao tác và phân tích dữ liệu. Hướng dẫn toàn diện này bao gồm các nguyên tắc cơ bản của Pandas, cung cấp nền tảng vững chắc để làm việc với dữ liệu có cấu trúc.

Từ nhập dữ liệu đến làm sạch và tiền xử lý, hợp nhất các tập dữ liệu đến xử lý dữ liệu chuỗi thời gian, khóa học này trang bị cho bạn các công cụ và kỹ thuật cần thiết để khai thác toàn bộ tiềm năng dữ liệu của bạn. Tìm hiểu cách tận dụng cấu trúc dữ liệu trực quan của Pandas, thực hiện lựa chọn và thao tác dữ liệu cũng như hiểu rõ hơn thông qua việc nhóm, tổng hợp và trực quan hóa.

Cho dù bạn là nhà phân tích dữ liệu, nhà nghiên cứu hay chỉ đơn giản là đam mê khám phá dữ liệu, khóa học này được thiết kế để trang bị cho bạn kiến ​​thức và ví dụ thực tế để giải quyết các thách thức phân tích dữ liệu phức tạp. Hãy tham gia cùng chúng tôi trên hành trình này và mở khóa những khả năng mới trong nỗ lực dựa trên dữ liệu của bạn.

Hashtags: #PandasFundamentals #DataAnalysis #PythonLibrary #DataScience #DataManipulation #DataInsights #LearningOpportunity #SkillsForSuccess
