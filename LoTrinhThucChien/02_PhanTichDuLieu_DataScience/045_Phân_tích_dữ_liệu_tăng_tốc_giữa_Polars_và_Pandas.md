## Phân tích tăng tốc dữ liệu giữa Polars và Pandas

Trang trình bày 1: Giới thiệu về Cực và Cấu trúc

Polars và Pandas đều là những thư viện thể thao mạnh dữ liệu mạnh mẽ trong Python. Trong khi Pandas đã là thư viện được sử dụng để phân tích dữ liệu trong nhiều năm thì Polars là một ứng cử viên mới hơn hứa hẹn sẽ cải thiện hiệu đáng kể. Bài trình bày này sẽ so sánh hai thư viện này, tập trung vào điểm mạnh, sự khác biệt của chúng và cách Polars có thể tăng tốc độ phân tích dữ liệu nhiệm vụ của bạn.

```python
import polars as pl
import pandas as pd

# Create sample dataframes
polars_df = pl.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
pandas_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

print("Polars DataFrame:")
print(polars_df)
print("\nPandas DataFrame:")
print(pandas_df)
```

Slide 2: So sánh hiệu suất

Một trong những ưu điểm chính của Polars so với Pandas là biểu tượng vượt trội. Polars được xây dựng trên Apache Arrow và được thiết kế để tiết kiệm bộ nhớ cũng như tận dụng các kiến ​​trúc CPU hiện đại. Điều này thường dẫn đến hoạt động nhanh hơn đáng kể, đặc biệt đối với các dữ liệu lớn.

```python
import time
import polars as pl
import pandas as pd
import numpy as np

# Generate large dataset
size = 1_000_000
data = {'A': np.random.rand(size), 'B': np.random.rand(size)}

# Pandas
start = time.time()
df_pandas = pd.DataFrame(data)
result_pandas = df_pandas.groupby('A').agg({'B': ['mean', 'max']})
pandas_time = time.time() - start

# Polars
start = time.time()
df_polars = pl.DataFrame(data)
result_polars = df_polars.groupby('A').agg([
    pl.col('B').mean(),
    pl.col('B').max()
])
polars_time = time.time() - start

print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Polars time: {polars_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")
```

Trang trình bày 3: Hiệu ứng bộ nhớ

Polars được thiết kế để tiết kiệm bộ nhớ, điều này rất quan trọng khi làm việc với các dữ liệu lớn. Nó sử dụng Apache Arrow làm bộ nhớ mô hình, cho phép thực hiện các hoạt động không sao chép và sử dụng kết quả hiệu quả bộ nhớ.

```python
import polars as pl
import pandas as pd
import sys

# Create large dataframes
size = 1_000_000
data = {'A': range(size), 'B': range(size)}

df_pandas = pd.DataFrame(data)
df_polars = pl.DataFrame(data)

# Check memory usage
pandas_memory = sys.getsizeof(df_pandas)
polars_memory = sys.getsizeof(df_polars)

print(f"Pandas DataFrame size: {pandas_memory / 1e6:.2f} MB")
print(f"Polars DataFrame size: {polars_memory / 1e6:.2f} MB")
print(f"Memory reduction: {pandas_memory / polars_memory:.2f}x")
```

Trang trình bày 4: Đánh giá lười biếng ở các vùng cực

Polars giới thiệu khái niệm đánh giá lười biếng, cho phép tối ưu hóa truy vấn trước khi thực hiện. Điều này có thể dẫn đến những cải tiến đáng kể, đặc biệt đối với các hoạt động phức tạp trên các tập dữ liệu lớn.

```python
import polars as pl

# Create a large dataframe
df = pl.DataFrame({'A': range(1_000_000), 'B': range(1_000_000)})

# Define a lazy computation
lazy_result = (
    df.lazy()
    .filter(pl.col('A') > 500_000)
    .groupby('B')
    .agg([pl.col('A').mean().alias('A_mean')])
    .sort('A_mean', descending=True)
    .limit(10)
)

# Execute the lazy computation
result = lazy_result.collect()
print(result)
```

Slide 5: Kiểu dữ liệu và lược đồ

Polars use the Lược đồ được định kiểu mạnh mẽ, điều này góp phần mang lại lợi ích về hiệu suất của nó. Nó hỗ trợ nhiều loại dữ liệu, bao gồm tất cả các loại thời gian và cho phép thao tác lược đồ một cách dễ dàng.

```python
import polars as pl

# Create a dataframe with various data types
df = pl.DataFrame({
    'int_col': [1, 2, 3],
    'float_col': [1.1, 2.2, 3.3],
    'str_col': ['a', 'b', 'c'],
    'date_col': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'bool_col': [True, False, True]
})

# Convert date_col to Date type
df = df.with_columns(pl.col('date_col').cast(pl.Date))

# Print schema
print(df.schema)

# Change data type of float_col to integer
df = df.with_columns(pl.col('float_col').cast(pl.Int64))

# Print updated schema
print(df.schema)
```

Slide 6: Thiếu dữ liệu xử lý

Cả Polars và Pandas đều cung cấp các dữ liệu xử lý phương pháp bị thiếu, nhưng Polars cung cấp một số tính năng độc đáo. Ví dụ: Polars use Không có để hiển thị các giá trị nhưng thiếu cho tất cả các loại dữ liệu, có thể trực quan hơn NaN của Pandas cho các số loại.

```python
import polars as pl
import pandas as pd
import numpy as np

# Create dataframes with missing values
polars_df = pl.DataFrame({
    'A': [1, None, 3],
    'B': [4.0, 5.0, None],
    'C': ['x', None, 'z']
})

pandas_df = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [4.0, 5.0, np.nan],
    'C': ['x', None, 'z']
})

print("Polars DataFrame:")
print(polars_df)
print("\nPandas DataFrame:")
print(pandas_df)

# Fill missing values
polars_filled = polars_df.fill_null(strategy='forward')
pandas_filled = pandas_df.fillna(method='ffill')

print("\nPolars Filled:")
print(polars_filled)
print("\nPandas Filled:")
print(pandas_filled)
```

Slide 7: Phân nhóm và tổng hợp

Cả Polars và Pandas đều cung cấp khả năng nhóm và tổng hợp mạnh mẽ, nhưng cú pháp của Polars có thể biểu hiện rõ hơn và hiệu suất của nó thường vượt trội hơn.

```python
import polars as pl
import pandas as pd
import numpy as np

# Create sample data
data = {
    'category': ['A', 'B', 'A', 'B', 'A', 'B'] * 1000,
    'value1': np.random.rand(6000),
    'value2': np.random.rand(6000)
}

# Polars
df_polars = pl.DataFrame(data)
result_polars = df_polars.groupby('category').agg([
    pl.col('value1').mean().alias('value1_mean'),
    pl.col('value2').sum().alias('value2_sum'),
    pl.col('value1').count().alias('count')
])

# Pandas
df_pandas = pd.DataFrame(data)
result_pandas = df_pandas.groupby('category').agg({
    'value1': 'mean',
    'value2': 'sum',
    'value1': 'count'
}).rename(columns={'value1': 'value1_mean', 'value2': 'value2_sum', 'value1': 'count'})

print("Polars result:")
print(result_polars)
print("\nPandas result:")
print(result_pandas)
```

Trang trình bày 8: Tham gia DataFrames

Các dữ liệu khung này là một biến phổ hoạt động trong dữ liệu phân tích. Cả Polars và Pandas đều hỗ trợ nhiều loại liên kết khác nhau, nhưng Polars thường thực hiện các thao tác này nhanh hơn, đặc biệt là trên các dữ liệu lớn.

```python
import polars as pl
import pandas as pd
import time

# Create sample dataframes
df1 = pl.DataFrame({'key': range(1000000), 'value_a': range(1000000)})
df2 = pl.DataFrame({'key': range(500000, 1500000), 'value_b': range(1000000)})

# Polars join
start = time.time()
result_polars = df1.join(df2, on='key', how='inner')
polars_time = time.time() - start

# Convert to Pandas for comparison
pdf1 = df1.to_pandas()
pdf2 = df2.to_pandas()

# Pandas join
start = time.time()
result_pandas = pdf1.merge(pdf2, on='key', how='inner')
pandas_time = time.time() - start

print(f"Polars join time: {polars_time:.2f} seconds")
print(f"Pandas join time: {pandas_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")
```

Slide 9: Xử lý thời gian chuỗi dữ liệu

Dữ liệu chuỗi thời gian được phổ biến trong nhiều lĩnh vực. Cả Polars và Pandas đều cung cấp các chức năng để làm việc với dữ liệu thời gian, nhưng hiệu suất của Polars có thể đặc biệt có lợi cho các chuỗi dữ liệu lớn trong thời gian.

```python
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate time series data
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(365)]
values = np.random.randn(365)

# Polars
df_polars = pl.DataFrame({'date': dates, 'value': values})
df_polars = df_polars.with_columns(pl.col('date').cast(pl.Date))

# Resample to monthly frequency and calculate mean
result_polars = df_polars.groupby_dynamic('date', every='1mo').agg([
    pl.col('value').mean().alias('monthly_mean')
])

# Pandas
df_pandas = pd.DataFrame({'date': dates, 'value': values})
df_pandas.set_index('date', inplace=True)

# Resample to monthly frequency and calculate mean
result_pandas = df_pandas.resample('M').mean()

print("Polars result:")
print(result_polars)
print("\nPandas result:")
print(result_pandas)
```

Slide 10: Thao tác trên chuỗi

Hiệu ứng chuỗi hoạt động rất quan trọng để xử lý văn bản dữ liệu. Polars cung cấp các hàm thao tác chuỗi nhanh có thể tăng tốc đáng kể các tính toán dựa trên văn bản.

```python
import polars as pl
import pandas as pd
import time

# Create a large dataframe with string data
size = 1_000_000
data = {'text': ['Hello world! ' * 10] * size}

# Polars
df_polars = pl.DataFrame(data)
start = time.time()
result_polars = df_polars.with_columns(
    pl.col('text').str.to_uppercase().alias('upper'),
    pl.col('text').str.contains('world').alias('contains_world'),
    pl.col('text').str.length().alias('length')
)
polars_time = time.time() - start

# Pandas
df_pandas = pd.DataFrame(data)
start = time.time()
result_pandas = df_pandas.copy()
result_pandas['upper'] = df_pandas['text'].str.upper()
result_pandas['contains_world'] = df_pandas['text'].str.contains('world')
result_pandas['length'] = df_pandas['text'].str.len()
pandas_time = time.time() - start

print(f"Polars time: {polars_time:.2f} seconds")
print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")
```

Slide 11: Xử lý bộ dữ liệu lớn

Một trong những điểm mạnh chính của Polars là khả năng xử lý các dữ liệu lớn có thể không phù hợp với bộ nhớ. Khả năng xử lý bên ngoài lõi của nó cho phép nó hoạt động với các bộ dữ liệu lớn hơn RAM có sẵn.

```python
import polars as pl
import os

# Function to generate a large CSV file
def generate_large_csv(filename, size):
    with open(filename, 'w') as f:
        f.write('id,value\n')
        for i in range(size):
            f.write(f'{i},{i*2}\n')

# Generate a 1GB CSV file
filename = 'large_file.csv'
generate_large_csv(filename, 50_000_000)

# Read and process the file using Polars' lazy evaluation
df = pl.scan_csv(filename)
result = df.filter(pl.col('value') > 1_000_000).select([
    pl.col('id'),
    pl.col('value'),
    (pl.col('value') * 2).alias('double_value')
]).collect()

print(result.head())

# Clean up
os.remove(filename)
```

Trang trình bày 12: Ví dụ thực tế: Phân tích dữ liệu thời gian

Vui lòng so sánh Polars và Pandas trong một vấn đề thực tế: phân tích một tập dữ liệu lớn. Chúng tôi sẽ thực hiện một số thao tác phổ biến như lọc, nhóm và tổng hợp.

```python
import polars as pl
import pandas as pd
import numpy as np
import time

# Generate synthetic weather data
size = 1_000_000
dates = pd.date_range('2020-01-01', periods=size)
cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], size)
temperatures = np.random.normal(20, 10, size)
humidity = np.random.uniform(30, 80, size)

# Polars
start = time.time()
df_polars = pl.DataFrame({
    'date': dates,
    'city': cities,
    'temperature': temperatures,
    'humidity': humidity
})

result_polars = (
    df_polars.filter(pl.col('temperature') > 25)
    .groupby(['city', pl.col('date').dt.month()])
    .agg([
        pl.col('temperature').mean().alias('avg_temp'),
        pl.col('humidity').mean().alias('avg_humidity')
    ])
    .sort(['city', 'date'])
)
polars_time = time.time() - start

# Pandas
start = time.time()
df_pandas = pd.DataFrame({
    'date': dates,
    'city': cities,
    'temperature': temperatures,
    'humidity': humidity
})

result_pandas = (
    df_pandas[df_pandas['temperature'] > 25]
    .groupby(['city', df_pandas['date'].dt.month])
    .agg({'temperature': 'mean', 'humidity': 'mean'})
    .reset_index()
    .sort_values(['city', 'date'])
)
pandas_time = time.time() - start

print(f"Polars time: {polars_time:.2f} seconds")
print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")

print("\nPolars result (first 5 rows):")
print(result_polars.head())
print("\nPandas result (first 5 rows):")
print(result_pandas.head())
```

Slide 13: Ví dụ thực tế: Xử lý văn bản

Xử lý văn bản là một nhiệm vụ phổ biến trong dữ liệu phân tích. Vui lòng so sánh Polars và Pandas trong một vấn đề mà chúng tôi cần phân tích tần số từ trong một kho văn bản lớn.

```python
import polars as pl
import pandas as pd
import time
import re

# Generate a large corpus of text
corpus = " ".join(["The quick brown fox jumps over the lazy dog"] * 100000)

# Polars
start = time.time()
df_polars = pl.DataFrame({'text': [corpus]})
words_polars = (
    df_polars.select(pl.col('text').str.to_lowercase().str.split_whitespace())
    .explode('text')
    .select(pl.col('text').str.replace_all(r'[^\w\s]', '').alias('word'))
    .filter(pl.col('word') != '')
    .groupby('word')
    .count()
    .sort('count', descending=True)
    .limit(10)
)
polars_time = time.time() - start

# Pandas
start = time.time()
df_pandas = pd.DataFrame({'text': [corpus]})
words_pandas = (
    df_pandas['text'].str.lower()
    .str.split()
    .explode()
    .str.replace(r'[^\w\s]', '', regex=True)
    .value_counts()
    .reset_index()
    .rename(columns={'index': 'word', 'text': 'count'})
    .head(10)
)
pandas_time = time.time() - start

print(f"Polars time: {polars_time:.2f} seconds")
print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")

print("\nPolars result:")
print(words_polars)
print("\nPandas result:")
print(words_pandas)
```

Slide 14: Kết luận

Trong suốt phần trình bày này, chúng tôi đã khám phá những điểm khác biệt chính giữa Polars và Pandas. Polars cung cấp những cải tiến đáng kể về hiệu suất, đặc biệt đối với các dữ liệu lớn, nhờ khả năng quản lý hiệu quả bộ nhớ và tận dụng các kiến ​​trúc CPU hiện đại. Tính chiến đấu lười biếng của nó cho phép truy vấn tối ưu hóa, nâng cao hơn nữa hiệu suất cho các hoạt động phức tạp.

Trong khi Pandas vẫn là một thư mạnh mẽ và được sử dụng rộng rãi với hệ sinh thái hoàn thiện, Polars đưa ra một giải pháp thay thế hấp dẫn cho các nhà khoa học và nhà phân tích dữ liệu làm việc với các tập dữ liệu lớn hoặc yêu cầu tính toán hiệu suất cao. Lựa chọn giữa Polars và Pandas tùy thuộc vào trường sử dụng cụ thể, kích thước tệp dữ liệu và yêu cầu hiệu suất của bạn.

```python
import polars as pl
import pandas as pd
import numpy as np
import time

# Generate a large dataset
size = 10_000_000
data = {'A': np.random.rand(size), 'B': np.random.rand(size)}

# Polars
start = time.time()
df_polars = pl.DataFrame(data)
result_polars = df_polars.filter(pl.col('A') > 0.5).groupby('B').agg(pl.col('A').mean())
polars_time = time.time() - start

# Pandas
start = time.time()
df_pandas = pd.DataFrame(data)
result_pandas = df_pandas[df_pandas['A'] > 0.5].groupby('B')['A'].mean().reset_index()
pandas_time = time.time() - start

print(f"Polars time: {polars_time:.2f} seconds")
print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Overall speedup: {pandas_time / polars_time:.2f}x")
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu thêm về Polars và các khả năng của nó, đây là một số tài nguyên có giá trị:

1. Tài liệu về Polars: [https://pola-rs.github.io/Polars-book/](https://pola-rs.github.io/pola-book/)
2. Kho lưu trữ Polars GitHub: [https://github.com/pola-rs/Polars](https://github.com/pola-rs/Polars)
3. "Polars: Thư viện DataFrame nhanh như chớp cho Rust và Python" (arXiv:2211.14502): [https://arxiv.org/abs/2211.14502](https://arxiv.org/abs/2211.14502)
4. Tài liệu về Pandas (để so sánh): [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

Tài nguyên này cung cấp thông tin chuyên sâu về các tính năng, hiệu suất đặc biệt và ví dụ về cách sử dụng của Polars. Bài viết arXiv cung cấp cái nhìn tổng quan về thiết bị kỹ thuật về các tiêu chuẩn hiệu suất và thiết kế của Polars.
